# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-process gRPC serving: owner + N worker gRPC servers + a thin front proxy.

Topology:
  - **owner** (this process): loads the model once, exports its weights via `ModelOwnerRegistry`
    (CUDA IPC), serves worker attach requests, and (as each worker's direct multiprocessing parent) is
    the authoritative OS liveness signal for its workers. After workers are up it runs the **proxy** gRPC
    server on the public port. Shutdown sends each worker an explicit neutral `STOP` over a one-way
    lifecycle channel; abnormal owner death is detected by the workers via the parent sentinel.
  - **N workers** (spawned): each attaches, builds a `Backend` and `alias_external_weights(...)` onto
    the owner's IPC tensors (zero-copy, 1× weights for the fleet), **preloads** that backend
    into a `SensorSimService.backend_cache`, and runs its own gRPC `SensorSimService` on a local port
    (one render at a time — `max_workers=1`). It refuses to start without the owner's parent-process death
    sentinel, then waits on that sentinel and the lifecycle channel: sentinel readiness means the owner
    died, a lifecycle `STOP` is an owner-directed graceful stop.
  - **proxy**: a `SensorsimServiceServicer` that **round-robins the render RPCs** across the worker stubs
    (so N concurrent requests fan out to N workers rendering in parallel) and forwards discovery RPCs.

The GIL escape only yields GPU concurrency with **MPS** running. Point any sensorsim gRPC
client (e.g. `concurrent_render_bench`) at the proxy's `--host:--port`.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import faulthandler
import functools
import gc
import itertools
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
import traceback

from concurrent import futures
from dataclasses import dataclass, field, replace
from multiprocessing import parent_process
from multiprocessing.connection import wait as wait_for_ready_connections
from pathlib import Path
from typing import Any, Callable, cast

import grpc
import torch
import torch.multiprocessing as mp
import yaml

from grpc_health.v1 import health, health_pb2, health_pb2_grpc  # type: ignore

from nurec.artifact.artifact import Artifact
from nurec.config.model import RendererBackend
from nurec.grpc.cache_keys import BackendCacheKey, SceneCacheKey
from nurec.grpc.downloader import Downloader, check_safe_scene_id
from nurec.grpc.grpc_server_config import GrpcServerConfig
from nurec.grpc.model_owner import (
    AttachSceneRequest,
    ModelOwnerRegistry,
    OwnerResponse,
    WorkerLifecycleCommand,
    WorkerLifecycleError,
    canonical_model,
    handle_owner_request,
    recv_worker_lifecycle,
    send_worker_stop,
)
from nurec.grpc.protos import common_pb2, sensorsim_pb2
from nurec.grpc.protos.sensorsim_pb2 import EditAssetsResponse
from nurec.grpc.protos.sensorsim_pb2_grpc import (
    SensorsimServiceServicer,
    SensorsimServiceStub,
    add_SensorsimServiceServicer_to_server,
)
from nurec.grpc.scene_cache import SceneCache, SceneCacheEviction
from nurec.grpc.serve import (
    Backend,
    BackendInUseError,
    MetricsInterceptor,
    SensorSimService,
    ServerMetricsManager,
    allowed_image_format_values,
    is_local_scene_uri,
    require_allowed_image_format,
    resolve_scene_artifact_path,
)
from nurec.models.gaussians.context_composite import alias_external_weights
from nurec.utils import mps_prelaunch
from nurec.utils.metrics import MetricsCollector, create_metric_sample


log = logging.getLogger("nurec.grpc.multiproc_serve")

# 4K JPEG frames exceed gRPC's default 4 MB cap; raise send/recv on every hop.
_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]
_RAY_CHUNK_SIZE = 1 << 31  # effectively "do not chunk rays"
_SHUTDOWN_GRACE_S = 2.0  # let any in-flight render drain before the worker stops

# Worker bring-up (start_worker_pool): bounds + poll cadence for the spawn->attach->ready handshake.
# The waits poll at `_BRINGUP_POLL_INTERVAL_S` so they notice a worker that died (no message coming) or a
# sibling's failure (the shared cancel event) promptly, instead of blocking for the full timeout.
_WORKER_ATTACH_REQ_TIMEOUT_S = 300.0  # bound on a spawned worker sending its initial attach request
_WORKER_ATTACH_RESP_TIMEOUT_S = (
    300.0  # worker-side bound on awaiting the owner's attach response (mirrors the request budget)
)
_WORKER_READY_TIMEOUT_S = 600.0  # bound on a worker attaching + binding its gRPC port after spawn
_BRINGUP_POLL_INTERVAL_S = 0.5  # liveness/cancel poll cadence while awaiting a worker message

# Runtime scene attach/detach: the owner pushes a scene's IPC-shared weights to a worker over
# a torch.multiprocessing queue; the worker builds + aliases a Backend for
# that scene and caches it by scene_id (so one running worker can serve N scenes). Commands are
# (kind, seq, *payload) and acks are (seq, status, detail), reusing the seq-correlated ack draining.
_SCENE_ATTACH = "scene_attach"
_SCENE_DETACH = "scene_detach"
_SCENE_STOP = "scene_stop"
_SCENE_ATTACH_ACK_TIMEOUT_S = 300.0  # attach loads the structure + aliases; match the startup attach budget
_SCENE_DETACH_ACK_TIMEOUT_S = 30.0  # detach just drops the worker's mapping
# Detach is idempotent; retry a lost/uncertain ack a bounded number of times (each attempt bounded by the
# detach ack timeout) before quarantining the scene and failing the fleet closed.
_SCENE_DETACH_MAX_ATTEMPTS = 3

# Admission control / OOM policy (fleet-wide, enforced at the front proxy). All default OFF so
# behavior is unchanged unless an operator opts in.
_DEFAULT_MAX_IN_FLIGHT = 0  # cap on concurrent in-flight renders across the pool (0 = unlimited)
_DEFAULT_ADMISSION_TIMEOUT_S = 0.0  # how long an over-limit render queues for a slot (0 = shed at once)
# Cadence for re-checking client cancellation/deadline while a render is queued for an admission slot, so a
# client that goes away frees its handler thread promptly instead of pinning it for the full admission wait.
_ADMISSION_CANCEL_POLL_INTERVAL_S = 0.1
# Upper bound separating a real forwardable client deadline from grpc's inf-future "no deadline" sentinel.
# `ServicerContext.time_remaining()` is `max(deadline - now, 0)` and never None; a deadline-less client
# surfaces as ~9.2e18 s, which must NOT be forwarded (it overflows grpc's deadline math into an immediate
# DEADLINE_EXCEEDED). 1 year is far above any real render deadline and far below the inf-future sentinel.
_MAX_FORWARDABLE_DEADLINE_S = 365 * 24 * 60 * 60.0
_PROXY_HEALTH_SERVICE_NAMES = ("", "nurec.grpc.protos.sensorsim.SensorsimService")


def _await_seq_ack(ack_q: Any, seq: int, timeout_s: float, *, timeout_status: str = "error") -> tuple[str, Any]:
    """Read the ack tagged ``seq`` from ``ack_q``, discarding stale acks from earlier commands.

    Per-worker acks are FIFO, so a late ack from a command that previously timed out (after which the
    worker recovered and processed it) carries an older seq; we drop those until we see ``seq`` (or time
    out). This seq correlation is what keeps a recovered worker from desyncing later commands. Used by the
    runtime scene-attach channel.

    A deadline (no matching ack) returns ``(timeout_status, detail)``. Callers that must distinguish an
    **uncertain** timeout (the worker may have applied the command) from an **explicit** worker error pass
    ``timeout_status="timeout"``. A worker's own status (``ok``/``error``) is always returned verbatim.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return timeout_status, f"timed out waiting for ack (seq={seq})"
        try:
            ack_seq, status, detail = ack_q.get(timeout=remaining)
        except queue.Empty:
            return timeout_status, f"timed out waiting for ack (seq={seq})"
        except Exception as exc:
            # A queue transport fault is NOT a plain timeout: surface the exception so a broken ack channel
            # is not misreported as "timed out" (which reads as a slow-but-healthy worker).
            return timeout_status, f"ack queue fault waiting for ack (seq={seq}): {exc!r}"
        if ack_seq == seq:
            return status, detail
        if ack_seq > seq:
            # Protocol invariant: each channel has at most one outstanding command (per-channel
            # serialization), so an ack for a LATER seq than the one we await should be
            # impossible. Surface it (a desync / future-pipelining bug) instead of silently discarding it,
            # but keep waiting so a correct ack can still arrive within the timeout.
            log.warning(
                f"_await_seq_ack: unexpected future ack (ack_seq={ack_seq} > seq={seq}); the channel expects "
                "one outstanding command at a time — discarding and continuing to wait"
            )
        # else ack_seq < seq: stale ack from a previously timed-out command — discard and keep waiting.


def _scene_attach_listener(
    attached: "dict[str, Any]",
    build_backend: "Callable[[str, str, dict], Any]",
    scene_q: Any,
    scene_ack_q: Any,
) -> None:
    """Worker thread: attach/detach scenes on demand from owner-pushed commands and ack each.

    ``attached`` is the worker's ``{scene_id: backend}`` map (the render path looks scenes up here);
    ``build_backend(scene_id, artifact_path, ipc_tensors)`` loads the structure and aliases it onto the
    owner's IPC weights — injected so this dispatch is unit-testable without a GPU. Attach is idempotent
    (a re-broadcast of an already-attached scene is a no-op), as is detach. Exits on the ``stop`` sentinel.
    """

    def acknowledge(seq: int, status: str, detail: Any) -> None:
        # An ack transport fault is itself acknowledgement uncertainty. Keep the listener alive so the owner can
        # time out, retain its attachment, and send the compensating detach; killing the listener here would make
        # an otherwise recoverable lost ack permanent.
        try:
            scene_ack_q.put((seq, status, detail))
        except Exception as exc:
            log.warning(f"scene listener could not send {status!r} ack for seq={seq}: {exc!r}")

    while True:
        message = scene_q.get()
        kind = message[0]
        if kind == _SCENE_STOP:
            return
        seq = message[1]
        if kind == _SCENE_ATTACH:
            _, _, scene_id, artifact_path, ipc_tensors = message
            if scene_id in attached:  # idempotent: a barrier re-attach must not rebuild
                acknowledge(seq, "ok", kind)
                continue
            try:
                # A build failure occurs before the worker publishes/mutates any scene state, so an explicit
                # error is proof that the attach did not install and the owner may roll its refcount back.
                backend = build_backend(scene_id, artifact_path, ipc_tensors)
            except Exception as exc:
                acknowledge(seq, "error", f"{kind}: {exc!r}")
                continue
            try:
                # Publishing into the production _WorkerSceneTable is multi-step. If it raises, it may have
                # installed part of the cache state; report UNCERTAIN so the owner retains the refcount and
                # drives an idempotent detach (or releases only after this worker is reaped).
                attached[scene_id] = backend
            except Exception as exc:
                acknowledge(seq, "uncertain", f"{kind}: {exc!r}")
                continue
            # Drop this frame's strong refs to the just-attached backend and its owner-IPC tensors before
            # looping to the next command. The listener frame lives for the whole worker process, so keeping
            # them would pin the detached scene's aliased owner-IPC mappings + worker-local allocations after
            # a later detach — until the *next* attach rebinds them (which may never come) — and the owner's
            # proven-detach release would then never actually reclaim the VRAM (silent over-admission). The
            # scene table (service BackendCache) is the sole intended retainer; mirrors the bootstrap cleanup.
            backend = None
            ipc_tensors = None
        elif kind == _SCENE_DETACH:
            _, _, scene_id = message
            try:
                attached.pop(scene_id, None)  # drop this worker's mapping (idempotent)
            except BackendInUseError as exc:
                # A render is still IN FLIGHT against this backend (the proxy released the coordinator lease on
                # a client deadline/cancel, but this worker runs the render to completion). The mapping is
                # intentionally retained and refcounts are CORRECT — a KNOWN-safe, definitely-retryable state.
                # Ack BUSY (distinct from uncertain) so the owner keeps the refcount (identical retain
                # semantics) but treats this as retry-later / pick-another-victim WITHOUT degrading the fleet.
                acknowledge(seq, "busy", f"{kind}: {exc!r}")
                continue
            except Exception as exc:
                # Removal can also fail after partial mutation. Keep ownership until a retry proves the mapping
                # gone; SceneAttachChannel.detach treats every non-ok outcome conservatively.
                acknowledge(seq, "uncertain", f"{kind}: {exc!r}")
                continue
        else:
            error = ValueError(f"unknown scene command: {kind!r}")
            acknowledge(seq, "error", f"{kind}: {error!r}")
            continue
        acknowledge(seq, "ok", kind)


def _model_of(owner: Any) -> Any:
    """The model whose params/buffers are IPC-exported: a `Backend`'s `renderable_model`, else ``owner``
    itself (so tests can register a bare ``nn.Module``)."""
    return getattr(owner, "renderable_model", owner)


class OwnerSceneRegistry:
    """Owner-side scene table with **per-scene first-load serialization**.

    Wraps a `ModelOwnerRegistry`: `ensure_loaded(scene_id, loader)` runs ``loader()`` **once** even under
    concurrent callers for the same not-yet-loaded scene (a per-scene lock), registers the loaded model's
    weights for CUDA-IPC export, and retains the owner object so those tensors stay valid. Different scenes
    load in parallel. Routing/eviction live elsewhere (the coordinator); `release` is the idle-drop hook,
    gated on the worker refcount (a scene is freed only once no worker still holds it).
    """

    def __init__(self, registry: "ModelOwnerRegistry | None" = None) -> None:
        self._registry = registry if registry is not None else ModelOwnerRegistry()
        self._guard = threading.Lock()  # guards the maps below (NOT held across a load)
        self._scene_locks: dict[str, threading.Lock] = {}
        self._owners: dict[str, Any] = {}  # scene_id -> loaded owner object (e.g. Backend), kept alive

    @property
    def registry(self) -> "ModelOwnerRegistry":
        return self._registry

    def is_loaded(self, scene_id: str) -> bool:
        with self._guard:
            return scene_id in self._owners

    def loaded_scenes(self) -> tuple[str, ...]:
        with self._guard:
            return tuple(sorted(self._owners))

    def owner_for(self, scene_id: str) -> Any:
        with self._guard:
            return self._owners.get(scene_id)

    def _scene_lock(self, scene_id: str) -> threading.Lock:
        with self._guard:
            return self._scene_locks.setdefault(scene_id, threading.Lock())

    def ensure_loaded(self, scene_id: str, loader: "Callable[[], Any]") -> Any:
        """Return the owner object for ``scene_id``, loading it **once** under concurrency.

        Concurrent callers for the same not-yet-loaded scene serialize on a per-scene lock, so the
        (expensive) ``loader()`` — download + GPU load — runs once and the rest find it loaded. The
        per-scene lock is held across the load but the shared guard is not, so other scenes load in
        parallel. A loader exception propagates and leaves the scene unloaded (the next call retries).
        """
        with self._scene_lock(scene_id):
            with self._guard:
                existing = self._owners.get(scene_id)
            if existing is not None:
                return existing
            owner = loader()
            self._registry.put_loaded(scene_id, _model_of(owner))
            with self._guard:
                self._owners[scene_id] = owner
            return owner

    def release(self, scene_id: str) -> bool:
        """Drop the owner ref + registry entry for an **idle** scene (no worker attachments). Returns True
        if freed; False if not loaded or still attached. Used by eviction after workers detach."""
        with self._scene_lock(scene_id):
            with self._guard:
                if scene_id not in self._owners:
                    return False
            freed = self._registry.release_if_idle(scene_id)  # False while any worker is still attached
            if freed:
                with self._guard:
                    self._owners.pop(scene_id, None)
            return freed

    def release_all_idle(self) -> "list[str]":
        """Release every currently-idle loaded scene (no worker still attached); return the freed ids.

        Used at post-reap teardown (`WorkerPool.shutdown()`): once every worker is dead and each
        `SceneAttachChannel` has dropped its retained refcounts, the loaded scenes are idle and their owner
        objects + IPC registry entries can be freed. A scene that is somehow still non-idle (refcount > 0) is
        left untouched (`release` returns False) rather than force-freed."""
        freed: list[str] = []
        for scene_id in self.loaded_scenes():
            if self.release(scene_id):
                freed.append(scene_id)
        return freed

    def forget_scene_lock(self, scene_id: str) -> None:
        """Drop the cached per-scene lock for a scene that is fully evicted (no owner ref, no attachments).

        The coordinator calls this only when the scene is quiescent (no attach/load in flight), so the lock
        is recreated by the next `ensure_loaded`/`release` via `setdefault`. Bounds `_scene_locks` instead of
        leaking one `threading.Lock` per distinct scene id ever loaded."""
        with self._guard:
            self._scene_locks.pop(scene_id, None)


class SceneDetachBusyError(RuntimeError):
    """A worker refused a detach because a render is still IN FLIGHT against the scene's backend.

    This is the F001 case: the proxy released the coordinator lease when the forwarded RPC raised (client
    deadline/cancel mid-render), but the worker runs the render to completion, so its `BackendCache` still
    holds the backend `in_use`. The worker reports this explicitly (a ``busy`` ack) — a KNOWN-safe,
    definitely-retryable state where the owner refcount and the worker mapping are BOTH intentionally retained
    and CORRECT (identical retain semantics to an uncertain ack: `release_if_idle` cannot free an in-use IPC
    mapping). It is distinct from genuine detach *uncertainty* (a lost/timed-out ack or a crash, where the
    fleet may be corrupt): the coordinator treats busy as retry-later / pick-another-victim WITHOUT counting it
    toward quarantine/degrade, so a single orphaned render can never escalate to a fleet-wide restart. Subclass
    of ``RuntimeError`` and phrased as a "still-live worker" refusal so it flows through the same
    retain-on-uncertainty call sites as the generic detach fault.
    """


class SceneAttachChannel:
    """Owner side of **one worker's** runtime scene attach/detach queue.

    Pairs with `_scene_attach_listener` in that worker. `attach` issues a refcounted `ModelOwnerRegistry`
    attachment (which also yields the scene's IPC tensors), records that ownership before any fallible queue
    operation, pushes an attach command, and waits for the worker's ack. A confirmed worker failure rolls the
    refcount back; uncertain delivery/rollback retains the record so post-reap cleanup cannot lose the owner
    reference. `detach` is the inverse. One channel per worker; fans out across channels (in parallel) for the
    all-worker attach barrier.
    """

    def __init__(
        self,
        push_q: Any,
        ack_q: Any,
        registry: "ModelOwnerRegistry",
        *,
        is_worker_alive: "Callable[[], bool] | None" = None,
    ) -> None:
        self._push_q = push_q
        self._ack_q = ack_q
        self._registry = registry
        # Liveness probe for this worker (e.g. `proc.is_alive`). Used on a detach ack failure to decide
        # whether the worker's IPC mapping can still be live: drop the owner refcount only when the worker is
        # provably dead, never on a mere ack timeout from a slow-but-alive worker. ``None`` ⇒ treat the worker
        # as possibly-alive (the safe default: never force-drop on an ack failure).
        self._is_worker_alive = is_worker_alive
        self._seq = itertools.count(1)
        self._attachments: dict[str, str] = {}  # scene_id -> registry attachment_id held for this worker
        self._lock = threading.Lock()  # serialize this worker's attach/detach (one in-flight at a time)
        # Post-reap queue teardown is retryable: successful endpoint operations are retired, while a failed
        # operation remains pending and the queue handle remains strongly owned by this channel for diagnosis
        # or a later retry. Capability checks happen lazily because unit-test queue.Queue instances do not
        # expose multiprocessing.Queue's cancel_join_thread()/close() methods.
        self._queue_cleanup_pending: "dict[str, dict[str, bool]] | None" = None

    def attached_scenes(self) -> set[str]:
        with self._lock:
            return set(self._attachments)

    def is_attached(self, scene_id: str) -> bool:
        with self._lock:
            return scene_id in self._attachments

    def seed_attachment(self, scene_id: str, attachment_id: str) -> None:
        """Register a pre-existing attachment (e.g. the bootstrap scene attached during pool startup, not via
        this channel) so a later `detach` can drop it through the normal path. Lets the bootstrap
        scene be evicted like any runtime-attached scene when it is not pinned."""
        with self._lock:
            self._attachments.setdefault(scene_id, attachment_id)

    def attach(self, scene_id: str, artifact_path: str, *, ack_timeout_s: float = _SCENE_ATTACH_ACK_TIMEOUT_S) -> None:
        """Attach ``scene_id`` (already loaded on the owner) onto this worker; idempotent per worker.

        Acknowledgement uncertainty is handled symmetrically with `detach`: an **explicit** worker
        error means the attach did not install, so the owner refcount is rolled back; a **timeout** means the
        worker may have installed the IPC mapping, so the refcount + attachment are **retained** unless the
        worker is provably dead (its mapping is then gone and freeing is safe). Retaining a possibly-mapped
        attachment keeps `release_if_idle` from `cudaFree`-ing tensors a live worker still maps; a later
        detach (or post-reap release) drops it. All non-``ok`` outcomes raise so the caller can quarantine.
        """
        with self._lock:
            if scene_id in self._attachments:
                return
            attachment = self._registry.attach(scene_id)  # refcount++ and hand back the IPC tensors
            # Record ownership *before* queue delivery. Queue delivery itself can fail, and a later rollback
            # can also fail; in either case post-reap cleanup must still be able to discover this registry
            # attachment. The record is removed only after a confirmed registry.detach below.
            self._attachments[scene_id] = attachment.attachment_id
            seq = next(self._seq)
            try:
                self._push_q.put((_SCENE_ATTACH, seq, scene_id, artifact_path, dict(attachment.tensors)))
            except Exception as exc:
                # Queue delivery failure is not proof that a multiprocessing feeder did not accept/deliver
                # the command. Retain the refcount and provisional record until worker reap (fail closed).
                raise RuntimeError(
                    f"scene attach {scene_id!r} could not be queued: {exc!r}; retaining the owner refcount "
                    "until the worker is reaped"
                ) from exc
            status, detail = _await_seq_ack(self._ack_q, seq, ack_timeout_s, timeout_status="timeout")
            if status == "ok":
                return
            if status == "error":
                # Explicit worker failure (build/alias raised): the attach did not install, so freeing the
                # owner refcount is safe and keeps `release_if_idle` correct.
                try:
                    self._registry.detach(attachment.attachment_id)
                except Exception as exc:
                    raise RuntimeError(
                        f"scene attach {scene_id!r} failed on worker: {detail}; owner refcount rollback "
                        f"also failed: {exc!r}; retaining the attachment for post-reap cleanup"
                    ) from exc
                self._attachments.pop(scene_id, None)
                raise RuntimeError(f"scene attach {scene_id!r} failed on worker: {detail}")
            # Timeout or an explicit `uncertain` worker outcome: the worker may have mapped the IPC tensors.
            uncertain_detail = (
                f"timed out: {detail}" if status == "timeout" else f"returned uncertain status {status!r}: {detail}"
            )
            worker_dead = self._is_worker_alive is not None and not self._is_worker_alive()
            if worker_dead:
                try:
                    self._registry.detach(attachment.attachment_id)  # dead worker can't map IPC -> safe to free
                except Exception as exc:
                    raise RuntimeError(
                        f"scene attach {scene_id!r} was uncertain on a dead worker ({uncertain_detail}); owner refcount "
                        f"rollback also failed: {exc!r}; retaining the attachment for post-reap cleanup"
                    ) from exc
                self._attachments.pop(scene_id, None)
                raise RuntimeError(f"scene attach {scene_id!r} was uncertain on a dead worker: {uncertain_detail}")
            # Live (or unknown liveness): RETAIN the refcount + record the attachment so a later detach/reap
            # can drop it, and so `release_if_idle` cannot free a possibly-mapped IPC tensor (no UAF).
            raise RuntimeError(
                f"scene attach {scene_id!r} was uncertain on a still-live worker ({uncertain_detail}); "
                f"retaining the owner refcount so release_if_idle cannot free a possibly-mapped IPC tensor"
            )

    def detach(self, scene_id: str, *, ack_timeout_s: float = _SCENE_DETACH_ACK_TIMEOUT_S) -> None:
        """Detach ``scene_id`` from this worker; idempotent.

        On a **successful** ack the owner refcount is dropped, so `release_if_idle` can free the scene's
        IPC tensors. On an **ack failure** the refcount is dropped **only when the worker is provably dead**
        (``is_worker_alive()`` False ⇒ its CUDA-IPC mapping is gone, so freeing is safe). If the worker is
        still alive — e.g. a slow or wedged detach ack — the refcount and the attachment record are
        **retained**, so `release_if_idle` cannot `cudaFree` tensors the worker may still be mapping (a
        use-after-free); the fault is raised for the owner-death/teardown path, and a later detach can retry.
        This deliberately prefers a bounded, recoverable leak over GPU memory corruption.
        """
        with self._lock:
            attachment_id = self._attachments.get(scene_id)
            if attachment_id is None:
                return
            seq = next(self._seq)
            self._push_q.put((_SCENE_DETACH, seq, scene_id))
            status, detail = _await_seq_ack(self._ack_q, seq, ack_timeout_s, timeout_status="timeout")
            if status == "ok":
                self._registry.detach(attachment_id)
                self._attachments.pop(scene_id, None)
                return
            if status == "busy":
                # The worker explicitly reports a render still IN FLIGHT against this scene's backend (in-use).
                # This is a KNOWN-safe, definitely-retryable state: refcounts are correct and the mapping is
                # intentionally kept. RETAIN the owner refcount + attachment (identical to the uncertain path —
                # `release_if_idle` must not free an in-use IPC mapping), but raise a DISTINCT typed error so
                # the coordinator treats it as retry-later (pick another victim), NOT as detach uncertainty that
                # would quarantine + degrade the fleet on a single orphaned render.
                raise SceneDetachBusyError(
                    f"scene detach {scene_id!r} refused on a still-live worker: a render is still in flight "
                    f"(backend in-use); retaining the owner refcount so release_if_idle cannot free an in-use "
                    f"IPC mapping — retry after the render completes: {detail}"
                )
            # Ack failed: only force-drop the refcount when the worker is provably dead (mapping gone).
            worker_dead = self._is_worker_alive is not None and not self._is_worker_alive()
            if worker_dead:
                self._registry.detach(attachment_id)
                self._attachments.pop(scene_id, None)
                raise RuntimeError(f"scene detach {scene_id!r} failed on a dead worker: {detail}")
            # Worker still alive (or liveness unknown): keep the refcount + attachment so the scene stays
            # non-idle and its IPC tensors are not freed under a possibly-live mapping. Surface the fault.
            raise RuntimeError(
                f"scene detach {scene_id!r} failed on a still-live worker: {detail}; "
                f"keeping the owner refcount so release_if_idle cannot free an in-use IPC mapping"
            )

    def release_after_reap(self) -> "tuple[list[str], list[tuple[str, Exception]]]":
        """Drop every retained attachment refcount **after the worker is proven dead** and close the queues.

        Retained attachments (a detach/attach whose ack was lost while the worker was still alive) keep the
        owner refcount so `release_if_idle` cannot free tensors the worker might still map. Once
        `WorkerPool.shutdown()` has reaped the worker its CUDA-IPC mappings are gone, so those refcounts can be
        force-dropped, letting the owner release the now-idle scenes.

        Returns ``(released, failures)``: registry-detach and queue-endpoint failures are **aggregated** (not
        suppressed). Only successfully-released attachment records and successfully-completed endpoint
        operations are cleared; failures retain their records/handles so the caller can diagnose or retry the
        accounting/resource leak. The worker is dead, so the queue endpoints are **closed**
        (``cancel_join_thread`` + ``close``) rather than enqueueing another stop sentinel to a consumer that
        will never read it. MUST be called only for a provably-dead worker."""
        released: list[str] = []
        failures: list[tuple[str, Exception]] = []
        with self._lock:
            for scene_id, attachment_id in list(self._attachments.items()):
                try:
                    self._registry.detach(attachment_id)
                    self._attachments.pop(scene_id, None)  # clear only what actually released
                    released.append(scene_id)
                except Exception as exc:  # noqa: BLE001 - aggregate; keep the record for diagnosis/retry
                    failures.append((scene_id, exc))
        # The consumer (worker) is dead: drop the queue endpoints without waiting on feeder joins to a dead
        # reader. Plain `queue.Queue` test doubles have neither method, hence the capability checks. Track
        # each operation separately:
        # a successful cancel must not be repeated merely because close failed, while the failed close remains
        # pending and the queue remains strongly referenced by this channel for a later retry.
        queues = {"push queue": self._push_q, "ack queue": self._ack_q}
        if self._queue_cleanup_pending is None:
            self._queue_cleanup_pending = {
                label: {
                    "cancel_join_thread": callable(getattr(q, "cancel_join_thread", None)),
                    "close": callable(getattr(q, "close", None)),
                }
                for label, q in queues.items()
            }
        for label, q in queues.items():
            pending = self._queue_cleanup_pending[label]
            for operation in ("cancel_join_thread", "close"):
                if not pending[operation]:
                    continue
                try:
                    getattr(q, operation)()
                    pending[operation] = False
                except Exception as exc:  # noqa: BLE001 - aggregate and retain the pending operation
                    failures.append((f"{label} {operation}", exc))
        return released, failures

    def stop(self) -> None:
        # Let the teardown owner aggregate a broken queue/send instead of falsely reporting a clean listener
        # stop. Callers that are already unwinding an earlier failure suppress this independently so every
        # worker still gets a reap attempt. Once post-reap endpoint cleanup has begun, the worker is already
        # proven dead and a retrying WorkerPool.shutdown must not enqueue another sentinel onto a closed queue.
        if self._queue_cleanup_pending is not None:
            return
        self._push_q.put((_SCENE_STOP,))


def _alias_and_release_worker_weights(backend: "Backend", ipc_tensors: dict) -> None:
    """Alias a freshly built worker backend onto the owner's IPC tensors, then return the replaced worker-local
    weight storage to the CUDA driver.

    ``Backend.from_artifact`` empties the CUDA allocator cache *before* aliasing, while this worker's own copy of
    the weights is still live, so those freed blocks otherwise linger in the caching allocator as per-worker
    *reserved* bytes even though nothing references them. ``alias_external_weights`` rebinds every param/buffer
    onto the owner's zero-copy IPC storage (preserving the read-only / external-storage invariant), dropping the
    worker's local tensors; a ``gc.collect()`` then breaks any cycle still holding them and
    ``torch.cuda.empty_cache()`` returns the now-unreferenced cached blocks to the driver. Ordering matters: the
    cache drop must run AFTER aliasing has replaced the local tensors, otherwise it frees nothing that the
    still-live locals occupy.
    """
    alias_external_weights(canonical_model(backend.renderable_model), ipc_tensors)

    # The replaced local tensors are now unreferenced by the module. Collect first so any container/cycle holding
    # them is dropped before empty_cache, which only frees allocator blocks with no live tensor.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pool_backend_cache_key(scene_id: str, artifact: "Artifact") -> BackendCacheKey:
    """Backend-cache identity for a scene loaded by the multi-process pool.

    Keyed by the **routing** ``scene_id`` (what the proxy, the coordinator and each worker's
    ``BackendCache`` resolve a render by) plus the artifact's exact content SHA, mirroring
    single-process ``serve``.

    The ``SceneCacheKey`` deliberately carries no ``canonical_url``: the owner and workers load from
    a local path while a coordinator-attached scene may arrive by URL, and keying those differently
    would hash the same scene differently across the owner and its own workers. ``content_sha256``
    still supplies the stale-bytes protection ``BackendCacheKey`` exists for.
    """
    return BackendCacheKey(SceneCacheKey(scene_id=scene_id), artifact.sha256_footprint)


def _build_and_release_worker_backend(
    artifact: "Artifact",
    renderer_name: str,
    ipc_tensors: dict,
    tracks_timestamp_ray_stride: int = 1,
    *,
    scene_id: str,
) -> "Backend":
    """Load a worker-side backend from ``artifact`` and alias + release it onto the owner's IPC tensors.

    The single build-and-release seam shared by both worker load sites — bootstrap ``_worker`` and runtime
    ``_scene_backend_builder`` — ensures the post-alias release runs identically in both and can be exercised in
    isolation from the rest of worker startup. Callers invoke it BEFORE the backend is made routable (server bind
    / scene-cache publication / ``ready``).
    """
    backend = Backend.from_artifact(
        artifact,
        _renderer_type(renderer_name),
        tracks_timestamp_ray_stride,
        backend_cache_key=_pool_backend_cache_key(scene_id, artifact),
    )
    _alias_and_release_worker_weights(backend, ipc_tensors)
    return backend


def _scene_backend_builder(
    renderer_name: str, tracks_timestamp_ray_stride: int = 1
) -> "Callable[[str, str, dict], Any]":
    """Build the worker-side `build_backend` used by `_scene_attach_listener`: load the structure from the
    scene's artifact, alias it onto the owner's IPC weights, and release the replaced worker-local storage."""

    def build(scene_id: str, artifact_path: str, ipc_tensors: dict) -> "Backend":
        return _build_and_release_worker_backend(
            Artifact(Path(artifact_path)),
            renderer_name,
            ipc_tensors,
            tracks_timestamp_ray_stride,
            scene_id=scene_id,
        )

    return build


class _WorkerSceneTable:
    """The mapping `_scene_attach_listener` fills in a live worker.

    Storing a scene's backend registers it in the worker's `SensorSimService` BackendCache as a spare (so
    `render_rgb` resolves it by `scene_id` via `checkout`); popping evicts that backend. Implements only
    the `in` / `[]` / `[]=` / `pop` the listener uses, so the listener stays generic (and unit-testable
    with a plain dict).

    Cache mutations go through the first-class `BackendCache.put`/`checkin`/`grow_maxsize`/`remove_by_key`
    API so capacity growth and by-key eviction take the cache lock and fire `capacity_changed` instead of
    reaching into its internals. Behavior is pinned by the worker-scene-table tests."""

    def __init__(self, service: Any) -> None:
        self._service = service
        self._scenes: dict[str, Any] = {}

    def __contains__(self, scene_id: str) -> bool:
        return scene_id in self._scenes

    def __getitem__(self, scene_id: str) -> Any:
        return self._scenes[scene_id]

    def get(self, scene_id: str, default: Any = None) -> Any:
        """Resolve ``scene_id`` to its resident backend (or ``default`` if not resident)."""
        return self._scenes.get(scene_id, default)

    def __setitem__(self, scene_id: str, backend: Any) -> None:
        # Publish the ownership record first. BackendCache.put/checkin are multi-step and may raise after
        # mutating cache state; retaining this record makes a subsequent idempotent detach able to find and
        # clean the partially-installed backend. The coordinator never routes this scene until the attach ack.
        self._scenes[scene_id] = backend
        # The coordinator — never this worker's own count-LRU — is the sole eviction authority over table
        # scenes: an LRU-dropped table scene is unrecoverable worker-side (it is not in `artifacts`) while
        # the coordinator still routes to it → permanent NOT_FOUND on a scene the pool reports as loaded.
        # With coordinator eviction ON the cache is pre-sized to max_scenes + 1 and this is a no-op; with
        # eviction OFF (max_scenes == 0: unbounded admission) grow the capacity in step so the attach
        # stream can never trip the worker LRU. Grow-only; `grow_maxsize` bumps maxsize under the cache lock.
        cache = self._service.backend_cache
        needed = len(self._scenes) + 1
        if needed > cache.maxsize:
            cache.grow_maxsize(needed - cache.maxsize)
        # Register before publishing: `get_backend` resolves the artifact registry BEFORE the backend
        # cache, so a resident-but-unregistered scene reads as NOT_FOUND. The registration is
        # resident-only (never rebuilt), preserving the fail-loud contract above. Both sides key off
        # `backend_cache_key`, so the registry lookup and the cache entry agree.
        self._service.register_resident_backend(backend)
        cache.put(backend.backend_cache_key.key_hash, backend)  # marked in-use
        cache.checkin(backend)  # -> spare, so a later checkout finds it

    def pop(self, scene_id: str, default: Any = None) -> Any:
        backend = self._scenes.get(scene_id)
        if backend is None:
            return default
        cache = self._service.backend_cache
        # Keep `_scenes` populated until the cache reference is removed. If `remove_by_key` raises, the record
        # survives and the owner's bounded detach retry can converge instead of receiving a false success while
        # the cache still holds IPC-backed tensors. `remove_by_key` **raises `BackendInUseError`** if a render
        # is still in flight against this backend (the proxy released the coordinator lease on a client
        # deadline/cancel but the worker runs the render to completion), so the exception propagates here: the
        # scene stays in `_scenes`, the listener acks the detach UNCERTAIN, and the owner retains the refcount +
        # attachment until the render checks the backend back in as a spare and a later detach removes it
        # cleanly — never freeing IPC tensors an in-flight render still aliases (use-after-free).
        cache.remove_by_key(backend.backend_cache_key.key_hash)
        # Only after the cache reference is gone (remove_by_key raises BackendInUseError while a render is
        # still in flight, leaving the record intact for the owner's retry) does the registration go too.
        self._service.unregister_resident_backend(backend.backend_cache_key)
        return self._scenes.pop(scene_id, backend)


class SceneNotKnownError(Exception):
    """A render targeted a `scene_id` the owner has no artifact for (maps to gRPC NOT_FOUND)."""


class SceneAttachError(Exception):
    """A scene failed to attach on one or more workers; the partial attach was rolled back (UNAVAILABLE)."""


class SceneLoadError(Exception):
    """A known scene's owner artifact/backend failed to load cleanly (maps to gRPC UNAVAILABLE).

    This is deliberately distinct from an unexpected proxy/coordinator invariant failure: only an exception
    raised while crossing the owner-load boundary is translated. The original loader exception is retained as
    ``__cause__`` for diagnostics, while callers receive a stable scene-domain status rather than gRPC UNKNOWN.
    """


class SceneBudgetError(Exception):
    """The scene budget is full and every resident scene is pinned or in-flight, so no idle scene can be
    evicted to make room for a new one (maps to gRPC RESOURCE_EXHAUSTED)."""


class ScenePoolDegradedError(Exception):
    """The scene residency control plane degraded: a detach/attach could not be driven to a
    proven-consistent state, so some worker may still map a scene whose owner tensors we must not free. New
    and blocked scene-backed work fails `UNAVAILABLE` while the whole fleet tears down for a supervisor
    restart, rather than routing to (or over-admitting on top of) an uncertain fleet."""


class SceneRegistry:
    """Scene **registration** bookkeeping — extracted from `SceneCoordinator` (F012) so the coordinator stays
    focused on residency, eviction, leases, and degradation.

    Owns: `scene_id -> artifact_path` (owner-loadable scenes), the on-demand `source` URI + disk-cache
    `generation` per scene, the set of **permanent** (glob/bootstrap) ids that survive disk invalidation, and
    the deferred (`_pending_invalidation`) drops for scenes still resident when their disk file was evicted.
    This is the registration/source/generation/invalidation cluster the review (F012) called out as largely
    independent of the residency/eviction machinery.

    LOCKING (decision): this registry **SHARES the coordinator's lock** (passed in), NOT a lock of its own.
    Every registration mutation therefore happens under the SAME `threading.Condition` as the coordinator's
    residency state, so the atomic 'check-residency-then-(defer-or-drop)' decisions in `invalidate_scene`
    remain byte-for-byte the transaction they were before the split — there is NO second lock and NO lock
    nesting to reason about against the coordinator's condition variable. Residency is queried through two
    injected predicates (`is_resident_locked`, `is_attached_locked`) evaluated under the held shared lock, so
    the registry never reaches into coordinator internals. Methods suffixed ``_locked`` assume the shared lock
    is already held (the coordinator calls them from its own locked sections); the registrar-facing methods
    acquire it (the coordinator's default `Condition` uses a reentrant RLock, so an incidental re-entry from a
    coordinator caller that already holds it is safe)."""

    def __init__(
        self,
        scene_artifacts: "dict[str, str]",
        lock: "threading.Condition",
        *,
        is_resident_locked: "Callable[[str], bool]",
        is_attached_locked: "Callable[[str], bool]",
    ) -> None:
        # The coordinator's condition (SHARED): registration + residency mutate under one lock, no nesting.
        self._lock = lock
        self._is_resident_locked = is_resident_locked
        self._is_attached_locked = is_attached_locked
        self._scene_artifacts = dict(scene_artifacts)  # scene_id -> artifact_path (owner-loadable scenes)
        # On-demand scenes' source URI (scene_id -> x-nre-scene-url), tracked so a re-registration of a
        # still-registered scene with a *different* source is rejected `INVALID_ARGUMENT`. Glob
        # scenes have no recorded source (they are never re-registered from a URL).
        self._scene_sources: dict[str, str] = {}
        # Remote cache generations make invalidation compare-and-delete: a delayed callback for an older
        # disk file can never remove a newer registration of the same public scene id.
        self._scene_generations: dict[str, int] = {}
        # Scenes registered at startup (glob/bootstrap/extras): their artifact path is a permanent
        # registration — kept across GPU eviction (re-attach with no client header) and never invalidated by
        # the disk cache. An on-demand scene's registration also survives GPU eviction: it stays
        # ID-only reloadable while its backing artifact is usable (a local path, or a still-present
        # disk-cache file), and is dropped only when the disk cache evicts it (see `invalidate_scene`).
        self._permanent_scene_ids = set(scene_artifacts)
        # On-demand scenes whose disk-cache file was evicted while they were still resident/in-flight: their
        # registration is dropped once they leave residency, so in-flight renders finish first.
        self._pending_invalidation: dict[str, int | None] = {}

    # --- residency-independent reads (caller already holds the shared lock) ---
    def artifact_path_locked(self, scene_id: str) -> "str | None":
        """The registered artifact path for ``scene_id`` (or None). Caller holds the shared lock."""
        return self._scene_artifacts.get(scene_id)

    def is_registered_locked(self, scene_id: str) -> bool:
        """Whether ``scene_id`` has a registered (re-loadable) artifact path. Caller holds the shared lock."""
        return scene_id in self._scene_artifacts

    def registered_scene_ids_locked(self) -> set[str]:
        """A snapshot of every registered scene id. Caller holds the shared lock."""
        return set(self._scene_artifacts)

    # --- registrar-facing transactions (acquire the shared lock) ---
    def registration(self, scene_id: str) -> "tuple[str, str | None, int | None] | None":
        """Return one atomic artifact/source/generation snapshot for registrar reconciliation."""
        with self._lock:
            artifact_path = self._scene_artifacts.get(scene_id)
            if artifact_path is None:
                return None
            return artifact_path, self._scene_sources.get(scene_id), self._scene_generations.get(scene_id)

    def register_scene(
        self,
        scene_id: str,
        artifact_path: str,
        *,
        source: "str | None" = None,
        generation: "int | None" = None,
    ) -> None:
        """Register an on-demand scene's artifact path so `ensure_attached` can first-load + attach it.
        Idempotent; a scene already attached (e.g. the primary) is a no-op, and re-registering a
        known scene with a **different** artifact path (or a different ``source`` URI) raises `ValueError`
        (mirrors `serve`'s conflict guard). ``source`` is the client ``x-nre-scene-url`` for an
        on-demand scene (None for glob/local registrations)."""
        with self._lock:
            existing = self._scene_artifacts.get(scene_id)
            if existing is not None:
                if existing != artifact_path:
                    raise ValueError(
                        f"scene {scene_id!r} already registered with a different artifact "
                        f"(existing: {existing}, requested: {artifact_path})"
                    )
                existing_source = self._scene_sources.get(scene_id)
                if source is not None and existing_source is not None and existing_source != source:
                    raise ValueError(
                        f"scene {scene_id!r} already registered from a different source "
                        f"(existing: {existing_source}, requested: {source})"
                    )
                existing_generation = self._scene_generations.get(scene_id)
                if generation is not None and existing_generation is not None and existing_generation != generation:
                    # A same-source replacement may renew the reload mapping after this exact prior disk
                    # generation was invalidated while the scene remained resident/attaching. The loaded
                    # backend can finish on its in-memory data; its next reload uses the fresh file.
                    if self._pending_invalidation.get(scene_id) != existing_generation:
                        raise ValueError(
                            f"scene {scene_id!r} already registered from a different cache generation "
                            f"(existing: {existing_generation}, requested: {generation})"
                        )
                    self._scene_generations[scene_id] = generation
                    self._pending_invalidation.pop(scene_id, None)
                return
            if self._is_attached_locked(scene_id):
                return
            self._scene_artifacts[scene_id] = artifact_path
            if source is not None:
                self._scene_sources[scene_id] = source
            if generation is not None:
                self._scene_generations[scene_id] = generation
            self._pending_invalidation.pop(scene_id, None)

    def check_source_conflict(self, scene_id: str, source: str) -> None:
        """Raise `ValueError` if ``scene_id`` is already registered from a *different* on-demand source URI
        (the interceptor maps it to `INVALID_ARGUMENT`). A scene with no recorded source (a glob
        registration) or the same source is not a conflict. Used by the on-demand registrar to reject a
        conflicting re-registration without re-resolving/re-downloading a scene it already knows."""
        with self._lock:
            existing_source = self._scene_sources.get(scene_id)
            if existing_source is not None and existing_source != source:
                raise ValueError(
                    f"scene {scene_id!r} already registered from a different source "
                    f"(existing: {existing_source}, requested: {source})"
                )

    def invalidate_scene(self, scene_id: str, generation: "int | None" = None) -> None:
        """Drop an on-demand scene's registration when its disk-cache file is evicted.

        After invalidation an ID-only request for the scene returns `NOT_FOUND` (never `UNKNOWN` from a
        vanished file) and the scene stops being advertised; repeating the `x-nre-scene-url` header
        re-registers and re-downloads it exactly once. A **permanent** (glob/bootstrap) registration is
        never invalidated. A currently-resident/in-flight scene defers the invalidation until it leaves
        residency (so existing/in-flight renders finish), applied by
        `apply_pending_invalidation_if_quiescent_locked`."""
        with self._lock:
            if scene_id in self._permanent_scene_ids:
                return  # glob/bootstrap scenes keep their mapping regardless of the disk cache
            if scene_id not in self._scene_artifacts:
                return  # already unknown / never registered here
            current_generation = self._scene_generations.get(scene_id)
            if generation is not None and current_generation != generation:
                return  # stale callback for an older (or not-yet-published) cache generation
            if self._is_resident_locked(scene_id):
                # In-flight renders may finish; drop the mapping only once the scene leaves residency.
                self._pending_invalidation[scene_id] = generation
                return
            self.drop_registration_locked(scene_id, generation)

    # --- shared-lock helpers the coordinator drives from its own locked sections ---
    def drop_registration_locked(self, scene_id: str, generation: "int | None") -> bool:
        """Compare-and-delete one on-demand registration. Caller holds the shared lock."""
        current_generation = self._scene_generations.get(scene_id)
        if generation is not None and current_generation != generation:
            return False
        self._scene_artifacts.pop(scene_id, None)
        self._scene_sources.pop(scene_id, None)
        self._scene_generations.pop(scene_id, None)
        self._pending_invalidation.pop(scene_id, None)
        return True

    def apply_pending_invalidation_if_quiescent_locked(self, scene_id: str) -> None:
        """Consume a deferred disk invalidation on every path that can make a scene quiescent. Caller holds
        the shared lock."""
        if scene_id not in self._pending_invalidation or scene_id in self._permanent_scene_ids:
            return
        if not self._is_resident_locked(scene_id):
            self.drop_registration_locked(scene_id, self._pending_invalidation[scene_id])


class _DetachResult(enum.Enum):
    """Terminal outcome of `SceneCoordinator._detach_release_or_quarantine` for one scene.

    RELEASED and QUARANTINED are the historical bool outcomes (proven-freed vs degrade-fail-closed). BUSY is
    the F001-refinement outcome: a worker still renders the scene (in-use) and refused the detach — a
    KNOWN-safe, definitely-retryable state (refcounts correct, mapping retained), so eviction picks another
    victim or fails the requester retryable WITHOUT quarantining or degrading the fleet."""

    RELEASED = "released"  # no worker maps it, owner tensors freed -> safe to drop its bookkeeping
    BUSY = "busy"  # a worker is still rendering it (in-use); refcounts correct, retry later (no degrade)
    QUARANTINED = "quarantined"  # detach unconfirmed / invariant fault -> pool degraded fail-closed, tensors retained


class SceneCoordinator:
    """Owner-side multi-scene gate: maps `scene_id` -> artifact, loads each scene **once**, and
    attaches it on **all** workers (the barrier) before the proxy routes a render to it.

    `ensure_attached(scene_id)` is idempotent and serialized per scene: it first-loads the scene on the
    owner (`OwnerSceneRegistry`), then fans the attach out across every worker's `SceneAttachChannel` in
    parallel. If any worker fails, the successful attaches are rolled back (detached) and the owner load is
    released, so a half-attached scene is never routable — preserving the round-robin invariant that all
    workers are interchangeable for an attached scene. The initial (`artifact_glob`) scene(s) are seeded as
    already attached. No eviction here; `release`/`detach` of runtime scenes is exercised there.
    """

    def __init__(
        self,
        owner_registry: "OwnerSceneRegistry",
        channels: "list[SceneAttachChannel]",
        scene_artifacts: "dict[str, str]",
        loader: "Callable[[str, str], Any]",
        *,
        attached_scenes: "tuple[str, ...]" = (),
        max_scenes: int = 0,
        pinned: "tuple[str, ...]" = (),
    ) -> None:
        self._owner_registry = owner_registry
        self._channels = list(channels)
        # Registration bookkeeping (artifact/source/generation maps + permanent ids + deferred disk
        # invalidations) lives in a composed `SceneRegistry` (F012). It SHARES this coordinator's lock (built
        # just below), so registration and residency still mutate under one lock — constructed at the end of
        # __init__ once the residency predicates it queries are wired.
        self._loader = loader  # (scene_id, artifact_path) -> owner object (e.g. Backend)
        # guards residency state + the shared registry (NOT held across a load/attach/detach)
        self._lock = threading.Condition()
        self._scene_locks: dict[str, threading.Lock] = {}
        # Count of threads currently inside a scene's attach critical section (holding its `_scene_lock`,
        # possibly waiting on the eviction barrier). Eviction only prunes a scene's per-scene bookkeeping when
        # this is 0, so it never drops a lock a concurrent re-attach still holds (which would let a stale and
        # a fresh lock both admit the same scene). Pruning a busy scene is simply deferred to a later eviction.
        self._attach_active: dict[str, int] = {}
        self._attached: set[str] = set(attached_scenes)  # scenes attached pool-wide (seed: the primary)
        # Coordinated eviction + memory budget:
        self._max_scenes = max_scenes  # cap on resident (attached) scenes; 0 = unlimited (no eviction)
        # Pinning is PATH-DRIVEN: a scene is evictable iff it has a registered artifact to re-load
        # from. `_pinned` holds the few **explicit** pins; a scene with no registered path is also
        # never evicted (it could not be reloaded). A glob fleet registers every scene's path ⇒ zero pinned.
        self._pinned: set[str] = set(pinned)
        self._inflight: dict[str, int] = {}  # scene_id -> active render/discovery leases (eviction skips > 0)
        self._lru: dict[str, int] = {}  # scene_id -> last-access stamp (lower = older); LRU victim selection
        self._evicting: set[str] = set()  # scenes mid-eviction (detach+release in progress); not re-routable
        # Scenes whose slot is claimed but not yet in `_attached` (mid first-load/attach). Counted toward the
        # budget so concurrent first-time attaches of *different* scenes can't both pass the cap (the
        # over-admission race): the budget check + the claim happen atomically under `_lock`.
        self._reserved: set[str] = set()
        # Quarantined residency: a victim/incoming scene whose detach could not be driven to a
        # proven-consistent terminal state (a live worker may still map its IPC tensors). Its owner tensors
        # stay referenced (never freed under a possibly-live mapping) and it stays **charged against the
        # budget** so the pool cannot over-admit on top of still-resident VRAM. Released only post-reap.
        self._retained: set[str] = set()
        # Fail-closed state: latched to the first degradation reason. Once set, `is_degraded()` is
        # True, new/blocked scene work fails `ScenePoolDegradedError`, and `_on_degraded` (wired to fleet
        # teardown in `serve_proxy`) fires once so the pool exits nonzero for a supervisor restart.
        self._degraded: "str | None" = None
        self._on_degraded: "Callable[[str], None] | None" = None
        self._lru_counter = itertools.count(1)
        # The registration bookkeeping collaborator (F012). Shares this coordinator's lock and queries
        # residency through the coordinator's `*_locked` predicates, so behavior is byte-for-byte preserved.
        self._registry = SceneRegistry(
            scene_artifacts,
            self._lock,
            is_resident_locked=self._is_resident_locked,
            is_attached_locked=self._is_attached_locked,
        )

    # The registration maps now live in `self._registry`; these private properties re-expose them read-through
    # so the existing white-box registration/eviction/invalidation tests keep asserting on `coord._scene_*`
    # unchanged (F012 is strictly behavior-preserving). They return the registry's live dicts/sets.
    @property
    def _scene_artifacts(self) -> "dict[str, str]":
        return self._registry._scene_artifacts  # noqa: SLF001 - re-expose the registry map for white-box tests

    @property
    def _scene_sources(self) -> "dict[str, str]":
        return self._registry._scene_sources  # noqa: SLF001

    @property
    def _scene_generations(self) -> "dict[str, int]":
        return self._registry._scene_generations  # noqa: SLF001

    @property
    def _permanent_scene_ids(self) -> "set[str]":
        return self._registry._permanent_scene_ids  # noqa: SLF001

    @property
    def _pending_invalidation(self) -> "dict[str, int | None]":
        return self._registry._pending_invalidation  # noqa: SLF001

    def _is_resident_locked(self, scene_id: str) -> bool:
        """Whether ``scene_id`` still occupies residency/VRAM: attached, mid-attach (reserved/attach-active),
        evicting, quarantined (retained), or leased/in-flight. Evaluated under `self._lock`; the shared
        `SceneRegistry` uses it to defer a disk invalidation while a scene is still resident. Mirrors the
        residency test the registration bookkeeping ran inline before the F012 split."""
        return (
            scene_id in self._attached
            or scene_id in self._reserved
            or scene_id in self._evicting
            or scene_id in self._retained
            or self._inflight.get(scene_id, 0) > 0
            or self._attach_active.get(scene_id, 0) > 0
        )

    def _is_attached_locked(self, scene_id: str) -> bool:
        """Whether ``scene_id`` is attached pool-wide (evaluated under `self._lock`). The shared registry uses
        it so `register_scene` no-ops for an already-attached seed (e.g. the primary)."""
        return scene_id in self._attached

    def known_scenes(self) -> set[str]:
        with self._lock:
            return self._registry.registered_scene_ids_locked() | set(self._attached)

    def attached_scenes(self) -> set[str]:
        with self._lock:
            return set(self._attached)

    @property
    def max_scenes(self) -> int:
        """The resident (attached) scene cap this coordinator enforces; 0 = unlimited (no eviction)."""
        return self._max_scenes  # immutable after construction; no lock needed

    def is_known(self, scene_id: str) -> bool:
        """Whether ``scene_id`` is already registered (has an artifact) or attached pool-wide — i.e. it
        needs no (re)download. Used by the on-demand download path to skip re-resolving a scene."""
        with self._lock:
            return self._registry.is_registered_locked(scene_id) or scene_id in self._attached

    def registration(self, scene_id: str) -> "tuple[str, str | None, int | None] | None":
        """Return one atomic artifact/source/generation snapshot for registrar reconciliation (F012: delegated
        to the composed `SceneRegistry`)."""
        return self._registry.registration(scene_id)

    def register_scene(
        self,
        scene_id: str,
        artifact_path: str,
        *,
        source: "str | None" = None,
        generation: "int | None" = None,
    ) -> None:
        """Register an on-demand scene's artifact path so `ensure_attached` can first-load + attach it (F012:
        delegated to the composed `SceneRegistry`). Idempotent; a scene already attached (e.g. the primary) is
        a no-op, and re-registering a known scene with a **different** artifact path (or a different ``source``
        URI) raises `ValueError`."""
        self._registry.register_scene(scene_id, artifact_path, source=source, generation=generation)

    def check_source_conflict(self, scene_id: str, source: str) -> None:
        """Raise `ValueError` if ``scene_id`` is already registered from a *different* on-demand source URI
        (F012: delegated to the composed `SceneRegistry`; the interceptor maps it to `INVALID_ARGUMENT`)."""
        self._registry.check_source_conflict(scene_id, source)

    def invalidate_scene(self, scene_id: str, generation: "int | None" = None) -> None:
        """Drop an on-demand scene's registration when its disk-cache file is evicted (F012: delegated to the
        composed `SceneRegistry`, which shares this coordinator's lock and defers the drop while the scene is
        still resident — applied by `_prune_evicted_bookkeeping_locked` / lease-release once it goes quiet)."""
        self._registry.invalidate_scene(scene_id, generation)

    def handle_cache_eviction(self, eviction: SceneCacheEviction) -> None:
        """Correctness-critical cache callback: invalidate this exact generation or degrade fail-closed.

        Stays on the coordinator (fail-closed **degradation** is a residency-control-plane responsibility, not
        registration bookkeeping) and calls `self.invalidate_scene` (the registry facade) so a test that
        monkeypatches `invalidate_scene` still intercepts the disk-eviction path."""
        try:
            self.invalidate_scene(eviction.scene_id, eviction.generation)
        except Exception as exc:
            with self._lock:
                self._mark_degraded_locked(
                    f"disk-cache invalidation failed for scene {eviction.scene_id!r} "
                    f"generation {eviction.generation}: {exc!r}"
                )
            raise

    def set_degraded_callback(self, on_degraded: "Callable[[str], None] | None") -> None:
        """Register the one-shot callback fired when the scene pool degrades (wired to fleet teardown in
        `serve_proxy`)."""
        with self._lock:
            self._on_degraded = on_degraded

    def is_degraded(self) -> bool:
        """True once a detach/attach could not be driven to a proven-consistent terminal state; the
        scene render/attach paths fail closed on this rather than routing to (or over-admitting on) an
        uncertain fleet."""
        with self._lock:
            return self._degraded is not None

    def _mark_degraded_locked(self, reason: str) -> None:
        """Latch the first degradation reason, wake **every** condition waiter (so blocked eviction/attach
        waiters re-check the predicate and fail bounded), and fire the one-shot teardown callback. The caller
        must hold ``self._lock``."""
        if self._degraded is None:
            self._degraded = reason
            log.error(f"scene residency pool degraded (fail-closed): {reason}")
            if self._on_degraded is not None:
                with contextlib.suppress(Exception):
                    self._on_degraded(reason)
        self._lock.notify_all()

    def _raise_if_degraded_locked(self) -> None:
        if self._degraded is not None:
            raise ScenePoolDegradedError(f"scene residency pool is degraded (fail-closed): {self._degraded}")

    def _scene_lock(self, scene_id: str) -> threading.Lock:
        with self._lock:
            return self._scene_locks.setdefault(scene_id, threading.Lock())

    def ensure_attached(self, scene_id: str) -> None:
        """Make ``scene_id`` renderable on every worker; idempotent. Raises `SceneNotKnownError` if the
        owner has no artifact for it, `SceneLoadError` if its artifact/backend cannot be owner-loaded,
        `SceneAttachError` (after cleanup) if a worker fails to attach, `SceneBudgetError` if the budget is
        full and no idle scene can be evicted to make room, or
        `ScenePoolDegradedError` once the pool has degraded — including for an already-attached scene,
        since a degraded pool is tearing down.

        Prefer `acquire`/`release_scene` from the render path: they additionally hold a lease so the scene
        cannot be evicted out from under an in-flight render. `ensure_attached` alone is the attach barrier."""
        with self._lock:
            # Fail closed even for an already-attached scene: once degraded the whole fleet is restarting.
            self._raise_if_degraded_locked()
            if scene_id in self._attached:
                return
            artifact_path = self._registry.artifact_path_locked(scene_id)
        if artifact_path is None:
            raise SceneNotKnownError(f"scene {scene_id!r} is not known to this server")
        # Register an active attach for this scene **before** taking the per-scene lock, so an eviction's
        # gated prune can never drop the lock in the window between our acquiring it and our marking the
        # attach active. While `_attach_active[scene_id] > 0` the lock is never pruned; the prune fires only
        # when no attach is active, at which instant no thread holds it and any future acquirer increments
        # (under `_lock`) before `setdefault`, so all acquirers still share one lock. Decremented in the
        # finally (which also covers the early "already attached" return).
        with self._lock:
            self._attach_active[scene_id] = self._attach_active.get(scene_id, 0) + 1
        try:
            with self._scene_lock(scene_id):  # serialize the barrier for this scene (one first-load + fan-out)
                with self._lock:
                    # If another thread is evicting this scene, let the detach+release finish before we re-load
                    # it (no attach racing a detach for the same scene). Break out on degradation too, so a
                    # waiter woken by a degrading eviction fails bounded instead of looping.
                    while scene_id in self._evicting and self._degraded is None:
                        self._lock.wait()
                    self._raise_if_degraded_locked()
                    if scene_id in self._attached:
                        return
                # Atomically evict-to-make-room + reserve a budget slot, so two concurrent
                # first-time attaches of different scenes can't both pass the cap.
                self._reserve_slot(scene_id)
                try:
                    self._owner_registry.ensure_loaded(scene_id, lambda: self._load_owner(scene_id, artifact_path))
                    self._attach_all(scene_id, artifact_path)
                    with self._lock:
                        self._attached.add(scene_id)
                        self._reserved.discard(scene_id)  # the reservation becomes a real attachment
                        self._lru[scene_id] = next(self._lru_counter)
                except BaseException as exc:
                    # Single centralized cleanup on load/attach failure — FAIL CLOSED. Default to
                    # UNCLEAN and never suppress a cleanup fault: detach any partial worker attachments fleet-wide
                    # with bounded retry, then release the now-idle owner load — or quarantine + degrade. A loader
                    # failure never attached anything (is_loaded False), so there is nothing to clean. If the
                    # cleanup helper itself raises (an unexpected queue/executor/registry fault), latch degradation
                    # and keep the scene charged/retained rather than letting the fault bypass quarantine.
                    cleaned = False
                    loaded = True
                    with contextlib.suppress(Exception):
                        loaded = self._owner_registry.is_loaded(scene_id)
                    if loaded:
                        try:
                            # A failed attach never became routable, so no render can be in flight against it:
                            # a BUSY outcome is unreachable here. Treat only a proven RELEASED as cleaned;
                            # anything else (QUARANTINED, or the unreachable BUSY) retains the scene charged so
                            # we never free tensors a worker might still map.
                            cleaned = self._detach_release_or_quarantine(scene_id) == _DetachResult.RELEASED
                        except Exception as cleanup_exc:  # noqa: BLE001 - a cleanup fault must fail closed
                            with self._lock:
                                self._mark_degraded_locked(
                                    f"scene {scene_id!r} rollback cleanup raised: {cleanup_exc!r}; "
                                    "retaining its residency budget charge until fleet reap"
                                )
                            cleaned = False
                    else:
                        cleaned = True  # nothing loaded/attached -> nothing to clean up
                    with self._lock:
                        self._reserved.discard(scene_id)  # release the claimed slot on failure
                        if not cleaned:
                            # Quarantined: keep the scene CHARGED (reserved -> retained) so we don't over-admit
                            # on still-resident VRAM; its owner tensors are freed only post-reap.
                            self._retained.add(scene_id)
                        degraded = self._degraded is not None
                        self._lock.notify_all()
                    # Truthful terminal status: once the pool has degraded, surface `ScenePoolDegradedError`
                    # (→ `_scene_lease` → gRPC UNAVAILABLE) rather than an opaque original error (e.g. a
                    # SceneAttachError, which would collapse to UNKNOWN) so the client sees "fleet restarting".
                    if degraded:
                        raise ScenePoolDegradedError(
                            f"scene {scene_id!r} attach failed and the scene pool degraded during rollback; "
                            f"the fleet is restarting"
                        ) from exc
                    raise
        finally:
            with self._lock:
                self._dec_attach_active_locked(scene_id)

    def acquire(self, scene_id: str) -> None:
        """Attach ``scene_id`` (driving the barrier + any eviction) **and** take a render lease on it, so it
        cannot be evicted while a render/forward is in flight. Pair with `release_scene` in a
        `finally`. The lease is taken **before** the attach barrier runs, so eviction (which only targets
        lease-free scenes) can never select a scene a caller is committed to. Raises the same errors as
        `ensure_attached`."""
        with self._lock:
            self._inflight[scene_id] = self._inflight.get(scene_id, 0) + 1
        try:
            self.ensure_attached(scene_id)
        except BaseException:
            with self._lock:
                self._dec_inflight_locked(scene_id)
            raise
        with self._lock:
            self._lru[scene_id] = next(self._lru_counter)  # touch LRU on access

    def release_scene(self, scene_id: str) -> None:
        """Drop a render lease taken by `acquire` (idempotent-safe; never goes negative)."""
        with self._lock:
            self._dec_inflight_locked(scene_id)
            self._lock.notify_all()

    def _dec_inflight_locked(self, scene_id: str) -> None:
        count = self._inflight.get(scene_id, 0) - 1
        if count > 0:
            self._inflight[scene_id] = count
        else:
            self._inflight.pop(scene_id, None)
        self._registry.apply_pending_invalidation_if_quiescent_locked(scene_id)

    def _dec_attach_active_locked(self, scene_id: str) -> None:
        count = self._attach_active.get(scene_id, 0) - 1
        if count > 0:
            self._attach_active[scene_id] = count
        else:
            self._attach_active.pop(scene_id, None)
        self._registry.apply_pending_invalidation_if_quiescent_locked(scene_id)

    def _prune_evicted_bookkeeping_locked(self, scene_id: str) -> None:
        """Drop a fully-evicted scene's per-scene bookkeeping so the maps stay bounded under on-demand churn.

        Skipped while an attach for the scene is active (a waiting re-attach holds the per-scene lock we'd
        drop — pruning it would let a stale and a fresh lock both admit the scene); such a scene is pruned by
        a later eviction once quiet. The per-scene locks (here + the owner registry's) are recreated on the
        next access via `setdefault`.

        The scene's **artifact registration survives GPU eviction**: both permanent (glob) and
        on-demand scenes stay ID-only reloadable while their backing artifact is usable. An on-demand
        scene's mapping is dropped only when the disk cache evicts its file — that arrives via
        `invalidate_scene`, which defers to here (`_pending_invalidation`) if it fired while the scene was
        still resident, so an in-flight render finishes before the mapping disappears. Must hold `self._lock`.
        """
        if self._attach_active.get(scene_id, 0) > 0:
            return
        self._scene_locks.pop(scene_id, None)
        # Apply a deferred disk-cache invalidation now that the scene has left residency (never for a
        # permanent glob/bootstrap scene, which keeps its mapping regardless of the disk cache).
        self._registry.apply_pending_invalidation_if_quiescent_locked(scene_id)
        self._owner_registry.forget_scene_lock(scene_id)

    def _reserve_slot(self, incoming: str) -> None:
        """Evict idle, unpinned, least-recently-used scenes until there is room, then **claim a budget slot**
        for ``incoming`` — all gated by `_lock` so the capacity check and the claim are atomic (guards
        against concurrent over-admission). Budget occupancy counts both attached **and** reserved
        (mid-attach) scenes, so two concurrent first-time attaches of different scenes can't both fit a single
        free slot. Raises `SceneBudgetError` if full and every resident scene is pinned or in-flight. Each
        victim is **detached on all workers first, then released on the owner** — so the owner never frees IPC
        tensors a worker is still mapping, and a render never hits a released mapping."""
        if self._max_scenes <= 0:
            with self._lock:
                self._raise_if_degraded_locked()  # a degraded pool admits nothing, even with eviction off
                self._reserved.add(incoming)
            return
        # Victims found BUSY this round (a worker still renders them after the proxy released the lease on a
        # client deadline/cancel — the F001 orphaned render). They are healthy and fully attached, so we put
        # them back and skip them here; the requester fails retryable if no other victim is free, and a later
        # attach (after the render completes) evicts them cleanly. Never counted toward degradation.
        busy_victims: set[str] = set()
        while True:
            with self._lock:
                self._raise_if_degraded_locked()  # re-checked after every wait (loop re-enters here)
                # Count `_evicting` **and** `_retained` toward occupancy: a victim moved out of
                # `_attached` still has its VRAM mapped until its detach+release completes (evicting) or,
                # if its detach could not be confirmed, indefinitely until post-reap (retained). A concurrent
                # reservation must NOT claim either slot early, or `attached + reserved + (still-resident)`
                # exceeds `max_scenes` and can nudge a tight box toward OOM / over-admission.
                occupancy = len(self._attached) + len(self._reserved) + len(self._evicting) + len(self._retained)
                if occupancy < self._max_scenes:
                    self._reserved.add(incoming)  # claim the slot atomically with the capacity check
                    return
                victim = self._pick_victim_locked(exclude=busy_victims)
                if victim is None:
                    # No idle victim right now. If an eviction is already in flight, wait for it to free a
                    # slot (its `_evicting` occupancy clears on completion) rather than spuriously shedding;
                    # only when nothing is evicting AND nothing is evictable is the budget genuinely full.
                    if self._evicting:
                        self._lock.wait()
                        continue
                    busy_note = (
                        f" ({len(busy_victims)} evictable scene(s) are currently rendering an orphaned request "
                        "and were deferred — retry shortly)"
                        if busy_victims
                        else ""
                    )
                    raise SceneBudgetError(
                        f"scene budget reached (max_scenes={self._max_scenes}); all "
                        f"{len(self._attached) + len(self._reserved) + len(self._retained)} resident/admitting "
                        f"scene(s) are pinned, in-flight, being attached, quarantined, or currently rendering "
                        f"(cannot evict to admit {incoming!r}){busy_note}"
                    )
                # Claim the victim: out of _attached (frees a slot, blocks new routing) and into _evicting
                # (a concurrent acquire of the victim will wait, then re-attach).
                self._evicting.add(victim)
                self._attached.discard(victim)
            result = _DetachResult.QUARANTINED
            try:
                # One bounded, structured detach+release-or-quarantine: retries a lost/uncertain
                # detach ack, releases the owner load once no worker maps the scene, reports BUSY if a worker
                # still renders it (in-use), or marks the pool degraded + retains the owner tensors if a live
                # worker may still map them for an uncertain reason.
                result = self._detach_release_or_quarantine(victim)
            except Exception as cleanup_exc:  # noqa: BLE001 - eviction cleanup must fail closed
                # An unexpected queue/executor/registry fault must not bypass quarantine or recreate
                # over-admission: latch degradation and keep the victim charged/retained (below). The next
                # loop iteration hits `_raise_if_degraded_locked`, so the reserving caller fails closed.
                with self._lock:
                    self._mark_degraded_locked(
                        f"scene {victim!r} eviction cleanup raised: {cleanup_exc!r}; "
                        "retaining its residency budget charge until fleet reap"
                    )
                result = _DetachResult.QUARANTINED
            finally:
                with self._lock:
                    self._evicting.discard(victim)
                    if result == _DetachResult.RELEASED:
                        self._lru.pop(victim, None)
                        # Bound the per-scene bookkeeping: drop the evicted victim's locks (+ on-demand path)
                        # unless a re-attach is already racing it (then a later eviction prunes it once quiet).
                        self._prune_evicted_bookkeeping_locked(victim)
                    elif result == _DetachResult.BUSY:
                        # A render is still in flight against the victim (in-use); it is healthy and fully
                        # attached (refcounts correct), so restore it to `_attached` and defer it this round
                        # (skip re-picking it). Keep its LRU stamp so it stays the oldest candidate for the
                        # next eviction once the render completes. NEVER degrade for a busy refusal.
                        self._attached.add(victim)
                        busy_victims.add(victim)
                    else:  # QUARANTINED
                        self._lru.pop(victim, None)
                        # Quarantine: a live worker may still map the victim, so keep it CHARGED against
                        # the budget (do not free the slot → no over-admission) and keep its owner tensors
                        # referenced. The pool is now degraded; its bookkeeping is released only post-reap.
                        self._retained.add(victim)
                    self._lock.notify_all()

    def _pick_victim_locked(self, *, exclude: "set[str] | None" = None) -> "str | None":
        # Evictable iff: re-loadable (has a registered artifact path), not explicitly pinned,
        # and not in-flight. A path-less or explicitly-pinned scene is never a victim. ``exclude`` skips
        # scenes already found BUSY (a worker still rendering them) this reservation round, so a busy victim
        # is retried later instead of re-picked in a spin.
        excluded = exclude or set()
        candidates = [
            s
            for s in self._attached
            if self._registry.is_registered_locked(s)
            and s not in self._pinned
            and self._inflight.get(s, 0) == 0
            and s not in excluded
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: self._lru.get(s, 0))  # least-recently-used

    def pinned_scenes(self) -> set[str]:
        """The currently-resident scenes that can never be evicted: the explicit pins plus any attached
        scene with no registered artifact path to re-load from. Empty for a fully path-registered glob
        fleet ⇒ zero pinned."""
        with self._lock:
            return {s for s in self._attached if s in self._pinned or not self._registry.is_registered_locked(s)}

    def _detach_all(self, scene_id: str) -> "list[tuple[int, Exception | None, bool, bool]]":
        """Detach ``scene_id`` on every worker in parallel; return per-channel ``(index, error, retained, busy)``.

        ``channel.detach`` drops the owner refcount on a successful ack (or a provably-dead worker) and
        **raises** on any failure. ``retained`` is True when that failure left the channel still holding the
        attachment (an ack timeout from a still-live worker, or an explicit in-use ``busy`` refusal) — i.e. a
        worker may still map the scene, so its owner tensors must not be freed. ``busy`` is True only for an
        explicit `SceneDetachBusyError` (the worker reports a render still in flight): a KNOWN-safe,
        definitely-retryable state distinct from genuine detach uncertainty. The caller
        (`_detach_release_or_quarantine`) uses these structured outcomes to decide
        release-vs-busy-vs-retry-vs-quarantine instead of suppressing the failures. Idempotent per worker (a
        channel not holding the scene is a no-op)."""
        if not self._channels:
            return []
        outcomes: list[tuple[int, "Exception | None", bool, bool]] = []
        with futures.ThreadPoolExecutor(max_workers=len(self._channels)) as pool:
            future_to_index = {pool.submit(channel.detach, scene_id): i for i, channel in enumerate(self._channels)}
            for future in futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()
                    outcomes.append((index, None, False, False))
                except Exception as exc:  # noqa: BLE001 - classify per worker for the quarantine decision
                    retained = self._channels[index].is_attached(scene_id)
                    outcomes.append((index, exc, retained, isinstance(exc, SceneDetachBusyError)))
        return outcomes

    def _detach_release_or_quarantine(
        self, scene_id: str, *, attempts: int = _SCENE_DETACH_MAX_ATTEMPTS
    ) -> _DetachResult:
        """Detach ``scene_id`` fleet-wide with bounded retry, then release the owner load — or report BUSY (a
        worker still renders it) or quarantine + degrade if a consistent terminal state cannot be **proven**.

        Returns `_DetachResult.RELEASED` iff the scene was proven fully released (no worker maps it, owner
        tensors freed → safe to drop its bookkeeping). Returns `_DetachResult.BUSY` iff every still-retained
        worker refused because a render is in flight (in-use): a KNOWN-safe, definitely-retryable state
        (refcounts correct, mapping retained) — the caller retries later / picks another victim WITHOUT
        degrading. Returns `_DetachResult.QUARANTINED` iff it had to be quarantined: a live worker may still
        map it for an *uncertain* reason (lost/timed-out ack, crash, or an ownership invariant failure), so its
        owner tensors stay referenced (no use-after-free) and the pool is marked degraded (the wired callback
        tears the fleet down for a nonzero supervisor restart). Shared by eviction, partial-attach rollback,
        and the load/attach failure cleanup so there is exactly one detach-safety path."""
        retained: list[tuple[int, "Exception | None", bool]] = []
        for attempt in range(1, attempts + 1):
            outcomes = self._detach_all(scene_id)
            errors = [(index, exc) for index, exc, _retained, _busy in outcomes if exc is not None]
            # Keep the full (worker index, exception, busy) for each still-retained worker so the terminal log
            # and degradation reason carry the actual failure detail, not just worker indices (diagnostics).
            retained = [(index, exc, is_busy) for index, exc, is_retained, is_busy in outcomes if is_retained]
            retained_indices = [index for index, _exc, _busy in retained]
            if not retained_indices:
                # No worker still maps the scene. If it was never loaded on the owner (e.g. a seeded bootstrap
                # with no owner load, or a load that failed before registering) there is nothing to free — a
                # clean terminal state. Safe ONLY because worker-side tensors can come solely from
                # `registry.attach`, which requires a prior `ensure_loaded` — so not-loaded ⇒ no worker
                # ever mapped this scene's tensors (implicit coupling to OwnerSceneRegistry.ensure_loaded).
                if not self._owner_registry.is_loaded(scene_id):
                    return _DetachResult.RELEASED
                # It is loaded and no worker retains it, so its refcount is fully dropped: release must free it.
                released = self._owner_registry.release(scene_id)
                if released:
                    if attempt > 1 or errors:
                        log.info(
                            f"scene {scene_id!r} detach recovered on attempt {attempt}/{attempts} "
                            f"(transient errors: {[str(exc) for _index, exc in errors]}); owner tensors released"
                        )
                    return _DetachResult.RELEASED
                # Loaded, no worker retains it, yet release still reports not-freed: an ownership-accounting
                # invariant failure (e.g. a refcount leak). Degrade rather than free-then-hope or spin.
                reason = (
                    f"scene {scene_id!r} release returned False on attempt {attempt}/{attempts} with no worker "
                    "retaining its mapping; owner refcount accounting is inconsistent; retaining the residency "
                    "budget charge until fleet reap"
                )
                with self._lock:
                    self._mark_degraded_locked(reason)
                log.error(f"scene residency terminal action: quarantine {scene_id!r} (invariant); reason={reason}")
                return _DetachResult.QUARANTINED
            # If EVERY still-retained worker is BUSY (an explicit in-use refusal — the F001 orphaned-render
            # case), this is NOT detach uncertainty: refcounts are correct and the mapping is intentionally
            # kept. Do NOT retry-spin (a rapid retry cannot change an in-flight render) and do NOT degrade —
            # return BUSY so eviction picks another victim or fails the requester retryable while the fleet
            # stays SERVING. This is the fix for the escalation Wave 1a's report flagged: without it, three
            # rapid busy refusals would (mis)quarantine + degrade the whole fleet on one orphaned render.
            if all(is_busy for _index, _exc, is_busy in retained):
                busy_detail = "; ".join(f"worker {index}: {exc!r}" for index, exc, _busy in retained)
                log.info(
                    f"scene {scene_id!r} detach deferred (attempt {attempt}/{attempts}): a render is still in "
                    f"flight (in-use) on [{busy_detail}]; refcounts correct — retry later / pick another victim "
                    f"(fleet stays SERVING, no degradation)"
                )
                return _DetachResult.BUSY
            retained_detail = "; ".join(f"worker {index}: {exc!r}" for index, exc, _busy in retained)
            log.warning(
                f"scene {scene_id!r} detach unconfirmed (attempt {attempt}/{attempts}) on [{retained_detail}]; "
                f"a live worker may still map it — retrying"
            )
        retained_detail = "; ".join(f"worker {index}: {exc!r}" for index, exc, _busy in retained)
        reason = (
            f"scene {scene_id!r} detach could not be confirmed after {attempts} attempts on [{retained_detail}]; "
            "quarantining owner tensors and retaining the residency budget charge until fleet reap "
            "(a live worker may still map them)"
        )
        with self._lock:
            self._mark_degraded_locked(reason)
        log.error(f"scene residency terminal action: quarantine {scene_id!r}; reason={reason}")
        return _DetachResult.QUARANTINED

    def _attach_all(self, scene_id: str, artifact_path: str) -> None:
        # Fan the attach out to every worker in parallel (so the barrier is one worker-load wide, not N), then
        # raise on any failure. Rollback of the succeeded/uncertain attachments is NOT done here: it is the
        # single centralized `_detach_release_or_quarantine` path driven by `ensure_attached`'s failure
        # cleanup, so an uncertain detach during rollback quarantines + degrades instead of being suppressed.
        if not self._channels:
            return
        errors: list[Exception] = []
        with futures.ThreadPoolExecutor(max_workers=len(self._channels)) as pool:
            future_to_channel = {
                pool.submit(channel.attach, scene_id, artifact_path): channel for channel in self._channels
            }
            for future in futures.as_completed(future_to_channel):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - aggregate all worker failures; cleanup is centralized
                    errors.append(exc)
        if errors:
            raise SceneAttachError(f"scene {scene_id!r} failed to attach on {len(errors)} worker(s): {errors}")

    def _load_owner(self, scene_id: str, artifact_path: str) -> Any:
        """Translate only the injected artifact/backend loader boundary into a scene-domain error.

        Registry publication happens in ``OwnerSceneRegistry.ensure_loaded`` after this method returns and is
        intentionally outside this boundary: bookkeeping/invariant defects must remain loud internal errors.
        """
        try:
            return self._loader(scene_id, artifact_path)
        except Exception as exc:
            raise SceneLoadError(f"scene {scene_id!r} failed to load from artifact {artifact_path!r}: {exc}") from exc


class OnDemandSceneRegistrar:
    """Proxy/owner-side on-demand scene registration.

    Resolves a client-supplied ``(scene_url, scene_id)`` to a local artifact path — downloading a remote URI
    into the `SceneCache` (once, via the cache file lock) or resolving a ``file://`` / local path — then
    registers ``scene_id -> path`` with the `SceneCoordinator`. The normal flow (first-load on the
    owner + the all-worker attach barrier, driven by the render's `_scene_lease` →
    `SceneCoordinator.acquire`) then serves it. Resolution
    is skipped for a scene the coordinator already knows (the primary or a previously-downloaded scene), so a
    repeat request does not re-download. Without a `scene_cache`/`downloader` only local URIs are accepted.
    """

    def __init__(
        self,
        coordinator: "SceneCoordinator",
        *,
        scene_cache: "SceneCache | None" = None,
        downloader: "Downloader | None" = None,
    ) -> None:
        self._coordinator = coordinator
        self._scene_cache = scene_cache
        self._downloader = downloader
        self._registration_locks_guard = threading.Lock()
        self._registration_locks: dict[str, tuple[threading.Lock, int]] = {}

    @contextlib.contextmanager
    def _registration_lock(self, scene_id: str):
        """Serialize one scene's resolve+publish transaction and prune unused locks safely."""
        with self._registration_locks_guard:
            lock, users = self._registration_locks.get(scene_id, (threading.Lock(), 0))
            self._registration_locks[scene_id] = (lock, users + 1)
        try:
            with lock:
                yield
        finally:
            with self._registration_locks_guard:
                current_lock, users = self._registration_locks[scene_id]
                if current_lock is lock and users == 1:
                    self._registration_locks.pop(scene_id)
                else:
                    self._registration_locks[scene_id] = (current_lock, users - 1)

    def register_scene_from_uri(self, scene_id: str, scene_url: str) -> None:
        with self._registration_lock(scene_id):
            # Re-check only after taking the per-scene lock: the first competing request may have published a
            # registration while this one waited. This makes source conflict selection deterministic and
            # prevents one URL from claiming bytes downloaded for another URL under the same public id.
            is_local = is_local_scene_uri(scene_url)
            registration = self._coordinator.registration(scene_id)
            if registration is not None:
                artifact_path, registered_source, generation = registration
                # Startup/glob registrations have no client source URI. They are still a live registration,
                # so a header may only name the exact same local artifact; a remote or different local source
                # is a conflicting re-registration, not a silent no-op.
                if registered_source is None:
                    if is_local:
                        requested_path = resolve_scene_artifact_path(scene_url, scene_id)
                        if requested_path == Path(artifact_path).resolve():
                            return
                    raise ValueError(
                        f"scene {scene_id!r} is already registered from startup artifact {artifact_path!r}; "
                        f"requested a different source {scene_url!r}"
                    )
                # On-demand re-registration with a recorded source. For a LOCAL source, compare RESOLVED
                # paths (not raw source strings): an equivalent-but-differently-spelled local URI
                # (`file:///tmp/a/../scene.usdz` vs `/tmp/scene.usdz`) resolves to the same registered
                # artifact and must be a no-op, mirroring single-process `_register_local_scene` and the
                # startup/glob branch above — rather than raising a raw-string conflict via
                # `check_source_conflict`. A genuinely different local file still resolves to a different
                # path and falls through to the conflict check below.
                if is_local:
                    requested_path = resolve_scene_artifact_path(scene_url, scene_id)
                    if requested_path == Path(artifact_path).resolve():
                        return
                self._coordinator.check_source_conflict(scene_id, scene_url)
                if is_local or generation is None:
                    return
                if self._scene_cache is None or self._downloader is None:
                    raise ValueError("a remote scene URL requires a scene_cache + downloader")
                if self._scene_cache.is_current(scene_id, generation, scene_url):
                    return
                # The cache entry was already popped (possibly while its callback is delayed). Compare-delete
                # the old generation before resolving a replacement; a later old callback is then harmless.
                self._coordinator.invalidate_scene(scene_id, generation)
            elif self._coordinator.is_known(scene_id):
                # An attached seed without an artifact mapping (e.g. legacy bootstrap) remains a no-op.
                self._coordinator.check_source_conflict(scene_id, scene_url)
                return

            if is_local:
                path = resolve_scene_artifact_path(scene_url, scene_id)
                self._coordinator.register_scene(scene_id, str(path), source=scene_url)
                return

            if self._scene_cache is None or self._downloader is None:
                raise ValueError("a remote scene URL requires a scene_cache + downloader")

            # Cache reuse is source-bound. An unbound or differently-sourced orphan cannot satisfy this
            # request; remove its exact generation first, then let the shared resolver download fresh bytes.
            try:
                cached = self._scene_cache.get_scene_value(scene_id, update_access_time=False)
            except KeyError:
                cached = None
            if cached is not None and (cached.source != scene_url or not cached.path.exists()):
                self._scene_cache.remove_scene(scene_id, generation=cached.generation)
                cached = None

            if cached is None:
                path = resolve_scene_artifact_path(
                    scene_url, scene_id, scene_cache=self._scene_cache, downloader=self._downloader
                )
                cached = self._scene_cache.bind_scene_source(scene_id, path, scene_url)
            else:
                path = self._scene_cache.get_scene_path(scene_id)

            self._coordinator.register_scene(
                scene_id,
                str(path),
                source=scene_url,
                generation=cached.generation,
            )
            # Eviction callbacks intentionally run outside the cache lock. If this generation disappeared in
            # the resolve→publish window, compare-and-delete the just-published mapping before returning. A
            # concurrent request may fail/retry, but no stale known path survives.
            if not self._scene_cache.is_current(scene_id, cached.generation, scene_url):
                self._coordinator.invalidate_scene(scene_id, cached.generation)
                raise ValueError(
                    f"scene {scene_id!r} cache generation {cached.generation} was evicted while registering; retry"
                )


class SceneDownloadInterceptor(grpc.ServerInterceptor):
    """Proxy-side ``x-nre-scene-url`` / ``x-nre-scene-id`` handler, mirroring `serve`'s interceptor.

    Where `serve` downloads/loads the scene into a local `SensorSimService`, the proxy instead registers it
    with the `SceneCoordinator` (via `OnDemandSceneRegistrar`); the owner loads + IPC-exports it lazily on the
    first render's attach barrier. Runs on every RPC: a no-op (cheap dict check + continuation) when the
    header is absent. An invalid scene id, or a download/resolve failure, aborts the RPC `INVALID_ARGUMENT`
    (matching `serve`) so the scene is never half-registered.
    """

    def __init__(self, registrar: "OnDemandSceneRegistrar | None" = None) -> None:
        self._registrar = registrar

    def set_registrar(self, registrar: "OnDemandSceneRegistrar") -> None:
        self._registrar = registrar

    def _terminate(self, code: grpc.StatusCode, details: str) -> grpc.RpcMethodHandler:
        def _terminator(ignored_request, context):
            context.abort(code, details)

        return grpc.unary_unary_rpc_method_handler(_terminator)

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)
        if "x-nre-scene-url" in metadata and self._registrar is not None:
            scene_url = metadata["x-nre-scene-url"]
            scene_id = metadata.get("x-nre-scene-id", "")
            if not check_safe_scene_id(scene_id):
                return self._terminate(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid scene ID: {scene_id}")
            try:
                self._registrar.register_scene_from_uri(scene_id, scene_url)
            except Exception as exc:  # noqa: BLE001 - surface any resolve/download failure as INVALID_ARGUMENT
                log.exception(f"Failed to register scene {scene_id} from {scene_url}")
                return self._terminate(
                    grpc.StatusCode.INVALID_ARGUMENT, f"Failed to register scene from {scene_url}: {exc}"
                )
        return continuation(handler_call_details)


class WorkerStopReason(enum.Enum):
    """Why a worker stopped serving (returned by `_serve_until_stop`).

    Distinct reasons keep an owner-directed graceful stop separate from abnormal owner death and from a
    control-plane failure. Worker liveness is OS-derived — there is no heartbeat, timeout-based lease,
    missed-message counter, or live-parent stall deadline."""

    OWNER_DEATH = "model-owner process exited (parent-process sentinel ready)"
    STOP = "owner directed the worker to stop via the lifecycle channel"
    CONTROL_PLANE_FAILURE = "worker lifecycle channel closed or invalid while the owner was still alive"

    @property
    def exit_code(self) -> int:
        """Process exit code for a worker that stopped for this reason.

        A control-plane failure is a fault the worker must surface nonzero, so the owner observes the worker
        death and its own loop returns ``WORKER_EXIT`` (nonzero). A graceful owner-directed stop, and stopping
        because the owner is already gone, are both clean worker exits (0)."""
        return 1 if self is WorkerStopReason.CONTROL_PLANE_FAILURE else 0


# Sentinel distinguishing "argument not supplied" (derive from the multiprocessing parent) from an
# explicit ``None`` (no parent signal available — a direct unit-test call in the main process).
_DERIVE_PARENT_SIGNALS: Any = object()


def _owner_parent_signals() -> "tuple[int | None, Callable[[], bool] | None]":
    """Return the owner process's death sentinel fd + liveness probe, or ``(None, None)`` if unavailable.

    A spawned worker's multiprocessing parent process IS the owner; its ``sentinel`` fd becomes readable
    the instant the owner exits or is killed (including SIGKILL) — the authoritative owner-death signal,
    independent of any application message or Python thread scheduling. ``is_alive()`` distinguishes a
    live owner (a lifecycle-channel EOF is then a control-plane failure) from a dead one.

    Returns ``(None, None)`` only when there is no multiprocessing parent (e.g. a direct call in the main
    process). A real spawned worker always has one; `_worker` treats ``None`` as a fatal startup condition
    and refuses to serve rather than silently degrading to lifecycle-channel-only supervision."""
    parent = parent_process()
    if parent is None:
        return None, None
    try:
        sentinel = parent.sentinel
    except (AttributeError, ValueError, OSError):
        return None, None
    return sentinel, parent.is_alive


def _serve_until_stop(
    server: grpc.Server,
    lifecycle_rx: Any,
    *,
    shutdown_grace_s: float = _SHUTDOWN_GRACE_S,
    parent_sentinel: Any = _DERIVE_PARENT_SIGNALS,
    is_parent_alive: Any = _DERIVE_PARENT_SIGNALS,
) -> WorkerStopReason:
    """Serve until the owner dies or directs a stop; then stop `server` and return the typed reason.

    This is the Lifecycle gate. The owner process holds the IPC-shared weights, so the worker must stop
    serving promptly on owner death instead of rendering from a now-dangling mapping. Liveness is derived
    from the OS, not from application traffic: the worker blocks on ``multiprocessing.connection.wait`` over
    the owner's **parent-process death sentinel** and the one-way **lifecycle channel**, with no heartbeat,
    timeout-based lease, missed-message counter, or live-parent stall deadline.

    Resolution:
    - parent sentinel ready → ``OWNER_DEATH`` (authoritative, immediate — including owner SIGKILL);
    - ``STOP`` on the lifecycle channel → ``STOP`` (owner-directed graceful stop);
    - lifecycle EOF/bad payload while the parent is still alive → ``CONTROL_PLANE_FAILURE`` (fail-closed).
      When the parent is already gone (or no parent signal is available) an EOF is treated as owner death.

    In production `_worker` always supplies the parent sentinel/liveness probe (it refuses to start without
    them). ``parent_sentinel=None`` is only for direct unit calls with no multiprocessing parent.

    Hard-kill limitation: stopping new service does not make an in-flight render safe — a render already
    running when the owner dies abruptly can read now-freed IPC pages before this observes the loss, because
    the owner and worker are separate processes with no cross-process render barrier. The drain window
    bounds but cannot eliminate this. See the API guide's multiprocess deployment/shutdown section.
    """
    if parent_sentinel is _DERIVE_PARENT_SIGNALS or is_parent_alive is _DERIVE_PARENT_SIGNALS:
        derived_sentinel, derived_is_alive = _owner_parent_signals()
        if parent_sentinel is _DERIVE_PARENT_SIGNALS:
            parent_sentinel = derived_sentinel
        if is_parent_alive is _DERIVE_PARENT_SIGNALS:
            is_parent_alive = derived_is_alive

    reason = _wait_for_worker_stop_reason(lifecycle_rx, parent_sentinel, is_parent_alive)
    log.info(f"worker lifecycle wait resolved: {reason.value}")
    stopped = server.stop(shutdown_grace_s)
    stopped.wait(timeout=shutdown_grace_s + 1.0)
    return reason


def _wait_for_worker_stop_reason(
    lifecycle_rx: Any,
    parent_sentinel: Any,
    is_parent_alive: "Callable[[], bool] | None",
) -> WorkerStopReason:
    """Block on the parent sentinel + lifecycle channel until one resolves the worker's stop reason.

    No timeout and no polling: ``multiprocessing.connection.wait`` sleeps until the owner exits (sentinel
    ready) or the owner writes/closes the lifecycle channel. Request traffic is deliberately not part of
    this decision — worker lifetime depends only on OS owner-liveness and explicit lifecycle control."""
    wait_objects: list = [lifecycle_rx]
    if parent_sentinel is not None:
        wait_objects.append(parent_sentinel)

    while True:
        ready = wait_for_ready_connections(wait_objects)
        # Authoritative, immediate owner death takes precedence: the parent-process sentinel is readable.
        if parent_sentinel is not None and parent_sentinel in ready:
            return WorkerStopReason.OWNER_DEATH
        if lifecycle_rx in ready:
            try:
                command = recv_worker_lifecycle(lifecycle_rx)
            except WorkerLifecycleError:
                # EOF or a bad payload. A live owner means the control channel broke under it -> fail closed;
                # a dead/absent owner means this is just owner death observed via the channel.
                if is_parent_alive is not None and is_parent_alive():
                    return WorkerStopReason.CONTROL_PLANE_FAILURE
                return WorkerStopReason.OWNER_DEATH
            if command is WorkerLifecycleCommand.STOP:
                return WorkerStopReason.STOP
            # A well-typed but unhandled lifecycle command is a control-plane contract violation.
            return WorkerStopReason.CONTROL_PLANE_FAILURE


def _renderer_type(name: str) -> "Backend.RendererType":
    return {
        "gsplat": Backend.RendererType.GSPLAT,
        "default": Backend.RendererType.DEFAULT,
    }[name]


def _renderer_name_from_backend(renderer: RendererBackend) -> str:
    """Map a `GrpcServerConfig.renderer` (the `serve-grpc` knob) to the multiproc renderer name."""
    return {
        RendererBackend.GSPLAT: "gsplat",
        RendererBackend.DEFAULT: "default",
    }[renderer]


# The multi-process pool is validated **only** for the gsplat renderer.
_SUPPORTED_RENDERER = "gsplat"


def _require_supported_renderer(renderer_name: str) -> None:
    """Guard: the multi-process pool is validated **only** for gsplat. Reject anything else, fast,
    before any worker is spawned.

    The `default` (PyTorch forward fallback) is not validated through the pool either.
    """
    if renderer_name != _SUPPORTED_RENDERER:
        raise ValueError(
            f"multi-process serving supports only the {_SUPPORTED_RENDERER!r} renderer; got {renderer_name!r}."
            f" Use the gsplat renderer (e.g. `--renderer gsplat`)."
        )


def _worker_metrics_dir(base: str, worker_id: int) -> str:
    """Per-worker metrics output dir under ``base``."""
    return os.path.join(base, f"worker_{worker_id}")


def merge_metrics_yaml(out_dir: str, source_dirs: "list[str]") -> bool:
    """Pool the per-worker (+ proxy) `metrics.yaml` raw samples into one merged `metrics.yaml`.

    Each source dir holds a `metrics.yaml` (dump of `MetricsFileContent`); we pool every
    sample value across sources -- both `metrics.general[<name>]` and the scene-attributed
    `metrics.per_sequence[<scene>].per_camera/per_lidar[<sensor>][<name>]` -- so fleet `render_time_ms`
    percentiles are computed over all workers' raw samples and keep the scene they were recorded under.
    Distinct metric names (worker `render_time_ms` vs proxy
    `grpc.e2e_sent.*`) stay separate. The pooled samples are fed back through a `MetricsCollector` so the
    merged file has the **full single-process shape** — raw `general` samples **and** `aggregated_metrics`
    (mean/min/max/std/…) — not just the raw samples. Missing/partial source files are skipped. Returns
    True if anything was written.
    """
    pooled: dict[str, list[float]] = {}
    pooled_per_sequence: dict[tuple[str, str, bool, str], list[float]] = {}
    pooled_metadata: dict[str, dict] = {}
    for source_dir in source_dirs:
        path = os.path.join(source_dir, "metrics.yaml")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                data = yaml.safe_load(handle) or {}
        except Exception as exc:
            # A corrupt/half-written worker metrics file is tolerated (one worker must not sink the fleet
            # merge), but log it per skip so a silently-dropped worker's samples are diagnosable.
            log.warning(f"merge_metrics_yaml: skipping unreadable metrics file {path!r}: {exc!r}")
            continue
        metrics = data.get("metrics") or {}
        for name, samples in ((metrics.get("general")) or {}).items():
            for sample in samples or []:
                if isinstance(sample, dict) and "value" in sample:
                    try:
                        pooled.setdefault(name, []).append(float(sample["value"]))
                    except (TypeError, ValueError):
                        continue
        # Scene-attributed samples land in per_sequence, not general; pooling only general would drop
        # every worker timing recorded against a scene.
        for scene_id, sensors in (metrics.get("per_sequence") or {}).items():
            for bucket, is_lidar in (("per_camera", False), ("per_lidar", True)):
                for sensor_id, by_name in ((sensors or {}).get(bucket) or {}).items():
                    for name, samples in (by_name or {}).items():
                        for sample in samples or []:
                            if isinstance(sample, dict) and "value" in sample:
                                try:
                                    value = float(sample["value"])
                                except (TypeError, ValueError):
                                    continue
                                key = (scene_id, sensor_id, is_lidar, name)
                                pooled_per_sequence.setdefault(key, []).append(value)
        for name, meta in (data.get("metric_metadata") or {}).items():
            if isinstance(meta, dict):
                pooled_metadata.setdefault(name, {}).update(meta)
    if not pooled and not pooled_per_sequence:
        return False
    # Re-emit through MetricsCollector so the merged file carries general samples + aggregates, matching
    # what single-process serve writes.
    collector = MetricsCollector(mode="grpc_server", run_id="multiproc")
    for name, meta in pooled_metadata.items():
        collector.set_metric_metadata(name, **meta)
    for name, values in pooled.items():
        for value in values:
            collector.save_metric(create_metric_sample(name, value))
    for (scene_id, sensor_id, is_lidar, name), values in pooled_per_sequence.items():
        for value in values:
            collector.save_metric(
                create_metric_sample(name, value),
                unique_sensor_id=sensor_id,
                sequence_id=scene_id,
                is_lidar=is_lidar,
            )
    os.makedirs(out_dir, exist_ok=True)
    collector.write_metrics_to_yaml(out_dir)
    return True


def mps_active() -> bool:
    """Best-effort check that the NVIDIA MPS control daemon is up (its `control` pipe exists).

    MPS is **mandatory** for multi-process throughput scaling — without it the workers
    time-slice the GPU and N>1 buys nothing. MPS is started externally; this only *detects* it, delegating
    to the stdlib-only detection helper (`nurec.utils.mps_prelaunch`) so every caller agrees on which pipe
    dirs count as "MPS up" (``$CUDA_MPS_PIPE_DIRECTORY`` if set, else both `/tmp/nvidia-mps` and
    `/tmp/nvidia-mps-{uid}`).
    """
    return mps_prelaunch.mps_active()


def mps_preflight(num_workers: int, require_mps: bool, progress: Callable[[str], None] = lambda _m: None) -> None:
    """Warn (or, with ``require_mps``, fail) when a multi-worker pool is launched without MPS.

    Single-worker pools don't need MPS, so this is a no-op for ``num_workers <= 1``.
    """
    if num_workers <= 1 or mps_active():
        return
    message = (
        f"MPS not detected but --multiproc-workers={num_workers}: renders will time-slice the GPU and "
        "throughput will NOT scale. Start the daemon: `nvidia-cuda-mps-control -d`."
    )
    if require_mps:
        raise RuntimeError(message + " (--require-mps set)")
    progress(f"WARNING: {message}")


# NOTE: the MPS control daemon must be started externally (operator / container entrypoint / k8s
# initContainer / systemd) before this server launches — CUDA is initialized at *import* time, so MPS has
# to be up before then to route this process (and its workers) through it. This module never starts MPS; it
# only detects it (`mps_active`) and verifies it here with `mps_preflight`.


def _resolve_download_cache_size(
    download_cache_size: int,
    max_scenes: int,
    *,
    require_scene_cache: bool = False,
    progress: Callable[[str], None] = lambda _m: None,
) -> int:
    """Return the effective on-disk download-cache size, guarding the re-attach edge.

    An evicted (but still-registered) **remote** scene keeps its artifact mapping for re-attach; if the
    on-disk `SceneCache` (LRU, `download_cache_size`) is smaller than the resident-scene budget (`max_scenes`
    = `cache_size`), that scene's downloaded file can be dropped from disk before re-attach, which then fails.
    The on-disk cache should therefore hold at least as many files as can be resident.

    **Automatic alignment:** by default an undersized cache is raised to the resident-scene budget and the
    adjustment is reported at INFO, preserving the remote re-registration invariant. Local `file://` scenes
    are not stored in the download cache and are unaffected. With ``require_scene_cache`` an operator can
    opt into a hard error instead. A no-op when eviction is off (``max_scenes <= 0``) or the cache is already
    large enough.
    """
    if max_scenes <= 0 or download_cache_size >= max_scenes:
        return download_cache_size
    message = (
        f"download_cache_size={download_cache_size} < max_scenes={max_scenes} (= cache_size): an evicted "
        f"remote scene's cached file could be dropped from disk before re-attach. Local file:// "
        f"scenes are unaffected."
    )
    if require_scene_cache:
        raise ValueError(message + " (--multiproc-require-scene-cache set)")
    progress(f"INFO: {message} Auto-bumping download_cache_size -> {max_scenes}.")
    return max_scenes


def validate_multiproc_admission(
    *,
    max_in_flight: int,
    admission_timeout_s: float,
) -> "str | None":
    """Validate the drop-in admission options.

    Returns an error message describing the first violation, or ``None`` when the combination is valid, so
    the Click layer can raise a `click.UsageError` and `run_drop_in` can raise a `ValueError` from the same
    contract. The render permit bounds only the active downstream worker render call — not the gRPC executor
    queue, download, owner load, or the coordinator/worker attach barrier.

    Contract:
    - ``max_in_flight`` >= 0 (0 = unlimited active downstream renders).
    - ``admission_timeout_s`` finite and >= 0 (0 = shed immediately). A nonzero timeout requires a nonzero
      ``max_in_flight`` (waiting for a slot is meaningless when admission is unlimited) — a nonzero timeout
      with unlimited admission is invalid rather than silently inert.
    """
    if max_in_flight < 0:
        return f"--multiproc-max-in-flight must be >= 0 (0 = unlimited); got {max_in_flight}"
    if not math.isfinite(admission_timeout_s) or admission_timeout_s < 0:
        return (
            "--multiproc-admission-timeout-s must be a finite value >= 0 (0 = shed immediately); "
            f"got {admission_timeout_s}"
        )
    if admission_timeout_s > 0 and max_in_flight == 0:
        return (
            "--multiproc-admission-timeout-s > 0 requires --multiproc-max-in-flight > 0 "
            "(a nonzero wait is meaningless when render admission is unlimited)"
        )
    return None


def validate_multiproc_harmonizer(grpc_config: GrpcServerConfig) -> "str | None":
    """Reject the temporal harmonizer in multiprocess mode.

    Temporal-harmonizer sessions are per-process state (a frame-history dict inside one service
    instance): the round-robin proxy would create a session on one worker and land subsequent
    renders on others, returning NOT_FOUND or silently corrupting the temporal conditioning.
    The stateless ``--enable-harmonizer`` IS supported — each worker applies it per frame — so
    only the temporal variant is refused. Returns an error message, or ``None`` when valid, from
    the same contract as `validate_multiproc_admission` (Click raises `click.UsageError`,
    `run_drop_in` raises `ValueError`).

    ``enable_harmonizer_temporal`` is a real `GrpcServerConfig` field (default False); the
    ``getattr`` stays only so programmatic callers may pass a config-like object without the field
    (treated as disabled).
    """
    if getattr(grpc_config, "enable_harmonizer_temporal", False):
        return (
            "--enable-harmonizer-temporal is not supported with --multiproc-workers > 0: harmonizer "
            "sessions are per-process state the multiprocess proxy cannot route consistently. Use "
            "single-process serving for temporal harmonization (the stateless --enable-harmonizer "
            "works with the pool)."
        )
    return None


def validate_multiproc_mps(*, num_workers: int, require_mps: bool) -> "str | None":
    """Pre-validate the MPS requirement so `--require-mps` on a multi-worker host without MPS active fails as
    a clean `click.UsageError` at launch, instead of a raw RuntimeError traceback from `mps_preflight` deep
    inside `run_drop_in`. Returns the error message (matching `mps_preflight`'s `--require-mps` hard-fail),
    or ``None`` when MPS is not required, the pool is single-worker, or MPS is active. Same contract as
    `validate_multiproc_admission`: the Click layer raises `click.UsageError`, while `run_drop_in`'s
    `mps_preflight` still raises for programmatic callers."""
    if not require_mps or num_workers <= 1 or mps_active():
        return None
    return (
        f"MPS not detected but --multiproc-workers={num_workers}: renders will time-slice the GPU and "
        "throughput will NOT scale. Start the daemon: `nvidia-cuda-mps-control -d`. (--require-mps set)"
    )


def validate_multiproc_download_cache(grpc_config: GrpcServerConfig, *, require_scene_cache: bool) -> "str | None":
    """Pre-validate the on-disk download-cache floor so `--multiproc-require-scene-cache` with an undersized
    `--download-cache-size` fails as a clean `click.UsageError` at launch, instead of a raw ValueError
    traceback from `_resolve_download_cache_size` inside `run_drop_in`. Returns the error message (matching
    that hard-fail), or ``None`` when no hard floor was requested, eviction is off, or the cache is already
    large enough (the default auto-bump path). Same contract as `validate_multiproc_admission`."""
    if not require_scene_cache:
        return None
    max_scenes = grpc_config.cache_size
    download_cache_size = grpc_config.download_cache_size
    if max_scenes <= 0 or download_cache_size >= max_scenes:
        return None
    return (
        f"download_cache_size={download_cache_size} < max_scenes={max_scenes} (= cache_size): an evicted "
        f"remote scene's cached file could be dropped from disk before re-attach. Local file:// "
        f"scenes are unaffected. (--multiproc-require-scene-cache set)"
    )


class _AdmissionRejected(Exception):
    """Raised by `RenderProxy._admit()` when no in-flight render slot is free within the admission timeout.

    Carries the public shed `details` so the two admission call sites derive from one implementation and
    cannot drift: a singular RPC translates it to `abort(RESOURCE_EXHAUSTED, details)`, while a batch item
    turns it into that item's `success=false` result. It is internal control flow, never surfaced as-is."""

    def __init__(self, details: str) -> None:
        super().__init__(details)
        self.details = details


class _ForwardedCallError(grpc.RpcError):
    """A downstream worker-call outcome that grpc raises as a **non-`grpc.RpcError`** future exception,
    normalized to look like a worker `RpcError` (with `code()`/`details()`).

    `stub.method.future(...).result()` raises `grpc.FutureCancelledError` when the proxy cancels the call
    (client deadline/cancel mid-forward) and `grpc.FutureTimeoutError` on a future deadline — neither is a
    `grpc.RpcError` subclass, so the relay's `except grpc.RpcError` would miss them and the exception would
    escape `render_rgb` as an uncaught "Exception calling application" ERROR traceback. Wrapping them in this
    `RpcError` lets the singular relay (`_relay_rpc_error`) and the per-item batch path (`_extract_worker_rpc_error`)
    surface them as a clean CANCELLED / DEADLINE_EXCEEDED instead."""

    def __init__(self, code: "grpc.StatusCode", details: str) -> None:
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self) -> "grpc.StatusCode":
        return self._code

    def details(self) -> str:
        return self._details


class RenderProxy(SensorsimServiceServicer):
    """Round-robins the render RPCs — `render_camera`/`batch_render_camera` and the deprecated
    `render_rgb`/`batch_render_rgb`, plus `render_lidar` — across worker stubs; forwards discovery to worker 0.

    Construct with `from_addresses(...)` in production (opens one gRPC channel per worker); the bare
    constructor takes pre-built stubs so the dispatch logic is unit-testable without real servers.
    """

    def __init__(
        self,
        stubs: list,
        channels: list | None = None,
        max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
        admission_timeout_s: float = _DEFAULT_ADMISSION_TIMEOUT_S,
        shutdown_event: Any = None,
        server_config_map: "dict[str, str] | None" = None,
        metrics_manager: Any = None,
        scene_coordinator: "SceneCoordinator | None" = None,
    ) -> None:
        if not stubs:
            raise ValueError("RenderProxy requires at least one worker stub")
        self._stubs = list(stubs)
        self._channels = list(channels) if channels is not None else []  # kept alive for the proxy's lifetime
        self._shutdown_event = shutdown_event  # set by shut_down() so the owner main loop tears down the fleet
        self._server_config_map = server_config_map  # effective launch config + runtime status
        configured_image_formats = server_config_map.get("allowed_image_formats") if server_config_map else None
        if configured_image_formats == "null":
            configured_image_formats = None
        self._allowed_image_formats = allowed_image_format_values(configured_image_formats)
        self._metrics_manager = metrics_manager  # proxy-side ServerMetricsManager for batch metrics
        self._scene_coordinator = scene_coordinator  # multi-scene gate; None -> single-scene passthrough
        self._round_robin = itertools.cycle(range(len(self._stubs)))
        self._lock = threading.Lock()
        # Admission control: cap concurrent in-flight renders fleet-wide so peak working-set VRAM
        # stays bounded (not growing with offered load); over-limit requests shed RESOURCE_EXHAUSTED.
        self._max_in_flight = max_in_flight
        self._admission_timeout_s = admission_timeout_s
        self._inflight = threading.BoundedSemaphore(max_in_flight) if max_in_flight > 0 else None

    @classmethod
    def from_addresses(
        cls,
        worker_addresses: list[str],
        max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
        admission_timeout_s: float = _DEFAULT_ADMISSION_TIMEOUT_S,
        shutdown_event: Any = None,
        server_config_map: "dict[str, str] | None" = None,
        metrics_manager: Any = None,
        scene_coordinator: "SceneCoordinator | None" = None,
    ) -> "RenderProxy":
        channels = [grpc.insecure_channel(addr, options=_CHANNEL_OPTIONS) for addr in worker_addresses]
        return cls(
            [SensorsimServiceStub(channel) for channel in channels],
            channels=channels,
            max_in_flight=max_in_flight,
            admission_timeout_s=admission_timeout_s,
            shutdown_event=shutdown_event,
            server_config_map=server_config_map,
            metrics_manager=metrics_manager,
            scene_coordinator=scene_coordinator,
        )

    def close(self) -> None:
        """Close the worker gRPC channels held for the proxy's lifetime.

        `serve_proxy` calls this on teardown. The single-shot `main()`/`run_drop_in` flow exits the process
        anyway, but `serve_proxy` is shared with the conformance smokes and a long-lived embedder that builds
        a proxy per call would otherwise leak a channel set (fds + background gRPC threads) each time.
        Best-effort + idempotent."""
        for channel in self._channels:
            with contextlib.suppress(Exception):
                channel.close()
        self._channels = []

    def _next_stub(self):
        # Hold the lock only to pick the next worker — NOT during the forward — so concurrent renders
        # fan out and overlap instead of serializing behind one another.
        with self._lock:
            return self._stubs[next(self._round_robin)]

    @contextlib.contextmanager
    def _scene_lease(self, scene_id: "str | None", context, *, reject=None):
        """Ensure ``scene_id`` is attached pool-wide and hold a **lease** on it for the duration of the
        wrapped forward, so coordinated eviction cannot evict the scene mid-render (the attach barrier
        + lease). A no-op on a single-scene pool (no coordinator).

        On a lease rejection it calls ``reject(code, message)`` — which MUST NOT return — with NOT_FOUND
        (unknown scene), UNAVAILABLE (attach barrier failed + rolled back), or RESOURCE_EXHAUSTED (scene
        budget full, no idle scene to evict), in each case before the lease is taken, so the `finally`
        release only runs for a successfully-leased scene. ``reject`` defaults to ``context.abort`` (the
        render/discovery paths surface a typed gRPC status)."""
        coordinator = self._scene_coordinator
        if coordinator is None:
            yield
            return
        # Callers pass a None scene_id only on a coordinator-less (single-scene) pool, handled above.
        assert scene_id is not None

        def _fail(code: "grpc.StatusCode", message: str) -> None:
            # Lazy: only touch ``context.abort`` on an actual rejection — the happy-path dispatch callers may
            # pass ``context=None`` and must not fault on entry (the reject translator, when supplied, raises).
            (reject if reject is not None else context.abort)(code, message)

        try:
            coordinator.acquire(scene_id)
        except SceneNotKnownError as exc:
            _fail(grpc.StatusCode.NOT_FOUND, str(exc))
        except ScenePoolDegradedError as exc:
            # The scene control plane is degraded and the fleet is restarting; fail bounded rather
            # than routing/over-admitting on an uncertain fleet.
            _fail(grpc.StatusCode.UNAVAILABLE, str(exc))
        except SceneLoadError as exc:
            _fail(grpc.StatusCode.UNAVAILABLE, str(exc))
        except SceneAttachError as exc:
            _fail(grpc.StatusCode.UNAVAILABLE, str(exc))
        except SceneBudgetError as exc:
            _fail(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        try:
            yield
        finally:
            coordinator.release_scene(scene_id)

    @staticmethod
    def _client_cancelled(context) -> bool:
        """True when the client has cancelled/terminated the RPC, so a queued admission wait can bail early.

        Best-effort: a lightweight test double (or any context without a callable ``is_active``) is treated
        as still active, so cancellation polling never spuriously sheds a live render."""
        if context is None:
            return False
        is_active = getattr(context, "is_active", None)
        if not callable(is_active):
            return False
        return not is_active()

    def _try_admit(self, context=None) -> bool:
        """Acquire an in-flight render slot, waiting up to `admission_timeout_s` (0 = don't wait). Returns
        whether a slot was acquired; always True when admission is unlimited.

        A positive wait is bounded by the client's remaining deadline (`min(admission_timeout_s,
        client_remaining)`) and polls cancellation, so a client that goes away while queued frees its handler
        thread promptly instead of pinning it for the full admission timeout — which, under load, can exhaust
        the bounded proxy executor. The default (`admission_timeout_s == 0`) is unchanged: a non-blocking
        acquire that sheds immediately."""
        if self._inflight is None:
            return True
        if self._admission_timeout_s <= 0:
            return self._inflight.acquire(blocking=False)  # 0 = shed at once, no wait / no deadline coupling
        deadline = time.monotonic() + self._admission_timeout_s
        client_timeout = self._client_timeout(context)
        if client_timeout is not None:
            deadline = min(deadline, time.monotonic() + client_timeout)
        while not self._client_cancelled(context):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False  # admission timeout or client deadline elapsed
            if self._inflight.acquire(timeout=min(remaining, _ADMISSION_CANCEL_POLL_INTERVAL_S)):
                return True
        return False  # client cancelled/abandoned the RPC while queued

    def _release_admission(self) -> None:
        if self._inflight is not None:
            self._inflight.release()

    @contextlib.contextmanager
    def _admit(self, context=None):
        """The single admission primitive shared by every render path: hold one in-flight render slot for the
        duration of a render so peak concurrent renders (peak working-set VRAM) stays bounded regardless of
        offered load. Raises `_AdmissionRejected` (carrying the public shed detail) if no slot frees within
        the admission timeout (or the client's shorter remaining deadline, or the client cancels); unlimited
        admission is a no-op. Releases the permit in `finally` ONLY after a successful acquisition, so a
        rejected acquire never releases a permit it did not take."""
        if not self._try_admit(context):
            raise _AdmissionRejected(f"render admission limit reached (max_in_flight={self._max_in_flight})")
        try:
            yield
        finally:
            self._release_admission()

    @contextlib.contextmanager
    def _admission(self, context):
        """Singular-RPC admission adapter over `_admit()`: translate an admission rejection into
        `context.abort(RESOURCE_EXHAUSTED, details)`. Batch items translate the same rejection into a per-item
        `success=false` result instead, but both go through the one `_admit()` primitive so their permit
        accounting and shed detail cannot drift."""
        try:
            with self._admit(context):
                yield
        except _AdmissionRejected as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, exc.details)

    def _lease_scene_id(self, request) -> "str | None":
        # Only resolve the request's scene_id on a multi-scene pool; a single-scene passthrough never
        # touches it (so the dispatch works for lightweight non-proto requests too).
        return request.scene_id if self._scene_coordinator is not None else None

    @staticmethod
    def _extract_worker_rpc_error(exc: "grpc.RpcError") -> "tuple[grpc.StatusCode, str]":
        """The single worker-RpcError decoder shared by the singular and batch paths, returning
        `(code, details)` with the fleet's canonical fallbacks: a missing/false code becomes `UNKNOWN` and
        missing/false details becomes `str(exc)`. Both public render paths derive their surfaced status/text
        from this so a downstream error can never render one way singularly and another way in a batch."""
        code = exc.code() or grpc.StatusCode.UNKNOWN  # type: ignore[attr-defined]
        details = exc.details() or str(exc)  # type: ignore[attr-defined]
        return code, details

    @classmethod
    def _relay_rpc_error(cls, exc: "grpc.RpcError", context) -> None:
        """Relay a worker's gRPC status (code + details) to the proxy's client. Without this, a downstream
        RpcError propagating out of a servicer method collapses to UNKNOWN, hiding the worker's real code
        (e.g. an unknown scene → NOT_FOUND, a bad timestamp → OUT_OF_RANGE)."""
        code, details = cls._extract_worker_rpc_error(exc)
        context.abort(code, details)

    @staticmethod
    def _client_timeout(context) -> "float | None":
        """The client's remaining **finite** deadline budget in seconds, or ``None`` when the client set no
        deadline (so the worker hop gets no deadline either).

        Forwarded as ``timeout=`` on every worker hop so the worker-side queued-shedding policy
        (`_grpc_backend_cache_wait_policy`) sees the true client budget, instead of a deadline-less second
        hop that keeps queuing/rendering for an already-abandoned request. This restores single-process
        parity: `serve` is deadline-aware only while a render is queued and runs to completion once
        rendering starts, and forwarding the budget lets each worker (itself a single-process
        `SensorSimService`) apply exactly that policy.

        CRITICAL: grpc's `ServicerContext.time_remaining()` is ``max(deadline - now, 0)`` and NEVER returns
        ``None`` — a client that set no deadline surfaces as grpc's inf-future sentinel (~9.2e18 s). Passing
        that as a downstream ``timeout=`` overflows grpc's deadline arithmetic into an *immediate*
        DEADLINE_EXCEEDED, so a deadline-less client would spuriously fail. Only a finite budget below
        `_MAX_FORWARDABLE_DEADLINE_S` is a real client deadline; the inf-future sentinel (or any non-finite
        value) means "no deadline" and must be forwarded as ``None``. A test context may return ``None``
        directly or omit the method entirely; both also mean "no deadline"."""
        if context is None:
            return None
        time_remaining = getattr(context, "time_remaining", None)
        if not callable(time_remaining):
            return None
        remaining = time_remaining()
        if remaining is None or not math.isfinite(remaining) or remaining > _MAX_FORWARDABLE_DEADLINE_S:
            return None  # inf-future sentinel / no real deadline -> no downstream deadline
        return remaining

    def _invoke(self, method, request, context):
        """Forward one unary RPC to a worker stub with deadline + cancellation propagation, **raising** any
        downstream `grpc.RpcError` (the caller decides whether to relay it as a typed abort or fold it into a
        per-item batch result).

        Passes ``timeout=`` = the client's finite remaining budget (see `_client_timeout`; ``None`` ⇒ no
        downstream deadline) so the worker sheds a *queued* render against the real budget, and wires
        ``context.add_callback`` to cancel the downstream call when the client goes away — so the proxy's
        admission permit and scene lease release instead of a handler thread blocking forever behind a
        wedged-but-alive worker. It does NOT interrupt a render already running on a worker (run-to-completion
        once rendering starts is the single-process policy and stays).

        Real gRPC stub multicallables expose ``.future`` (so cancellation can propagate); the lightweight
        test doubles expose a plain callable and fall back to a blocking call, carrying the ``timeout`` kwarg
        only when a deadline exists (a no-deadline forward passes no ``timeout``)."""
        timeout = self._client_timeout(context)
        future_call = getattr(method, "future", None)
        if callable(future_call):
            call = future_call(request, timeout=timeout)
            if context is not None:
                add_callback = getattr(context, "add_callback", None)
                if callable(add_callback):
                    # Fires when the client cancels/expires the RPC (and, harmlessly, on normal completion —
                    # cancelling an already-done call is a no-op). Frees this proxy's permit + lease promptly.
                    add_callback(call.cancel)
            try:
                return call.result()
            except grpc.FutureCancelledError as exc:
                # The cancellation callback above cancelled the downstream future (client deadline/cancel
                # mid-forward). `FutureCancelledError` is NOT a `grpc.RpcError`, so translate it to a clean
                # CANCELLED RpcError rather than letting it escape as an uncaught application exception.
                raise _ForwardedCallError(grpc.StatusCode.CANCELLED, "downstream render cancelled") from exc
            except grpc.FutureTimeoutError as exc:
                # Same normalization for a downstream future deadline (also not a `grpc.RpcError`).
                raise _ForwardedCallError(
                    grpc.StatusCode.DEADLINE_EXCEEDED, "downstream render deadline exceeded"
                ) from exc
        # Test-double fallback: a plain callable with no future API. Only pass timeout= when the client set a
        # deadline, so a no-deadline forward is a single-argument call.
        if timeout is None:
            return method(request)
        return method(request, timeout=timeout)

    def _forward(self, method, request, context):
        """`_invoke` + typed relay: forward the RPC and, on a downstream `grpc.RpcError`, relay the worker's
        real status (code + details) to the client via ``context.abort`` — so a downstream DEADLINE_EXCEEDED /
        NOT_FOUND / etc. surfaces as that typed status, never as UNKNOWN "Exception calling application" from
        an uncaught rendezvous. Used by the singular render + every discovery forward. The batch path calls
        `_invoke` directly so a downstream error becomes that item's ``success=false`` rather than a
        whole-batch abort. When no relay target is available (``context`` is None / has no ``abort``, e.g. a
        lightweight unit call) the error is re-raised unchanged."""
        try:
            return self._invoke(method, request, context)
        except grpc.RpcError as exc:
            if context is not None and callable(getattr(context, "abort", None)):
                self._relay_rpc_error(exc, context)  # -> context.abort(code, details); raises
            raise  # no relay target, or abort somehow returned: propagate the original error

    def render_rgb(self, request, context):
        require_allowed_image_format(request, self._allowed_image_formats, context)
        if getattr(request, "use_cuda_ipc", False):
            # open_ipc_session is always refused under multi-process serving (see
            # below), so ipc_session_id can never be valid here. Abort the same clear,
            # typed UNIMPLEMENTED the dedicated IPC RPCs give, rather than forwarding to
            # an arbitrary worker and letting it fail with an opaque "session not found".
            context.abort(grpc.StatusCode.UNIMPLEMENTED, self._CUDA_IPC_MULTIPROC_UNSUPPORTED)
        with self._scene_lease(self._lease_scene_id(request), context):  # attach barrier + lease
            with self._admission(context):
                return self._forward(self._next_stub().render_rgb, request, context)

    def render_lidar(self, request, context):
        with self._scene_lease(self._lease_scene_id(request), context):
            with self._admission(context):
                return self._forward(self._next_stub().render_lidar, request, context)

    def render_camera(self, request, context):
        # The current (non-deprecated) singular camera render; routed exactly like `render_rgb`.
        require_allowed_image_format(request, self._allowed_image_formats, context)
        if getattr(request, "use_cuda_ipc", False):
            # Same reasoning as render_rgb: open_ipc_session is always refused under
            # multi-process serving, so ipc_session_id can never be valid here.
            context.abort(grpc.StatusCode.UNIMPLEMENTED, self._CUDA_IPC_MULTIPROC_UNSUPPORTED)
        with self._scene_lease(self._lease_scene_id(request), context):
            with self._admission(context):
                return self._forward(self._next_stub().render_camera, request, context)

    def batch_render_rgb(self, request, context):
        """DEPRECATED batch RGB render. See `_batch_render` for the shared fan-out contract."""
        self._require_allowed_batch_image_formats(request, context, single_item_rpc="render_rgb")
        return self._batch_render(
            request,
            context,
            worker_rpc="render_rgb",
            return_cls=sensorsim_pb2.BatchRGBRenderReturn,
            item_cls=sensorsim_pb2.BatchRGBRenderReturnItem,
            # Main consolidated the batch counters: `record_batch_rgb_render_time` is gone and
            # single-process `batch_render_rgb` records to the camera counter too.
            metrics_method="record_batch_camera_render_time",
        )

    def batch_render_camera(self, request, context):
        """Current batch camera render. Same fan-out/per-item contract as `batch_render_rgb`, over
        `CameraRenderRequest` items (`BatchCameraRenderReturnItem.result` is a `CameraRenderReturn`)."""
        self._require_allowed_batch_image_formats(request, context, single_item_rpc="render_camera")
        return self._batch_render(
            request,
            context,
            worker_rpc="render_camera",
            return_cls=sensorsim_pb2.BatchCameraRenderReturn,
            item_cls=sensorsim_pb2.BatchCameraRenderReturnItem,
            metrics_method="record_batch_camera_render_time",
        )

    def _require_allowed_batch_image_formats(self, request, context, *, single_item_rpc: str) -> None:
        for index, item in enumerate(request.items):
            require_allowed_image_format(
                item.request,
                self._allowed_image_formats,
                context,
                request_description=f"Batch item[{index}] {item.camera_name!r}",
            )
            # use_cuda_ipc now exists on both RGBRenderRequest and CameraRenderRequest (the
            # getattr default below is defensive, not load-bearing). Neither batch RPC's IPC
            # delivery is supported under multi-process serving: single-process batch_render_rgb
            # now does support it (atomic multi-buffer reservation against one session -- see
            # serve.py), but the proxy's round-robin worker fan-out has no way to reserve
            # buffers across workers, and open_ipc_session is refused here entirely (see the
            # class-level note above) so no ipc_session_id could ever be valid regardless. Abort
            # the same typed UNIMPLEMENTED the dedicated IPC RPCs and the singular
            # render_rgb/render_camera give, rather than forwarding to an arbitrary worker and
            # letting it fail with an opaque error.
            if getattr(item.request, "use_cuda_ipc", False):
                context.abort(
                    grpc.StatusCode.UNIMPLEMENTED,
                    f"{self._CUDA_IPC_MULTIPROC_UNSUPPORTED} (item[{index}] {item.camera_name!r}); "
                    f"use {single_item_rpc} for CUDA IPC delivery of a single camera at a time.",
                )
                raise  # unreachable, but needed for mypy

    def _batch_render(self, request, context, *, worker_rpc, return_cls, item_cls, metrics_method):
        """Fan the batch's items across workers: each item is a singular render (`worker_rpc`) to a round-robin
        worker, **admitted per item** (so a batch can't exceed `max_in_flight` concurrent renders), with
        `serve`'s **per-item** result contract — preserve order + `camera_name`, and map a worker/RPC
        failure or an admission shed to that item's `success=false`/`error_message` (never a batch failure).
        A whole-batch failure is reserved for a malformed request (mismatched scene ids), matching `serve`.

        Shared by the deprecated `batch_render_rgb` and the current `batch_render_camera`; the two proto
        item/return shapes are structurally identical (`camera_name` + `result` + `success`/`error_message`),
        so only the worker RPC, the message classes and the metrics counter differ.
        """
        items = list(request.items)
        if not items:
            return return_cls(items=[])

        # Parity with serve: a batch must target a single scene_id.
        scene_id = items[0].request.scene_id
        mismatched = [
            (i, it.camera_name, it.request.scene_id) for i, it in enumerate(items) if it.request.scene_id != scene_id
        ]
        if mismatched:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "All cameras in batch must use the same scene_id; mismatches: "
                + ", ".join(f"item[{i}] {name!r}={sid!r}" for i, name, sid in mismatched),
            )

        def _render_item(index: int, item) -> "tuple[int, Any]":
            # Same `_admit()` primitive as the singular path, but an admission shed becomes THIS item's
            # `success=false` (never an outer-batch abort). `_admit()` only releases after a successful
            # acquire, so a shed item leaks no permit.
            try:
                with self._admit(context):
                    # `_invoke` (not `_forward`): a downstream RpcError must become THIS item's
                    # success=false below, never a whole-batch abort via context.abort.
                    rendered = self._invoke(getattr(self._next_stub(), worker_rpc), item.request, context)
                    return index, item_cls(camera_name=item.camera_name, result=rendered, success=True)
            except _AdmissionRejected as exc:
                return index, item_cls(
                    camera_name=item.camera_name, success=False, error_message=f"RESOURCE_EXHAUSTED: {exc.details}"
                )
            except grpc.RpcError as exc:
                code, details = self._extract_worker_rpc_error(exc)  # shared decoder → identical to singular relay
                return index, item_cls(
                    camera_name=item.camera_name, success=False, error_message=f"{code.name}: {details}"
                )
            except Exception as exc:  # never let one item fail the whole batch
                return index, item_cls(camera_name=item.camera_name, success=False, error_message=f"INTERNAL: {exc}")

        # One lease for the whole batch's scene (barrier + eviction lease) held across all
        # items, so the scene cannot be evicted mid-batch.
        with self._scene_lease(scene_id, context):
            results: list = [None] * len(items)
            # Per-batch fan-out pool size. These threads are blocking gRPC waiters (each blocks in `_invoke`'s
            # `call.result()` on a worker render, or on the `_admit()` admission semaphore) — not CPU/GPU
            # consumers, so the sizing targets keeping every worker busy, not core count. `N*4` gives pipelining
            # headroom (some items rendering while others queue/forward) with an `8` floor for the 1-2 worker
            # case; `min(len(items), …)` never spawns more threads than items. The total thread fan-out is
            # BOUNDED, not offered-load-driven: at most `max(8, N*2)` batch handlers run concurrently (the proxy
            # gRPC executor cap in `serve_proxy`), so worst-case batch-pool threads ≤ max(8, N*2) × max(8, N*4)
            # (e.g. 64 at N=1, 128 at N=4, 512 at N=8) — excess concurrent batch RPCs queue at the gRPC layer
            # rather than spawning more pools. Actual downstream render concurrency is capped independently by
            # `--multiproc-max-in-flight` admission and by each worker serializing renders (worker `max_workers=1`);
            # the extra waiters just block, so a larger multiplier would not over-drive the GPUs.
            max_threads = min(len(items), max(8, len(self._stubs) * 4))
            batch_start = time.perf_counter()
            with futures.ThreadPoolExecutor(max_workers=max_threads) as pool:
                pending = [pool.submit(_render_item, i, item) for i, item in enumerate(items)]
                for future in futures.as_completed(pending):
                    index, item_result = future.result()
                    results[index] = item_result
            # Record the batch handler duration + per-camera under the same names as serve (the proxy
            # is where the batch RPC lives; the workers' per-item render_time_ms is recorded separately).
            if self._metrics_manager is not None:
                getattr(self._metrics_manager, metrics_method)(
                    (time.perf_counter() - batch_start) * 1000.0, len(items), scene_id=scene_id
                )
            return return_cls(items=results)

    # Discovery / metadata: workers are equivalent, so answer from worker 0. The forward still carries the
    # client's deadline/cancellation (via `_forward`) so a wedged worker can't pin a proxy handler thread.
    def get_version(self, request, context):
        return self._forward(self._stubs[0].get_version, request, context)

    def get_available_scenes(self, request, context):
        """Report every scene the owner can serve (primary + glob + on-demand-registered), not just one
        worker's currently-attached set.

        **Parity note (intentional difference from single-process `serve`):** `serve` lists only scenes
        already loaded/registered in its `artifacts`, whereas the pool reports
        `coordinator.known_scenes()` = every registered + attached scene, including ones not yet attached
        (and, for a bad on-demand URL, possibly not loadable until the first render's attach barrier). So a
        drop-in client that enumerates scenes then renders one may hit a late NOT_FOUND/UNAVAILABLE at render
        time rather than at enumeration. Documented as a known drop-in behavioral difference; not aligned to
        "attached + loadable" because that would hide registered-but-not-yet-attached scenes the pool serves
        lazily by id."""
        if self._scene_coordinator is not None:
            return common_pb2.AvailableScenesReturn(scene_ids=sorted(self._scene_coordinator.known_scenes()))
        return self._forward(self._stubs[0].get_available_scenes, request, context)

    def get_loaded_scenes(self, request, context):
        """Report the pool's resident scenes from the owner's authoritative fleet view (never one
        worker's local cache, which can disagree with the fleet mid-attach/evict).

        Response semantics, adapted from single-process `serve` (which reports its `BackendCache`):
        a scene is *loaded* when it is attached pool-wide (the all-worker attach barrier), and every
        attached scene is then resident on every worker — so `loaded_instance_count` is the worker
        count. The pool has no spare/checkout instances (each worker's instance is bound to that
        worker), so `reusable_instance_count` is always 0. `loaded_instance_capacity` is the fleet
        instance cap: the coordinator's resident-scene budget × workers, with the coordinator's
        "0 = unlimited" reported as 0 (single-process `BackendCache.maxsize` has no unlimited form)."""
        if self._scene_coordinator is not None:
            num_workers = len(self._stubs)
            max_scenes = self._scene_coordinator.max_scenes
            return sensorsim_pb2.LoadedScenesReturn(
                loaded_instance_capacity=max_scenes * num_workers if max_scenes > 0 else 0,
                scenes=[
                    sensorsim_pb2.LoadedSceneEntry(
                        scene_id=scene_id,
                        loaded_instance_count=num_workers,
                        reusable_instance_count=0,
                    )
                    for scene_id in sorted(self._scene_coordinator.attached_scenes())
                ],
            )
        # Coordinator-less (single-scene) pool: every worker holds exactly the bootstrap scene, so
        # worker 0's local view is the fleet view (same fallback as get_available_scenes).
        return self._forward(self._stubs[0].get_loaded_scenes, request, context)

    def get_available_cameras(self, request, context):
        with self._scene_lease(self._lease_scene_id(request), context):  # attach + hold across the forward
            return self._forward(self._stubs[0].get_available_cameras, request, context)

    def get_available_trajectories(self, request, context):
        with self._scene_lease(self._lease_scene_id(request), context):
            return self._forward(self._stubs[0].get_available_trajectories, request, context)

    def get_dynamic_objects(self, request, context):
        with self._scene_lease(self._lease_scene_id(request), context):
            return self._forward(self._stubs[0].get_dynamic_objects, request, context)

    def get_external_asset_objects(self, request, context):
        with self._scene_lease(self._lease_scene_id(request), context):
            return self._forward(self._stubs[0].get_external_asset_objects, request, context)

    def get_server_config(self, request, context):
        # The owner process knows the effective launch config + runtime status; answer from it
        # when available.
        if self._server_config_map is not None:
            return sensorsim_pb2.ServerConfig(server_config=self._server_config_map)
        # Fallback (the standalone `multiproc-serve` CLI builds no server_config_map): forward worker 0's
        # config but overlay the pool's true topology. Worker 0's SensorSimService(server_config=None)
        # unconditionally reports serving.mode=single_process, so without this overlay a drop-in client
        # reading the discovery key would see single_process for a running pool — a present-but-wrong key
        # that defeats the purpose of exposing serving.mode at all.
        forwarded = self._forward(self._stubs[0].get_server_config, request, context)
        server_config = dict(forwarded.server_config)
        server_config["serving.mode"] = "multiprocess"
        server_config["multiproc.workers"] = str(len(self._stubs))
        return sensorsim_pb2.ServerConfig(server_config=server_config)

    def get_available_ego_masks(self, request, context):
        return self._forward(self._stubs[0].get_available_ego_masks, request, context)

    def shut_down(self, request, context):
        """Graceful fleet shutdown: signal the owner main loop to stop the proxy + tear down the workers.

        Returns immediately (the client gets its response); the owner loop, watching `shutdown_event`,
        then stops the proxy server and calls `WorkerPool.shutdown()` (worker join), so there are no
        orphans. When no event is wired (unit tests) this is a no-op ack. Recorded as a REQUESTED stop
        (exit 0) — distinct from a degradation/worker-death nonzero exit.
        """
        _signal_shutdown(self._shutdown_event, PoolTerminalReason.REQUESTED)
        return common_pb2.Empty()

    def edit_assets(self, request, context):
        """Editing is not supported with multi-process serving.

        Kept as an explicit override (rather than falling through to the base servicer, which would return a
        typed UNIMPLEMENTED status) so a client that only branches on ``response.success`` still sees a clean
        in-band failure: transport ``OK`` with ``success=False``. Use single-process `serve-grpc` for in-place
        edits.
        """
        return EditAssetsResponse(
            success=False,
            message=(
                "UNIMPLEMENTED: editing (edit_assets) is not supported with multi-process serving "
                "(--multiproc-workers > 0); use single-process serve-grpc for in-place edits"
            ),
        )

    def restore_model_parameters(self, request, context):
        """Editing is not supported with multi-process serving: no in-place edits can exist to restore."""
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "restore_model_parameters is not supported with multi-process serving "
            "(--multiproc-workers > 0): no in-place edits can exist in this mode",
        )

    def create_harmonizer_session(self, request, context):
        """Temporal-harmonizer sessions are per-process state the proxy cannot route across the worker pool,
        and `validate_multiproc_harmonizer` already refuses `--enable-harmonizer-temporal` at launch, so a
        session could never be valid. Abort FAILED_PRECONDITION with an actionable message — mirroring
        single-process `serve`'s `create_harmonizer_session` precondition abort — rather than falling through
        to the generated base servicer's generic UNIMPLEMENTED "Method not implemented!" (the proto comment
        promises FAILED_PRECONDITION here)."""
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "harmonizer sessions are not supported with multi-process serving (--multiproc-workers > 0); "
            "sessions are per-process state the proxy cannot route across the worker pool",
        )

    def end_harmonizer_session(self, request, context):
        """No harmonizer session can exist under multi-process serving (see `create_harmonizer_session`), so
        there is nothing to end. Abort FAILED_PRECONDITION with the same actionable message rather than the
        base servicer's generic UNIMPLEMENTED."""
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "harmonizer sessions are not supported with multi-process serving (--multiproc-workers > 0); "
            "sessions are per-process state the proxy cannot route across the worker pool",
        )

    # CUDA IPC delivery pins a client's session to the buffer pool a single worker
    # process allocated and exported cudaIpcMemHandle_t's for, but the proxy
    # load-balances renders across the pool and cannot guarantee a `use_cuda_ipc`
    # render reaches that same worker. Both IPC RPCs are therefore refused (typed
    # UNIMPLEMENTED, not a fall-through to the base servicer's generic one) — use
    # single-process serve-grpc for CUDA IPC.
    _CUDA_IPC_MULTIPROC_UNSUPPORTED = (
        "CUDA IPC transport is not supported with multi-process serving (--multiproc-workers > 0); the "
        "session-to-worker buffer affinity cannot be routed across the pool — use single-process serve-grpc"
    )

    def open_ipc_session(self, request, context):
        """CUDA IPC delivery is not supported with multi-process serving (see the class-level note)."""
        context.abort(grpc.StatusCode.UNIMPLEMENTED, self._CUDA_IPC_MULTIPROC_UNSUPPORTED)

    def close_ipc_session(self, request, context):
        """No IPC session can exist under multi-process serving, so there is nothing to close."""
        context.abort(grpc.StatusCode.UNIMPLEMENTED, self._CUDA_IPC_MULTIPROC_UNSUPPORTED)


@dataclass(frozen=True)
class PostProcessConfig:
    """Render post-processing knobs threaded to each worker so the pool matches `serve`: egocar
    hood masks (`insert_ego_mask`) and the Harmonizer/Difix output filter. A frozen dataclass of plain
    types so it is picklable across the `spawn` boundary."""

    egocar_hoods_dir: "str | None" = None
    enable_harmonizer: bool = False
    harmonizer_url: str = ""
    harmonizer_cache: str = ""
    harmonizer_model_filename: str = ""
    harmonizer_resolution: tuple[int, int] = (576, 1024)

    @classmethod
    def from_grpc_config(cls, config: GrpcServerConfig) -> "PostProcessConfig":
        """Mirror `serve`'s post-processing wiring (`serve.py` service ctor + `set_harmonizer`) from the config."""
        return cls(
            egocar_hoods_dir=config.egocar_hood_dir,
            enable_harmonizer=config.enable_harmonizer,
            harmonizer_url=config.harmonizer_url,
            harmonizer_cache=config.harmonizer_cache,
            harmonizer_model_filename=config.harmonizer_model_filename,
            harmonizer_resolution=cast("tuple[int, int]", tuple(config.harmonizer_resolution)),
        )


@dataclass(frozen=True)
class WorkerSpawnConfig:
    """The static per-worker spawn knobs, bundled so they thread through the bring-up chain (`run_drop_in` ->
    `start_worker_pool` -> `_spawn_worker` -> `ctx.Process` -> `_worker`) as ONE picklable object instead of
    ~7 individual positional `_worker` params (which forced shotgun surgery per knob and a misalignment-prone
    16-slot positional tuple).

    A frozen dataclass of plain types + a `PostProcessConfig` (itself frozen/plain), so it is picklable across
    the `spawn` boundary — the child unpickles it at module scope (no lambdas/locks/closures; mirrors
    `PostProcessConfig`). These fields are identical across all workers EXCEPT `metrics_output_dir`, which
    `_spawn_worker` specializes per worker (via `dataclasses.replace`) into that worker's metrics subdirectory."""

    renderer_name: str
    enable_editing_actors: bool
    metrics_output_dir: "str | None" = None
    ray_chunk_size: int = _RAY_CHUNK_SIZE
    scene_cache_size: int = 1
    post_process: "PostProcessConfig | None" = None
    tracks_timestamp_ray_stride: int = 1


class _WorkerSupervisionUnavailableError(RuntimeError):
    """Raised in a worker when OS owner-liveness supervision (the parent-process sentinel) is unavailable.

    A spawned worker always has a multiprocessing parent, so this is fatal: the worker refuses to attach,
    load, bind, or serve without the authoritative owner-death signal rather than degrading to
    lifecycle-channel-only supervision."""


def _worker(
    worker_id: int,
    scene_id: "str | None",
    artifact_path: "str | None",
    config: WorkerSpawnConfig,
    attach_req_q,
    attach_resp_q,
    port_q,
    lifecycle_rx,
    scene_q=None,
    scene_ack_q=None,
) -> None:
    faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` dumps worker stacks
    service = None  # bound below once built; the finally flushes its metrics on every exit path
    exit_code = 0  # nonzero on a fault the owner must observe as WORKER_EXIT (control-plane failure / crash)
    try:
        # Establish OS owner-liveness supervision FIRST — before any attach, model load, port bind, or
        # readiness. A spawned worker's multiprocessing parent IS the owner, and its death sentinel is the
        # authoritative liveness signal. If it is unavailable, refuse to start (fail closed) rather than
        # silently degrading to lifecycle-channel-only supervision.
        parent_sentinel, is_parent_alive = _owner_parent_signals()
        if parent_sentinel is None:
            raise _WorkerSupervisionUnavailableError(
                f"worker {worker_id}: no multiprocessing parent-process death sentinel; refusing to start "
                "without OS owner-liveness supervision"
            )

        # The bootstrap scene (scene_id) is OPTIONAL. When present, request its attach from the owner
        # and build the IPC-aliased worker backend BEFORE binding the port. When absent (a zero-scene
        # startup), skip the attach handshake + backend build entirely: the worker comes up scene-less and
        # the runtime scene table — fed by the coordinator on the first x-nre-scene-url registration — drives
        # resolution. Supervision above is still established first either way.
        backend = None
        artifact = None
        if scene_id is not None:
            assert artifact_path is not None  # the owner always spawns a worker with both a scene + artifact
            attach_req_q.put(AttachSceneRequest(scene_id))
            response: OwnerResponse = attach_resp_q.get(timeout=_WORKER_ATTACH_RESP_TIMEOUT_S)
            if not response.ok or response.attachment is None:
                port_q.put(("error", worker_id, response.error or "attach failed"))
                return
            ipc_tensors = dict(response.attachment.tensors)
            artifact = Artifact(Path(artifact_path))
            backend = _build_and_release_worker_backend(
                artifact,
                config.renderer_name,
                ipc_tensors,
                config.tracks_timestamp_ray_stride,
                scene_id=scene_id,
            )

        # One render at a time per worker: requests serialize at the gRPC layer so a preloaded (aliased)
        # backend is reused serially. The cache holds the primary scene plus any runtime-attached scenes
        # , so a render resolves its scene by id without a cache-miss re-load.
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=_CHANNEL_OPTIONS)
        service = SensorSimService(
            server,
            artifacts_glob=None,
            ray_chunk_size=config.ray_chunk_size,  # honor --ray-chunk-size (OOM avoidance)
            # Egocar hood masks (insert_ego_mask) loaded into every worker, like serve.
            egocar_hoods_dir=config.post_process.egocar_hoods_dir if config.post_process is not None else None,
            cache_size=max(1, config.scene_cache_size),
            metrics_output_dir=config.metrics_output_dir,  # per-worker dir; records render_time_ms
            enable_editing_actors=config.enable_editing_actors,
            server_config=None,
        )
        # Harmonizer output filter, wired exactly as serve does (no-op when disabled). Only the
        # stateless (non-temporal) harmonizer: --enable-harmonizer-temporal is refused at multiproc
        # launch by validate_multiproc_harmonizer, so the temporal kwargs keep their disabled defaults.
        if config.post_process is not None:
            service.set_harmonizer(
                config.post_process.enable_harmonizer,
                config.post_process.harmonizer_url,
                config.post_process.harmonizer_cache,
                config.post_process.harmonizer_model_filename,
                config.post_process.harmonizer_resolution,
            )
        scene_table = _WorkerSceneTable(service) if (scene_q is not None and scene_ack_q is not None) else None

        # Register the bootstrap scene (if any) so `render_rgb` resolves it by scene_id. zero-pinned: register
        # it **through the scene table** — exactly like a runtime-attached scene — so a coordinator detach
        # removes it cleanly (no `artifacts` handle, so a cache-miss can't rebuild a non-aliased copy of shared
        # weights). The single-scene fallback (no scene channel) keeps the historical artifacts/cache path. A
        # zero-scene worker registers nothing here; its scene table stays empty until the first attach.
        if scene_id is not None:
            assert backend is not None  # built above whenever a bootstrap scene is present
            if scene_table is not None:
                scene_table[scene_id] = backend
                # The scene table (service BackendCache) is now the ONLY intended worker-side retainer of
                # the bootstrap. `_worker`'s frame lives for the whole process (it blocks in
                # `_serve_until_stop` below), so keeping these locals would pin the bootstrap's aliased
                # owner-IPC mappings and worker-local allocations across a coordinator eviction — the
                # owner's proven-detach release would then never actually reclaim the VRAM (silent
                # over-admission on a GPU sized to max_scenes). The single-scene fallback branch below
                # intentionally KEEPS `backend`: the service BackendCache (plus this frame's local) is the
                # designed retainer and that topology has no coordinator eviction.
                backend = None
                artifact = None
                del ipc_tensors, response
            else:
                # Single-scene fallback (no scene channel — only reachable from direct `_worker` unit
                # calls; `_spawn_worker` always passes scene queues). Publish only the already-aliased
                # bootstrap backend into the cache; deliberately DO NOT register the artifact for cache-miss
                # rebuild. A rebuild here would use RendererType.DEFAULT + stride 1 and a private, non-aliased
                # weight copy that silently diverges from the owner's IPC-shared, gsplat backend, so a
                # cache-miss must fail loud (NOT_FOUND) rather than resolve a wrong-renderer scene.
                service.register_resident_backend(backend)
                service.backend_cache.put(backend.backend_cache_key.key_hash, backend)
                service.backend_cache.checkin(backend)

        # Runtime scene attach/detach: the owner pushes additional scenes over scene_q; the
        # listener builds + aliases each and registers it in this service's BackendCache so render_rgb
        # resolves it by scene_id. Daemon; exits on the owner's stop sentinel.
        if scene_table is not None:
            scene_thread = threading.Thread(
                target=_scene_attach_listener,
                args=(
                    scene_table,
                    _scene_backend_builder(config.renderer_name, config.tracks_timestamp_ray_stride),
                    scene_q,
                    scene_ack_q,
                ),
                daemon=True,
            )
            scene_thread.start()

        add_SensorsimServiceServicer_to_server(service, server)
        # A failed port bind (add_insecure_port -> 0) must fail worker startup loudly BEFORE start()/ready:
        # publishing "ready" for an unbound server would hand the owner a dead port. Mirrors the proxy/health
        # bind guards above. The raise is caught by this function's except → an ("error", ...) is published on
        # port_q instead of ("ready", ...), so the owner observes a startup fault rather than a phantom port.
        port = server.add_insecure_port("127.0.0.1:0")
        if port == 0:
            raise RuntimeError("failed to bind worker gRPC server on 127.0.0.1:0 (port unavailable)")
        server.start()
        port_q.put(("ready", worker_id, port))

        # Block here serving requests until the owner is authoritatively lost (its parent-process sentinel
        # becomes ready) or it sends an explicit STOP over the lifecycle channel. Request traffic is not part
        # of this decision; there is no heartbeat or timeout-based lease. The sentinel/liveness probe derived
        # at startup are passed through so supervision cannot silently be lost between startup and serving.
        reason = _serve_until_stop(
            server, lifecycle_rx, parent_sentinel=parent_sentinel, is_parent_alive=is_parent_alive
        )
        # A control-plane failure is a fault: exit nonzero so the owner observes WORKER_EXIT. An owner-directed
        # stop and stopping because the owner is gone are clean exits (0).
        exit_code = reason.exit_code
        print(f"worker {worker_id}: {reason.value}; gRPC server stopped, exiting {exit_code}", flush=True)
    except _WorkerSupervisionUnavailableError as exc:
        # Clean, expected startup refusal (no traceback): the owner surfaces this as a startup failure.
        port_q.put(("error", worker_id, str(exc)))
        exit_code = 1
    except BaseException:
        port_q.put(("error", worker_id, traceback.format_exc()))
        exit_code = 1
    finally:
        # The worker owns the lifecycle receiver. Close it deterministically on every startup/serve exit path
        # rather than relying on process teardown/GC to release the descriptor.
        with contextlib.suppress(Exception):
            lifecycle_rx.close()
        # Flush this worker's metrics.yaml on every exit path — including an abnormal teardown, which the
        # merge step explicitly tolerates (merge_metrics_yaml skips a missing source) — so a worker that
        # dies mid-serve does not silently lose all its samples.
        if service is not None:
            with contextlib.suppress(Exception):
                service._metrics_manager.shutdown()
    # Exit nonzero on a fault so the owner's `_await_pool_shutdown` maps this worker's death to WORKER_EXIT
    # (a nonzero fleet exit). A clean/owner-directed stop returns normally (exit 0).
    if exit_code:
        sys.exit(exit_code)


def _proc_provably_dead(proc: Any) -> bool:
    """True only if ``proc.is_alive()`` cleanly reports False.

    Used to skip a graceful lifecycle STOP for a worker already known dead (partial-startup rollback, or a
    repeated shutdown after a prior reap). Unknown liveness (a raising ``is_alive``) returns False so the
    caller still best-effort sends STOP rather than silently skipping a possibly-live worker."""
    try:
        return not proc.is_alive()
    except Exception:  # noqa: BLE001 - unknown liveness: treat as not-provably-dead
        return False


def _terminate_and_reap(proc: Any, *, join_timeout_s: float = 5.0) -> None:
    """Escalate SIGTERM -> SIGKILL to guarantee a worker is reaped within a bound.

    ``terminate()`` (SIGTERM) can be ignored/blocked by a worker wedged in a CUDA kernel or gRPC I/O; left
    unreaped it keeps its CUDA context, IPC tensor mappings, and port binding alive (so the next pool start
    fails to bind and GPU memory leaks). So after SIGTERM we ``join(timeout)`` and, if the process is still
    alive, send SIGKILL and join again. No-op if the process already exited."""
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=join_timeout_s)
    if proc.is_alive():
        proc.kill()  # SIGKILL: uninterruptible, guarantees the OS reaps the worker
        proc.join(timeout=join_timeout_s)


@dataclass
class WorkerPool:
    """A running owner + N-worker pool: the public worker addresses and the owner's loaded model. Holds
    the worker processes, the per-worker lifecycle channels, and the IPC registry alive; call `shutdown()`
    to tear it down (it stops the scene listeners, sends each still-alive worker a lifecycle STOP, then
    bounded join/terminate/reap). `main()` wraps this with the proxy server; tests/tools (e.g. the smoke)
    drive it directly."""

    scene_id: "str | None"  # the eager bootstrap scene, or None for a zero-scene startup
    worker_addresses: list[str]
    owner_backend: Any  # serve.Backend on the owner: model + asset bank (None in zero-pinned mode)
    coordinator: "SceneCoordinator | None" = None  # multi-scene gate; None on a single-scene pool
    scene_channels: list = field(default_factory=list)  # SceneAttachChannel per worker (runtime scene attach)
    _procs: list = field(default_factory=list)
    _registry: Any = None  # ModelOwnerRegistry: kept alive so the owner's IPC tensors outlive the workers
    _owner_scene_registry: Any = None  # OwnerSceneRegistry: keeps every loaded scene's owner object alive
    # Owner write ends of each worker's one-way lifecycle channel, index-aligned with `_procs` (None for a
    # never-spawned slot). `shutdown()` sends a lifecycle STOP on each still-alive worker's channel so a stop
    # is an explicit owner-directed exit, not an inferred owner death; a closed channel is set back to None so
    # a repeated shutdown neither resends nor re-closes it.
    _lifecycle_txs: list = field(default_factory=list)
    # Cleanup handles (e.g. the proxy's worker channels) whose close was DEFERRED because the proxy stop could
    # not be confirmed: closing them before the workers are reaped could cut a still-live handler's
    # forward. `shutdown()` runs them only after every worker is proven dead. Never run on a surviving worker.
    _deferred_closers: list = field(default_factory=list)

    @property
    def owner_renderable(self) -> Any:
        # None in zero-pinned mode (no persistent owner Backend).
        return self.owner_backend.renderable_model if self.owner_backend is not None else None

    def all_workers_alive(self) -> bool:
        return all(proc.is_alive() for proc in self._procs)

    def defer_close(self, closer: "Callable[[], Any]") -> None:
        """Hand a cleanup callable (e.g. `proxy.close`) to fleet teardown so its strong reference survives until
        after the workers are reaped. Used when the proxy stop could not be confirmed: its worker
        channels must not be closed out from under a possibly-live handler."""
        self._deferred_closers.append(closer)

    def shutdown(self) -> None:
        pre_reap_failures: list[tuple[str, Exception]] = []
        # Stop the scene-attach listeners FIRST, while the workers are still alive to drain their stop
        # sentinels, and only THEN send the lifecycle STOP that ends each worker's serve loop. If STOP went
        # first the worker could exit and tear down its listener threads before their sentinels were
        # enqueued, leaving the owner-side stop enqueuing onto a dead reader.
        for index, channel in enumerate(self.scene_channels):  # stop each worker's scene-attach listener
            try:
                channel.stop()
            except Exception as exc:  # one broken queue must not skip stopping/reaping the rest of the fleet
                pre_reap_failures.append((f"scene channel {index} stop", exc))
        # Worker-stop phase: direct each still-alive worker to stop via its lifecycle channel BEFORE the
        # bounded join below, so a stop is an explicit owner-directed exit rather than an inferred owner death.
        # A worker already proven dead (e.g. reaped during a partial-startup rollback, or on a repeated
        # shutdown after a prior reap) needs no STOP — sending one would only fault on its closed read end. A
        # failed send to a live/unknown worker is recorded (not silently clean); the bounded terminate/reap
        # escalation below remains the fallback for any worker that does not exit on the request.
        for index, lifecycle_tx in enumerate(self._lifecycle_txs):
            if lifecycle_tx is None:  # never-spawned slot, or already closed by an earlier shutdown pass
                continue
            proc = self._procs[index] if index < len(self._procs) else None
            if proc is not None and _proc_provably_dead(proc):
                continue  # no graceful stop needed; reap already handles it
            try:
                send_worker_stop(lifecycle_tx)
            except Exception as exc:  # keep stopping/reaping independent resources; reap is the fallback
                pre_reap_failures.append((f"worker {index} lifecycle stop send", exc))
        for index, proc in enumerate(self._procs):
            try:
                proc.join(timeout=10.0)  # give the worker a chance to exit after the stop request
            except Exception as exc:
                pre_reap_failures.append((f"worker {index} graceful join", exc))
            try:
                _terminate_and_reap(proc)  # SIGTERM -> SIGKILL escalation; continue even if another proc failed
            except Exception as exc:
                pre_reap_failures.append((f"worker {index} terminate/reap", exc))
        unconfirmed_workers: list[int] = []
        for index, proc in enumerate(self._procs):
            try:
                alive = proc.is_alive()
            except Exception as exc:
                # A failed liveness probe is not proof of death. Conservatively retain every shared owner
                # reference and deferred closer, just as for a process that explicitly reports alive.
                pre_reap_failures.append((f"worker {index} final liveness probe", exc))
                unconfirmed_workers.append(index)
            else:
                if alive:
                    unconfirmed_workers.append(index)
        if unconfirmed_workers:
            # Retain every owner-side scene reference (and every deferred channel closer): freeing one
            # while a surviving worker may still map its CUDA storage — or closing a channel it still forwards
            # over — would turn a shutdown failure into a use-after-free / crash.
            raise RuntimeError(
                f"failed to prove worker(s) {unconfirmed_workers} dead; pre-reap failures={pre_reap_failures}; "
                "retaining scene owner references for safety"
            )
        # Every worker is provably dead. Close lifecycle transmitters first, then release owner references and
        # queue endpoints, then close deferred proxy channels. All three phases aggregate failures and continue
        # independently so a broken close never skips another post-reap resource.
        cleanup_failures: list[tuple[str, Exception]] = []
        # A successfully-closed endpoint is dropped from the list (set to None) so a repeated shutdown neither
        # resends a STOP through it nor re-closes it. An endpoint whose close FAILS is retained in place so the
        # caller can diagnose/retry it after this pass finishes the other post-reap cleanup.
        for index, lifecycle_tx in enumerate(self._lifecycle_txs):
            if lifecycle_tx is None:
                continue
            try:
                lifecycle_tx.close()
            except Exception as exc:
                cleanup_failures.append((f"worker {index} lifecycle channel close", exc))
            else:
                self._lifecycle_txs[index] = None
        for index, channel in enumerate(self.scene_channels):
            try:
                _released, failures = channel.release_after_reap()  # drop retained scene refcounts
            except Exception as exc:  # an unexpected channel fault must not skip the remaining cleanup
                cleanup_failures.append((f"scene channel {index}", exc))
            else:
                cleanup_failures.extend((f"scene channel {index} {resource}", exc) for resource, exc in failures)
        leaked_scenes: list[str] = []
        if self._owner_scene_registry is not None:
            try:
                self._owner_scene_registry.release_all_idle()  # free every now-idle owner scene load
            except Exception as exc:
                cleanup_failures.append(("owner scene release_all_idle", exc))
            # After every worker is dead and every channel refcount dropped, no owner scene should remain
            # loaded. A still-loaded scene means a refcount leak — surface it loudly instead of hiding it.
            try:
                leaked_scenes = list(self._owner_scene_registry.loaded_scenes())
            except Exception as exc:
                cleanup_failures.append(("owner scene loaded_scenes verification", exc))
        # Close deferred proxy channels only now (post-reap): no live handler can be forwarding over them.
        failed_closers: list = []
        for index, closer in enumerate(self._deferred_closers):
            try:
                closer()
            except Exception as exc:
                # Keep the strong reference so a caller can diagnose/retry a failed resource close. A failed
                # closer must be reported, but it must not prevent later independent closers from running.
                failed_closers.append(closer)
                cleanup_failures.append((f"deferred closer {index}", exc))
        self._deferred_closers = failed_closers
        if pre_reap_failures or cleanup_failures or leaked_scenes:
            # Fail loudly on any post-reap cleanup/accounting fault; retained records aid diagnosis/retry.
            prefix = "shutdown incomplete after worker reap" if pre_reap_failures else "post-reap cleanup incomplete"
            raise RuntimeError(
                f"{prefix}: pre-reap failures={pre_reap_failures}, post-reap failures={cleanup_failures}, "
                f"still-loaded owner scenes={leaked_scenes}"
            )


class WorkerPoolStartupCleanupError(RuntimeError):
    """Pool startup failed and its partial fleet could not be cleanly torn down.

    ``retained_pool`` deliberately keeps the partial fleet state available for diagnosis/retry; when worker death
    could not be proven, this is also what keeps every owner registry, scene attachment, and process handle alive.
    Entry points treat this as fatal/nonzero; an embedding caller may inspect the original
    ``startup_error``/``cleanup_error`` and retry ``retained_pool.shutdown()``.
    """

    def __init__(
        self,
        startup_error: BaseException,
        cleanup_error: Exception,
        retained_pool: WorkerPool,
    ) -> None:
        super().__init__(
            f"worker-pool startup failed ({startup_error!r}) and partial-fleet cleanup was incomplete "
            f"({cleanup_error!r}); retaining partial-pool state for diagnosis/retry"
        )
        self.startup_error = startup_error
        self.cleanup_error = cleanup_error
        self.retained_pool = retained_pool


class _BringupCancelled(Exception):
    """Raised inside a bring-up task when another worker's failure cancels the rest of pool startup.

    Distinct from a genuine bring-up failure: the orchestrator only ever surfaces the *first* real error,
    and cancelled siblings raise this (cheap, no spawn) so they unwind fast instead of blocking the
    `ThreadPoolExecutor` teardown for the full ready timeout."""


@dataclass
class _WorkerHandle:
    """The owner-side artifacts of one spawned worker: its process, the three handshake queues, its
    scene-attach channel, and the owner's write end of the one-way lifecycle channel (the neutral STOP).
    Returned by `_spawn_worker` and consumed by `_await_worker_ready`.

    RETENTION BARRIER: this handle is the ONLY owner-side reference keeping the three handshake queues'
    named semaphores alive, and the attach/ready handshake is what proves the spawned child finished
    unpickling them (any worker message ⇒ the whole spawn payload was rebuilt). Dropping the handle —
    or awaiting readiness on a detached task — before `_await_worker_ready` returns lets the queue
    finalizers `sem_unlink` names the still-importing child has yet to `sem_open`, killing it with
    FileNotFoundError at unpickle. The invariant is: never create a multiprocessing queue the owner will
    not retain until the child has provably finished importing."""

    worker_id: int
    proc: Any
    attach_req_q: Any
    attach_resp_q: Any
    port_q: Any
    channel: "SceneAttachChannel"
    lifecycle_tx: Any


def _poll_worker_queue(
    q: Any,
    proc: Any,
    cancel_event: "threading.Event",
    timeout_s: float,
    poll_interval_s: float,
    *,
    what: str,
) -> Any:
    """Block for the next message on ``q``, polling so the wait stays responsive during bring-up.

    Unlike a single ``q.get(timeout=...)``, this wakes every ``poll_interval_s`` to (a) bail with
    `_BringupCancelled` if a sibling's failure set ``cancel_event``, and (b) fail fast if ``proc`` has
    died without sending anything (a hard crash that skips the worker's error-ack path). Raises
    `TimeoutError` if nothing arrives within ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while True:
        if cancel_event.is_set():
            raise _BringupCancelled(f"{what}: cancelled by another worker's failure")
        try:
            return q.get(timeout=poll_interval_s)
        except queue.Empty:
            pass
        # Got nothing this interval: a dead process with an empty queue will never send -> fail now
        # rather than wait out the full timeout (the worker's own error ack covers graceful failures).
        if proc is not None and not proc.is_alive() and q.empty():
            raise RuntimeError(f"{what}: worker process exited during startup")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{what}: timed out after {timeout_s:.0f}s")


def _spawn_worker(
    worker_id: int,
    *,
    ctx: Any,
    scene_id: "str | None",
    artifact_path: "Path | None",
    spawn_config: WorkerSpawnConfig,
    registry: "ModelOwnerRegistry",
    on_started: "Callable[[Any, Any], None] | None" = None,
) -> _WorkerHandle:
    """Create one worker's queues + lifecycle channel, start its process, and register its owner-side
    scene-attach channel. Returns a `_WorkerHandle` carrying the owner's write end of the one-way
    lifecycle channel (used to send the lifecycle STOP at shutdown); the caller drives the attach/ready
    handshake via `_await_worker_ready`. ``on_started`` is invoked with ``(proc, lifecycle_tx)`` the instant
    the process starts — before any wiring that can fail — so the caller records the process AND its
    lifecycle endpoint together (aligned ownership), never one without the other. Safe to call concurrently
    from the bring-up thread pool."""
    attach_req_q, attach_resp_q, port_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    scene_q, scene_ack_q = ctx.Queue(), ctx.Queue()
    # One-way owner->worker lifecycle channel: the owner keeps `lifecycle_tx` to send the neutral lifecycle
    # STOP at shutdown; the worker waits on `lifecycle_rx` alongside the owner's parent-death sentinel.
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)
    # Specialize the shared spawn config for THIS worker: its metrics land in a per-worker subdirectory
    # (None when metrics are off), while every other knob is identical fleet-wide. `replace` keeps the frozen
    # config picklable across the spawn boundary.
    worker_metrics_dir = (
        _worker_metrics_dir(spawn_config.metrics_output_dir, worker_id) if spawn_config.metrics_output_dir else None
    )
    worker_config = replace(spawn_config, metrics_output_dir=worker_metrics_dir)
    proc = ctx.Process(
        target=_worker,
        args=(
            worker_id,
            scene_id,
            # A zero-scene worker has no bootstrap artifact; pass None (not the string "None") so the
            # worker skips the eager Artifact/backend build and comes up scene-less.
            str(artifact_path) if artifact_path is not None else None,
            worker_config,  # static per-worker spawn knobs (renderer, editing, ray-chunk, cache, post-process, ...)
            attach_req_q,
            attach_resp_q,
            port_q,
            lifecycle_rx,
            scene_q,
            scene_ack_q,
        ),
    )
    try:
        proc.start()
        # Publish the process handle AND its lifecycle write end together, immediately after start and before
        # any wiring that can fail, so the caller's aligned pool lists never retain a process without its
        # lifecycle endpoint. The global rollback can then send STOP / verify reap even if this function's
        # local best-effort cleanup also raises.
        if on_started is not None:
            on_started(proc, lifecycle_tx)
        # The worker inherited its own copy of the read end; close the owner's copy so the worker's channel
        # sees an EOF if (and only if) the owner closes its write end.
        lifecycle_rx.close()
        # Pass this worker's liveness probe so a detach ack failure only force-drops the owner refcount
        # when the worker is actually dead (never on a slow ack from a live worker → no use-after-free).
        channel = SceneAttachChannel(scene_q, scene_ack_q, registry, is_worker_alive=proc.is_alive)
    except BaseException:
        # Close the owner's receiver on start/record/wiring failure as well. If the process did start, closing
        # this duplicate preserves the one-reader/one-writer channel contract while local rollback reaps it.
        with contextlib.suppress(Exception):
            lifecycle_rx.close()
        try:
            if getattr(proc, "pid", None) is not None:
                _terminate_and_reap(proc)
        finally:
            # A start failure has no partial pool to own this transmitter; after a successful start the caller
            # recorded it before wiring, but closing here is still safe once local reap has completed/failed
            # closed and prevents descriptor leakage during the global rollback.
            with contextlib.suppress(Exception):
                lifecycle_tx.close()
        raise
    return _WorkerHandle(
        worker_id=worker_id,
        proc=proc,
        attach_req_q=attach_req_q,
        attach_resp_q=attach_resp_q,
        port_q=port_q,
        channel=channel,
        lifecycle_tx=lifecycle_tx,
    )


def _await_worker_ready(
    handle: _WorkerHandle,
    registry: "ModelOwnerRegistry",
    scene_id: "str | None",
    cancel_event: "threading.Event",
    progress: Callable[[str], None],
    *,
    req_timeout_s: float = _WORKER_ATTACH_REQ_TIMEOUT_S,
    ready_timeout_s: float = _WORKER_READY_TIMEOUT_S,
    poll_interval_s: float = _BRINGUP_POLL_INTERVAL_S,
) -> str:
    """Service a spawned worker's bootstrap attach (if any) and wait for it to bind its gRPC port; return its
    address. Raises `_BringupCancelled` if a sibling failed, `RuntimeError` if the worker reports/dies on
    failure, or `TimeoutError` if it never responds. The bootstrap is zero-pinned, so its startup
    attachment is handed to the scene-attach channel so the coordinator can later detach + release it like
    any runtime scene.

    A ``scene_id`` of ``None`` is a zero-scene startup: the worker requests no bootstrap attach, so
    skip servicing one entirely and wait only for the worker's ``("ready", ...)`` port message."""
    if scene_id is not None:
        request = _poll_worker_queue(
            handle.attach_req_q,
            handle.proc,
            cancel_event,
            req_timeout_s,
            poll_interval_s,
            what=f"worker {handle.worker_id} attach request",
        )
        attach_response = handle_owner_request(registry, request)
        if attach_response.ok and attach_response.attachment is not None:
            # Record owner attachment ownership BEFORE queue delivery. A synchronous response-queue fault must
            # not lose the only attachment id needed to release the bootstrap after the partial worker is reaped.
            handle.channel.seed_attachment(scene_id, attach_response.attachment.attachment_id)
        handle.attach_resp_q.put(attach_response)
    message = _poll_worker_queue(
        handle.port_q,
        handle.proc,
        cancel_event,
        ready_timeout_s,
        poll_interval_s,
        what=f"worker {handle.worker_id} ready",
    )
    if message[0] != "ready":
        raise RuntimeError(f"worker {message[1]} failed to start:\n{message[2]}")
    address = f"127.0.0.1:{message[2]}"
    progress(f"  worker {handle.worker_id} serving at {address}")
    return address


def _bring_up_pool(
    num_workers: int,
    max_concurrent_bringup: int,
    bring_up_one: "Callable[[int, threading.Event], Any]",
) -> None:
    """Run ``bring_up_one(worker_id, cancel_event)`` for every worker with bounded concurrency.

    At most ``max_concurrent_bringup`` workers come up at once (``<= 0`` ⇒ all in parallel; ``1`` ⇒ the
    historical serialized startup), which bounds the transient per-worker model load that peaks before
    `alias_external_weights` rebinds onto the owner's IPC tensors. ``bring_up_one`` spawns + awaits one
    worker (registering its proc/channel/lifecycle write end as side effects the caller tears down). On the
    **first** failure the shared cancel event is set so in-flight bring-ups bail fast, then that error is
    re-raised for the caller's global rollback."""
    cancel_event = threading.Event()
    concurrency = num_workers if max_concurrent_bringup <= 0 else min(max_concurrent_bringup, num_workers)
    with futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        pending = {pool.submit(bring_up_one, worker_id, cancel_event): worker_id for worker_id in range(num_workers)}
        try:
            for future in futures.as_completed(pending):
                future.result()  # surface (and stop on) the first real bring-up failure
        except BaseException:
            cancel_event.set()  # tell still-running / not-yet-started bring-ups to unwind fast
            raise


def start_worker_pool(
    artifact_path: "Path | None" = None,
    *,
    num_workers: int,
    renderer_name: str,
    enable_editing_actors: bool,
    extra_scene_artifacts: "dict[str, str] | None" = None,
    worker_scene_cache_size: "int | None" = None,
    max_scenes: int = 0,
    post_process: "PostProcessConfig | None" = None,
    metrics_output_dir: "str | None" = None,
    ray_chunk_size: int = _RAY_CHUNK_SIZE,
    max_concurrent_bringup: int = 1,
    tracks_timestamp_ray_stride: int = 1,
    progress: Callable[[str], None] = lambda _msg: None,
) -> WorkerPool:
    """Load the owner model, spawn `num_workers` worker gRPC servers attached to its IPC weights, and wire
    each worker's one-way lifecycle channel. Returns a `WorkerPool` (caller runs the proxy and must call
    `shutdown()`). Raises if any worker fails to come up.

    `max_concurrent_bringup` caps how many workers come up at once: `1` (default) is the historical
    serialized startup (one worker fully attaches + binds its port before the next spawns); a higher
    value overlaps that work; `<= 0` brings every worker up in parallel. Each worker transiently holds a
    full duplicate of the model weights on the GPU while `Backend.from_artifact` runs (before
    `alias_external_weights` rebinds onto the owner's zero-copy IPC tensors), so the cap bounds the
    startup VRAM peak (~owner + min(N, cap) loads) — raise it for faster startup on GPUs with headroom,
    keep it low on tight ones.

    `artifact_path` is the eager **bootstrap** scene, pre-attached on every worker at startup. It is
    **optional**: ``None`` is a valid zero-startup-scene launch — no eager owner load and no startup
    attach, the workers come up scene-less, and the first scene arrives on demand (via the
    `SceneCoordinator` / `x-nre-scene-url`), mirroring single-process `serve` starting with an empty
    ``artifact_glob``. `extra_scene_artifacts` (scene_id -> artifact path) are additional **owner-loadable**
    scenes the proxy can attach on demand via the `SceneCoordinator`. `worker_scene_cache_size` budgets how
    many scenes each worker's BackendCache holds resident (default = the pre-known set: bootstrap + extras,
    or just extras with no bootstrap); raise it for on-demand download
    so runtime-attached scenes don't LRU-evict each other. `max_scenes` caps the resident scene
    count fleet-wide and enables coordinated LRU eviction; 0 = unlimited; the worker cache is
    sized to at least `max_scenes` so the coordinator (not the worker LRU) is what evicts.

    A fully path-registered glob fleet is **zero-pinned** (every scene evictable, serve-like)."""
    _require_supported_renderer(renderer_name)  # Gsplat-only pool; fail fast before spawning workers
    extra_scene_artifacts = dict(extra_scene_artifacts or {})
    owner_scene_registry = OwnerSceneRegistry()
    registry = owner_scene_registry.registry
    # Path-driven pinning: register the bootstrap's path (re-loadable ⇒ evictable) and pin nothing.
    # (A scene with no registered path stays resident.)
    coordinator_scene_artifacts = dict(extra_scene_artifacts)
    pinned_scene_ids: tuple[str, ...] = ()
    # The bootstrap scene (glob's first match) is OPTIONAL. When present, the owner loads a full
    # Backend (not just a RenderableModel); only
    # its model weights are exported via IPC (workers alias those). The OwnerSceneRegistry wraps the IPC
    # registry with per-scene first-load serialization for the extras. When absent (an empty glob), skip the
    # eager load entirely: the pool comes up scene-less and the first scene loads/attaches on demand.
    bootstrap_scene_id: "str | None" = None
    if artifact_path is not None:
        artifact = Artifact(artifact_path)
        bootstrap_scene_id = artifact.scene_id
        owner_backend = Backend.from_artifact(
            artifact,
            _renderer_type(renderer_name),
            tracks_timestamp_ray_stride,
            backend_cache_key=_pool_backend_cache_key(bootstrap_scene_id, artifact),
        )
        owner_scene_registry.ensure_loaded(bootstrap_scene_id, lambda: owner_backend)  # register the bootstrap
        coordinator_scene_artifacts.setdefault(bootstrap_scene_id, str(artifact_path))
    # The worker BackendCache must hold the bootstrap + every scene it may attach without LRU churn (an
    # evicted runtime scene is unrecoverable: it is not in the worker's `artifacts`, so a cache-miss can't
    # reload it). Pre-known scenes set the floor; `worker_scene_cache_size`/`max_scenes` raise it so the
    # coordinator's bounded eviction — not the worker's own LRU — is what drops scenes.
    #
    # Size it strictly ABOVE the coordinator's resident cap (`max_scenes + 1`) when eviction is on: a
    # coordinated attach does `_WorkerSceneTable.__setitem__` = `BackendCache.put` then `checkin`, and `put`
    # runs the worker's count-LRU while the new backend is still in-use — so the worker momentarily holds
    # `max_scenes + 1` backends. Without the +1 headroom the worker would evict one of its own spares (a
    # scene the coordinator still treats as attached, and which a cache-miss cannot rebuild), making the
    # worker a second, uncoordinated eviction authority. The +1 keeps the coordinator the sole authority over
    # coordinator-managed scenes. (With eviction off — `max_scenes <= 0`, unbounded admission — no static
    # size can hold the invariant; `_WorkerSceneTable.__setitem__` grows the cache capacity in step with the
    # attach stream instead.)
    worker_eviction_headroom = max_scenes + 1 if max_scenes > 0 else 0
    # Pre-known scenes set the cache floor: the bootstrap (1, when present) + the extras. Drop the bootstrap
    # reservation for a zero-scene start — there is no eager scene to hold resident.
    bootstrap_reservation = 1 if bootstrap_scene_id is not None else 0
    scene_cache_size = max(
        bootstrap_reservation + len(extra_scene_artifacts), worker_scene_cache_size or 1, worker_eviction_headroom
    )
    if bootstrap_scene_id is not None:
        progress(
            f"owner: loaded scene {bootstrap_scene_id!r}; spawning {num_workers} worker(s)"
            + (f" (+{len(extra_scene_artifacts)} on-demand scene(s))" if extra_scene_artifacts else "")
        )
    else:
        progress(
            f"owner: no bootstrap scene (zero-scene startup); spawning {num_workers} worker(s)"
            + (f" ({len(extra_scene_artifacts)} on-demand scene(s) registered)" if extra_scene_artifacts else "")
        )

    # Bundle the static per-worker spawn knobs ONCE (identical fleet-wide; `_spawn_worker` specializes only the
    # per-worker metrics dir). `scene_cache_size` is derived from the resolved scene set just above, so this is
    # the natural single construction point for a complete, always-valid config threaded down to every worker.
    spawn_config = WorkerSpawnConfig(
        renderer_name=renderer_name,
        enable_editing_actors=enable_editing_actors,
        metrics_output_dir=metrics_output_dir,
        ray_chunk_size=ray_chunk_size,
        scene_cache_size=scene_cache_size,
        post_process=post_process,
        tracks_timestamp_ray_stride=tracks_timestamp_ray_stride,
    )

    ctx = mp.get_context("spawn")
    # Pre-sized + index-addressed so concurrent bring-up tasks write their own slot without a lock and the
    # results stay ordered by worker_id (deterministic round-robin) regardless of completion order. Entries
    # stay None for workers that never spawned (e.g. a cancelled sibling); the rollback path skips those.
    procs: list = [None] * num_workers
    scene_channels: list = [None] * num_workers  # SceneAttachChannel per worker (runtime attach/detach)
    worker_addresses: list = [None] * num_workers
    # Owner write end of each worker's one-way lifecycle channel (the neutral STOP at shutdown). Index-addressed
    # like `procs` (and written together with it by `_record_started`), so a concurrent bring-up task writes
    # its own slot without a lock; a never-spawned slot stays None and the shutdown path skips it.
    lifecycle_txs: list = [None] * num_workers

    def _bring_up_one(worker_id: int, cancel_event: "threading.Event") -> None:
        # A sibling already failed (cancel set): bail before spawning, so a cancelled pool doesn't leak a
        # process that the rollback would then have to terminate.
        if cancel_event.is_set():
            raise _BringupCancelled(f"worker {worker_id}: cancelled before spawn")

        def _record_started(proc: Any, lifecycle_tx: Any) -> None:
            # Record the process AND its lifecycle endpoint together (same worker slot), the instant the
            # process starts, so a partial-startup failure never retains a process without its endpoint.
            procs[worker_id] = proc
            lifecycle_txs[worker_id] = lifecycle_tx

        handle = _spawn_worker(
            worker_id,
            ctx=ctx,
            scene_id=bootstrap_scene_id,
            artifact_path=artifact_path,
            spawn_config=spawn_config,
            registry=registry,
            on_started=_record_started,
        )
        # proc + lifecycle endpoint were recorded together by `_record_started` before any failure point; the
        # scene-attach channel only exists once `_spawn_worker` returns, so record it here for teardown before
        # the handshake below can fail.
        scene_channels[worker_id] = handle.channel
        worker_addresses[worker_id] = _await_worker_ready(handle, registry, bootstrap_scene_id, cancel_event, progress)

    try:
        _bring_up_pool(num_workers, max_concurrent_bringup, _bring_up_one)
    except BaseException as startup_error:
        # Route partial startup through the SAME reap-before-release ownership gate as a running fleet. This
        # attempts every worker, verifies final liveness, releases bootstrap attachments/queues only after all
        # workers are dead, and retains the partial pool inside a typed fatal error if cleanup is incomplete.
        # Keep `_procs` and `_lifecycle_txs` index-aligned: filter them TOGETHER on the process slot (both are
        # set atomically by `_record_started`, so a spawned slot has both), so shutdown's per-worker liveness
        # guard pairs each lifecycle endpoint with the right process.
        spawned = [(proc, tx) for proc, tx in zip(procs, lifecycle_txs) if proc is not None]
        partial_pool = WorkerPool(
            scene_id=bootstrap_scene_id,
            worker_addresses=[],
            owner_backend=None,
            scene_channels=[channel for channel in scene_channels if channel is not None],
            _procs=[proc for proc, _tx in spawned],
            _registry=registry,
            _owner_scene_registry=owner_scene_registry,
            _lifecycle_txs=[tx for _proc, tx in spawned],
        )
        try:
            partial_pool.shutdown()
        except Exception as cleanup_error:
            raise WorkerPoolStartupCleanupError(startup_error, cleanup_error, partial_pool) from startup_error
        raise

    # The coordinator first-loads + attaches an extra scene on ALL workers before the proxy routes to it;
    # the bootstrap (when present) is seeded as already attached (it was attached at worker startup above) —
    # a zero-scene start seeds an empty attached set and the first scene attaches on demand. Every glob
    # scene, including the bootstrap, has a reloadable path and `pinned_scene_ids` is empty, so no scene is
    # held resident.
    def _load_owner_backend(sid: str, path: str) -> "Backend":
        # Keyed by the coordinator's routing `sid` (not `artifact.scene_id`): an extra scene may be
        # registered under a caller-chosen id, and that id is what the proxy and workers resolve by.
        scene_artifact = Artifact(Path(path))
        return Backend.from_artifact(
            scene_artifact,
            _renderer_type(renderer_name),
            tracks_timestamp_ray_stride,
            backend_cache_key=_pool_backend_cache_key(sid, scene_artifact),
        )

    coordinator = SceneCoordinator(
        owner_scene_registry,
        scene_channels,
        coordinator_scene_artifacts,
        _load_owner_backend,
        attached_scenes=(bootstrap_scene_id,) if bootstrap_scene_id is not None else (),
        max_scenes=max_scenes,  # Coordinated LRU eviction at this resident-scene cap (0 = off)
        pinned=pinned_scene_ids,
    )

    return WorkerPool(
        scene_id=bootstrap_scene_id,
        worker_addresses=worker_addresses,
        # No persistent owner-Backend pin — the OwnerSceneRegistry holds each scene's owner object
        # (loaded on demand, released on eviction), so the fleet is zero-pinned.
        owner_backend=None,
        coordinator=coordinator,
        scene_channels=scene_channels,
        _procs=procs,
        _registry=registry,
        _owner_scene_registry=owner_scene_registry,
        _lifecycle_txs=lifecycle_txs,
    )


class PoolTerminalReason(enum.Enum):
    """Why the owner main loop stopped, and whether that is a clean stop.

    A client `shut_down` RPC follows the requested health-transition/drain path and exits 0; a **worker death**, a
    **scene-pool degradation** (a detach/attach that could not be driven to a
    proven-consistent state — see `SceneCoordinator`), or a **control-plane teardown failure** (the
    proxy could not be confirmed stopped) exits nonzero so a supervisor (systemd / k8s) restarts the
    pod rather than leaving a half-torn-down or over-admitting fleet in rotation. The reason
    is typed (not a bare event) so degradation is never mistaken for a normal shutdown. The launch wrappers map
    `KeyboardInterrupt` to exit 0 for compatibility, but that exception bypasses this normal proxy-drain path."""

    REQUESTED = "shut_down requested"
    WORKER_EXIT = "a worker exited"
    SCENE_POOL_DEGRADED = "scene residency pool degraded"
    CONTROL_PLANE_FAILURE = "control-plane teardown could not be confirmed"

    @property
    def is_clean(self) -> bool:
        """True only for an operator/client-requested stop (the sole exit-0 reason)."""
        return self is PoolTerminalReason.REQUESTED

    @property
    def exit_code(self) -> int:
        """Process exit code: 0 for a requested stop, nonzero for worker death / pool degradation."""
        return 0 if self.is_clean else 1


class PoolShutdown:
    """Shared owner-loop shutdown signal that also records **why** the fleet is stopping.

    Behaves like a `threading.Event` for the wait loop (`set`/`is_set`/`wait`), but `set` optionally carries a
    typed `PoolTerminalReason` (first reason wins). The client `shut_down` RPC sets it with the default
    ``REQUESTED``; the scene-pool degraded callback sets it with ``SCENE_POOL_DEGRADED`` **before** waking the
    loop, so `_await_pool_shutdown` can distinguish a clean stop (exit 0) from a degradation/worker death
    (nonzero exit) instead of treating any set event as a normal shutdown.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: "PoolTerminalReason | None" = None
        self._lock = threading.Lock()

    def set(self, reason: PoolTerminalReason = PoolTerminalReason.REQUESTED) -> None:
        # Record the reason (first one wins) BEFORE waking the loop, so the loop never observes a set event
        # without the reason that set it.
        with self._lock:
            if self._reason is None:
                self._reason = reason
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: "float | None" = None) -> bool:
        return self._event.wait(timeout)

    @property
    def reason(self) -> "PoolTerminalReason | None":
        with self._lock:
            return self._reason


def _signal_shutdown(shutdown_event: Any, reason: PoolTerminalReason) -> None:
    """Signal the owner loop to stop, recording ``reason`` when the signal supports it.

    Branches on type so a `PoolShutdown` records the typed terminal reason (mapped to a nonzero exit for a
    degradation) while a bare `threading.Event` (used by unit tests / the render-path fail-closed) just sets.
    A `PoolShutdown.set` ignores ``reason`` after the first, so a later requested stop cannot overwrite an
    earlier ``SCENE_POOL_DEGRADED``. Using an explicit `isinstance` rather than a bare `except TypeError`
    means an unexpected `TypeError` raised *inside* a genuine `PoolShutdown.set` surfaces instead of being
    silently swallowed (which would drop the typed degradation reason with no diagnostic)."""
    if shutdown_event is None:
        return
    if isinstance(shutdown_event, PoolShutdown):
        shutdown_event.set(reason)  # PoolShutdown records the reason
    else:
        shutdown_event.set()  # bare threading.Event takes no argument


def _await_pool_shutdown(pool: WorkerPool, shutdown_event: Any, *, poll_timeout_s: float = 0.5) -> PoolTerminalReason:
    """Block until a worker exits or something sets ``shutdown_event``; return the typed loss reason.

    Whole-fleet-shutdown contract: a multi-process pool tears the **whole fleet** down when **any** worker
    exits — there is no per-worker respawn or removal-from-rotation, so one worker OOM/segfault takes the
    server down rather than degrading to N-1. Extracted from the serve loop so the teardown decision is
    unit-testable without standing up real servers.

    The reason is typed: a set event carries a `PoolTerminalReason` (``SCENE_POOL_DEGRADED`` from the
    scene-pool degraded callback, else the client-requested default ``REQUESTED``); a loop that exits because
    a worker died (event never set) is ``WORKER_EXIT``. Only ``REQUESTED`` exits 0; every other reason exits
    nonzero so a supervisor restarts the fleet. See the API guide's multiprocess deployment/shutdown section
    for operator guidance.
    """
    while pool.all_workers_alive() and not shutdown_event.wait(timeout=poll_timeout_s):
        pass
    if not shutdown_event.is_set():
        return PoolTerminalReason.WORKER_EXIT  # loop exited because a worker died, not on a set signal
    # Set event: a bare threading.Event (tests) carries no reason -> a requested stop; PoolShutdown carries
    # SCENE_POOL_DEGRADED when the degraded callback set it.
    return getattr(shutdown_event, "reason", None) or PoolTerminalReason.REQUESTED


def _set_proxy_health_status(health_servicer: Any, status: int) -> None:
    """Set the aggregate and concrete Sensorsim health names to the same lifecycle status."""
    for service_name in _PROXY_HEALTH_SERVICE_NAMES:
        health_servicer.set(service_name, status)


def _stop_server_bounded(server: Any, *, grace: float | None, timeout_s: float, label: str) -> None:
    """Request a server stop and require confirmation within ``timeout_s``.

    gRPC's ``Server.stop`` only requests a stop; returning from it does not prove handlers are gone. Keeping
    this check in one helper prevents startup or normal teardown from accidentally treating a
    timeout/exception as a confirmed stop and closing worker channels out from under a possibly-live handler.
    """
    try:
        stopped = bool(server.stop(grace).wait(timeout=timeout_s))
    except Exception as exc:
        raise RuntimeError(f"{label} stop raised {exc!r}") from exc
    if not stopped:
        raise RuntimeError(f"{label} stop was not confirmed within {timeout_s:.1f}s")


def _close_proxy_after_reap(proxy_server: Any, proxy: "RenderProxy", *, label: str) -> None:
    """Retry an uncertain proxy stop after workers are dead, then close its worker channels.

    Channel closure is safe post-reap even if the retry still cannot confirm the proxy stop. Report every
    failure so ``WorkerPool.shutdown`` retains this closer for diagnosis/retry instead of silently discarding it.
    """
    failures: list[Exception] = []
    try:
        _stop_server_bounded(proxy_server, grace=None, timeout_s=5.0, label=label)
    except Exception as exc:
        failures.append(exc)
    try:
        proxy.close()
    except Exception as exc:
        failures.append(exc)
    if failures:
        raise RuntimeError(f"{label} cleanup incomplete: {failures}")


def _stop_health_after_reap(health_server: Any, *, label: str) -> None:
    """Retry an unconfirmed health-server stop during post-reap cleanup."""
    _stop_server_bounded(
        health_server,
        grace=None,
        timeout_s=5.0,
        label=label,
    )


def _drain_and_close_proxy(
    proxy_server: Any,
    proxy: "RenderProxy",
    health_server: Any,
    reason: PoolTerminalReason,
    *,
    progress: Callable[[str], None] = lambda _msg: None,
    defer_close: "Callable[[Callable[[], None]], None] | None" = None,
    grace_s: float = 2.0,
    drain_wait_s: float = 5.0,
    forced_wait_s: float = 5.0,
) -> "tuple[PoolTerminalReason, bool]":
    """Stop the proxy deterministically, then close the worker channels — never the other way round.

    ``Server.stop(grace)`` is non-blocking and returns an Event set once the drain finishes; we **check** that
    result instead of assuming it. If the graceful drain times out or its stop/wait raises, we request an
    immediate stop and wait again with a bound. Worker gRPC channels are closed only once the proxy server
    confirms its handlers stopped, so an in-flight render is never cut off mid-forward.

    Returns ``(reason, proxy_closed)``. If the proxy stop cannot be confirmed, ``proxy_closed`` is False and the
    caller MUST keep the proxy's worker channels alive until the fleet is reaped (then close them) — this
    function does **not** close them, and does **not** return early: the dedicated health server is still stopped
    and checked. ``defer_close``, when provided, transfers both an uncertain proxy's server+channel ownership
    and an uncertain dedicated health server to post-reap cleanup; otherwise the caller owns those retries. An
    unconfirmed stop never reports a clean requested shutdown: a non-clean root cause is preserved, else it is
    escalated to ``CONTROL_PLANE_FAILURE`` (nonzero exit for a supervisor restart).
    """
    try:
        _stop_server_bounded(
            proxy_server,
            grace=grace_s,
            timeout_s=drain_wait_s,
            label="proxy graceful drain",
        )
    except Exception as graceful_exc:
        progress(f"proxy graceful drain was not confirmed ({graceful_exc!r}); forcing an immediate stop")
        try:
            _stop_server_bounded(
                proxy_server,
                grace=None,
                timeout_s=forced_wait_s,
                label="proxy forced stop",
            )
        except Exception as forced_exc:
            progress(
                "proxy forced stop could not be confirmed; retaining worker channels for fleet teardown "
                f"({forced_exc!r})"
            )
            proxy_confirmed = False
            if reason.is_clean:  # preserve a non-clean root cause; escalate a REQUESTED stop that cannot drain
                reason = PoolTerminalReason.CONTROL_PLANE_FAILURE
            if defer_close is not None:
                # Retain both the server and proxy until every worker is reaped. The closer retries the server
                # stop before releasing channels, and WorkerPool retains it if that post-reap cleanup fails.
                defer_close(
                    functools.partial(
                        _close_proxy_after_reap,
                        proxy_server,
                        proxy,
                        label="draining proxy server",
                    )
                )
        else:
            proxy_confirmed = True
    else:
        proxy_confirmed = True
    proxy_closed = False
    if proxy_confirmed:
        proxy.close()  # confirmed stopped -> safe to close the worker channels (fds + bg gRPC threads)
        proxy_closed = True
    # Always stop + check the dedicated health server, even when the proxy stop could not be confirmed (it has
    # no worker channels, so stopping it is safe and must not be skipped by an early return).
    if health_server is not None:
        try:
            _stop_server_bounded(
                health_server,
                grace=1.0,
                timeout_s=2.0,
                label="dedicated health server",
            )
        except Exception as exc:
            progress(f"dedicated health server stop could not be confirmed ({exc!r})")
            if reason.is_clean:
                reason = PoolTerminalReason.CONTROL_PLANE_FAILURE
            if defer_close is not None:
                # Retain the server object itself until the worker fleet is reaped, and make a bounded retry.
                # WorkerPool.shutdown reports and retains this closer if the retry is still unconfirmed.
                defer_close(
                    functools.partial(
                        _stop_health_after_reap,
                        health_server,
                        label="dedicated health server",
                    )
                )
    return reason, proxy_closed


def serve_proxy(
    pool: WorkerPool,
    *,
    host: str,
    port: int,
    shutdown_event: Any,
    health_port: int | None = None,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
    admission_timeout_s: float = _DEFAULT_ADMISSION_TIMEOUT_S,
    server_config_map: "dict[str, str] | None" = None,
    metrics_output_dir: "str | None" = None,
    scene_cache: "SceneCache | None" = None,
    downloader: "Downloader | None" = None,
    progress: Callable[[str], None] = lambda _msg: None,
) -> PoolTerminalReason:
    """Run the front proxy over ``pool`` until a worker dies or `shut_down` sets ``shutdown_event``.

    Stands up the `RenderProxy` (admission wired) plus a gRPC health servicer (same port, or a
    dedicated ``health_port``), serves until the loop condition trips, then stops the proxy/health servers
    and returns the loss reason. The caller owns `pool.shutdown()`. Shared by `main()` and the
    conformance smokes. When ``metrics_output_dir`` is set, a `MetricsInterceptor` records the proxy-side
    e2e/decode/encode per RPC to ``<metrics_output_dir>/proxy``. On a multi-scene pool, a
    `SceneDownloadInterceptor` handles client ``x-nre-scene-url`` headers; ``scene_cache`` +
    ``downloader`` enable remote downloads (local ``file://`` URIs work without them).
    """
    # Proxy-side metrics (e2e incl. the double-hop) — separate from the workers' render_time_ms.
    proxy_metrics: "ServerMetricsManager | None" = None
    interceptors: list = []
    # Request handling stays free of any worker-liveness work: no heartbeat/liveness interceptor is wired
    # here. Worker lifetime is governed only by the parent-process sentinel and the explicit lifecycle
    # channel, so request traffic is never part of the worker lifetime contract.
    if metrics_output_dir:
        proxy_metrics = ServerMetricsManager(os.path.join(metrics_output_dir, "proxy"))
        metrics_interceptor = MetricsInterceptor()
        metrics_interceptor.set_metrics_manager(proxy_metrics)
        interceptors.append(metrics_interceptor)

    # On-demand scene download/registration via x-nre-scene-url (mirror serve's interceptor). The
    # registrar resolves the scene to an owner-loadable artifact path + registers it with the coordinator;
    # the render's _scene_lease (SceneCoordinator.acquire) then drives the first-load + all-worker attach barrier.
    if pool.coordinator is not None:
        registrar = OnDemandSceneRegistrar(pool.coordinator, scene_cache=scene_cache, downloader=downloader)
        interceptors.append(SceneDownloadInterceptor(registrar))

    proxy_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max(8, len(pool.worker_addresses) * 2)),
        options=_CHANNEL_OPTIONS,
        interceptors=interceptors,
    )
    proxy = RenderProxy.from_addresses(
        pool.worker_addresses,
        max_in_flight=max_in_flight,
        admission_timeout_s=admission_timeout_s,
        shutdown_event=shutdown_event,
        server_config_map=server_config_map,
        metrics_manager=proxy_metrics,
        scene_coordinator=pool.coordinator,
    )
    # Fail-closed on scene detach/attach uncertainty: if the scene control plane degrades, tear the fleet
    # down for a supervisor restart rather than serving an over-admitting fleet. Record the typed degradation
    # reason (not a bare/requested stop) so the launch path exits NONZERO and the supervisor restarts the
    # pod. First reason wins.
    if pool.coordinator is not None:
        pool.coordinator.set_degraded_callback(
            lambda _reason: _signal_shutdown(shutdown_event, PoolTerminalReason.SCENE_POOL_DEGRADED)
        )
    add_SensorsimServiceServicer_to_server(proxy, proxy_server)

    # gRPC health (parity with serve-grpc): same port by default, or a dedicated server.
    health_servicer = health.HealthServicer()
    health_server: "grpc.Server | None" = None
    # Bind + start under a guard: a failed port bind (add_insecure_port -> 0) or start must fail startup
    # NONZERO with NO partial resources left behind. Clean up the proxy channels, any started server,
    # and the metrics manager, then re-raise so the launch path exits nonzero.
    try:
        if proxy_server.add_insecure_port(f"{host}:{port}") == 0:
            raise RuntimeError(f"failed to bind proxy gRPC server on {host}:{port} (port in use or unavailable)")
        if health_port is None or health_port == port:
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, proxy_server)
        else:
            health_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, health_server)
            if health_server.add_insecure_port(f"{host}:{health_port}") == 0:
                raise RuntimeError(
                    f"failed to bind dedicated health gRPC server on {host}:{health_port} (port in use or unavailable)"
                )
            health_server.start()
        _set_proxy_health_status(health_servicer, health_pb2.HealthCheckResponse.SERVING)
        proxy_server.start()
    except BaseException:
        # Confirm (bounded) that any server that may have started has actually stopped BEFORE closing the proxy
        # channels. A bind/start failure does not prove the server has no live handler: if stop is unconfirmed,
        # retain both server and proxy through a deferred closer and release the channels only after fleet reap.
        try:
            _stop_server_bounded(
                proxy_server,
                grace=None,
                timeout_s=5.0,
                label="partial-startup proxy server",
            )
        except Exception as exc:
            log.error(f"partial-startup proxy stop unconfirmed; deferring channel close until post-reap: {exc!r}")
            pool.defer_close(
                functools.partial(
                    _close_proxy_after_reap,
                    proxy_server,
                    proxy,
                    label="partial-startup proxy server",
                )
            )
        else:
            try:
                proxy.close()  # confirmed stopped -> safe to close channels immediately
            except Exception as exc:
                log.error(f"partial-startup proxy close failed; deferring retry until post-reap: {exc!r}")
                pool.defer_close(proxy.close)
        if health_server is not None:
            try:
                _stop_server_bounded(
                    health_server,
                    grace=None,
                    timeout_s=5.0,
                    label="partial-startup dedicated health server",
                )
            except Exception as exc:
                log.error(f"partial-startup health stop unconfirmed; deferring retry until post-reap: {exc!r}")
                pool.defer_close(
                    functools.partial(
                        _stop_health_after_reap,
                        health_server,
                        label="partial-startup dedicated health server",
                    )
                )
        if proxy_metrics is not None:
            with contextlib.suppress(Exception):
                proxy_metrics.shutdown()  # flush metrics on every partial-startup failure
        raise

    health_note = "" if (health_port is None or health_port == port) else f" + health on {health_port}"
    progress(f"Serving on {host}:{port} (proxy over {len(pool.worker_addresses)} worker(s)){health_note}")

    # Keep the owner alive (workers map its weights); exit on a worker death or a shut_down RPC. The pool
    # tears down whole on any worker exit (see `_await_pool_shutdown`'s operational contract).
    reason = _await_pool_shutdown(pool, shutdown_event)
    progress(f"{reason.value}; shutting down")
    # Stop advertising every name registered above before the grace drain. Updating only the aggregate empty
    # name leaves clients that probe the concrete service routing to a degraded/draining fleet.
    _set_proxy_health_status(health_servicer, health_pb2.HealthCheckResponse.NOT_SERVING)
    # Deterministic drain: confirm the proxy stopped before closing worker channels; escalate to
    # a nonzero control-plane reason (retaining channels) if it cannot be confirmed.
    reason, _ = _drain_and_close_proxy(
        proxy_server,
        proxy,
        health_server,
        reason,
        progress=progress,
        defer_close=pool.defer_close,
    )
    if proxy_metrics is not None:
        proxy_metrics.shutdown()  # flush proxy metrics.yaml
    return reason


def _plan_glob_scenes(
    matches: "list[str]",
    *,
    scene_id_of: "Callable[[str], str] | None" = None,
) -> "tuple[Path | None, str | None, dict[str, str]]":
    """Resolve glob-matched artifact paths into ``(bootstrap_path, bootstrap_scene_id, extra_scene_artifacts)``
    for `start_worker_pool` (glob parity with `serve`).

    The first match is the eager **bootstrap** (workers attach it at startup); the rest are registered as
    owner-loadable on-demand scenes keyed by their **embedded** `scene_id` (so a client renders any of them
    by id with no header, lazily on first request — like `serve`). Raises on a duplicate `scene_id`
    (mirrors `serve`'s duplicate-scene guard). ``scene_id_of`` is injectable for testing.

    An empty glob is a valid **zero-startup-scene** state: returns ``(None, None, {})`` — no eager
    bootstrap — exactly like single-process `serve`, which binds and reports SERVING with an empty/absent
    ``artifact_glob`` and registers its first scene later via the ``x-nre-scene-url`` header. The pool then
    comes up with scene-less workers; the first on-demand registration loads/attaches the first scene.
    """
    if not matches:
        return None, None, {}
    read = scene_id_of or (lambda path: Artifact(Path(path)).scene_id)
    sid_paths = [(read(path), path) for path in matches]
    scene_to_path: dict[str, str] = {}
    for sid, path in sid_paths:
        if sid in scene_to_path:
            raise ValueError(
                f"duplicate scene_id {sid!r} in artifact glob (paths {scene_to_path[sid]!r} and {path!r}); "
                f"each scene needs a unique embedded scene_id"
            )
        scene_to_path[sid] = path
    bootstrap_scene_id, bootstrap_path = sid_paths[0]
    extra = {sid: path for sid, path in sid_paths if sid != bootstrap_scene_id}
    return Path(bootstrap_path), bootstrap_scene_id, extra


def _validate_startup_scenes(
    coordinator: "SceneCoordinator | None",
    scene_ids: "list[str]",
    *,
    progress: Callable[[str], None] = lambda _msg: None,
) -> int:
    """Drive every startup glob scene through the real owner/coordinator/all-worker attach barrier to honor
    `test_scenes_are_valid` in multiprocess mode. Returns the count validated.

    Runs after the worker pool exists but before the proxy becomes `SERVING`. Uses normal residency
    accounting, so a glob larger than ``max_scenes`` validates by evicting idle scenes rather than exceeding
    the cap (each `ensure_attached` may evict an earlier idle scene). Raises on the first scene that fails to
    load/attach; the caller tears the partial fleet down and exits startup nonzero, so no partially validated
    proxy serves. A no-op (returns 0) on a single-scene pool with no coordinator. The flag covers startup
    glob scenes only — a future `x-nre-scene-url` registration is validated on its first scene-using RPC."""
    if coordinator is None:
        return 0
    total = len(scene_ids)
    for index, scene_id in enumerate(scene_ids, start=1):
        coordinator.ensure_attached(scene_id)  # first-load + all-worker attach barrier; raises on failure
        progress(f"validated startup scene {scene_id!r} ({index}/{total})")
    return total


def _multiproc_server_config_map(
    grpc_config: GrpcServerConfig,
    *,
    num_workers: int,
    renderer_name: str,
    max_concurrent_bringup: int,
    max_in_flight: int,
    admission_timeout_s: float,
    startup_scenes: int,
    validated_startup_scenes: int,
    download_cache_size: int,
    tracks_timestamp_ray_stride: int = 1,
    mps_is_active: "bool | None" = None,
) -> "dict[str, str]":
    """Build the effective `get_server_config` map for the multiprocess drop-in.

    Unprefixed keys are the requested/common `GrpcServerConfig` inputs (kept in single-process
    serialization); the ``multiproc.*`` keys are the AUTHORITATIVE effective values and win where a requested
    value was transformed (e.g. ``renderer=default`` → effective gsplat, or a cache-capacity auto-bump). The
    single-process-only ``max_workers`` key is dropped — it does not describe the pool; ``multiproc.workers``
    is the worker count. ``serving.mode`` is present so a client never infers topology from absent keys.
    Booleans in the multiproc/serving namespaces serialize as lowercase ``"true"``/``"false"`` — one
    documented form (snapshotted exactly in tests)."""

    def _b(value: bool) -> str:
        return "true" if value else "false"

    config_map = dict(SensorSimService.build_server_config_map(grpc_config))
    config_map.pop("max_workers", None)  # single-process worker-thread count; not meaningful for the pool
    # Popped before GrpcServerConfig (like single-process serve), so surface it here for
    # get_server_config parity — same unprefixed key as the single-process map.
    config_map["tracks_timestamp_ray_stride"] = str(tracks_timestamp_ray_stride)
    mps_is_active = mps_active() if mps_is_active is None else mps_is_active
    config_map.update(
        {
            "serving.mode": "multiprocess",
            "multiproc.workers": str(num_workers),
            "multiproc.renderer": renderer_name,  # effective (default -> gsplat)
            "multiproc.mps_active": _b(mps_is_active),
            "multiproc.bringup_concurrency": str(max_concurrent_bringup),  # workers brought up in parallel
            "multiproc.max_scenes": str(grpc_config.cache_size),  # resident-scene cap (LRU eviction)
            "multiproc.startup_scenes": str(startup_scenes),  # a startup snapshot, NOT a dynamic total
            "multiproc.download_cache_size": str(download_cache_size),  # effective (>= max_scenes)
            "multiproc.max_in_flight": str(max_in_flight),  # 0 = unlimited active downstream renders
            "multiproc.admission_timeout_s": str(admission_timeout_s),  # 0 = shed immediately
            "multiproc.egocar_hoods": _b(bool(grpc_config.egocar_hood_dir)),  # ego masks loaded
            "multiproc.harmonizer": _b(grpc_config.enable_harmonizer),  # Harmonizer/Difix post-proc
            "multiproc.test_scenes_are_valid": _b(grpc_config.test_scenes_are_valid),
            "multiproc.validated_startup_scenes": str(validated_startup_scenes),
        }
    )
    return config_map


def run_drop_in(
    grpc_config: GrpcServerConfig,
    *,
    num_workers: int,
    require_mps: bool = False,
    require_scene_cache: bool = False,
    max_concurrent_bringup: int = 1,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
    admission_timeout_s: float = _DEFAULT_ADMISSION_TIMEOUT_S,
    tracks_timestamp_ray_stride: int = 1,
) -> int:
    """Run the multi-process pool from a `serve-grpc` `GrpcServerConfig` — the drop-in launch path.

    Accepts the same config surface as single-process `serve` so an operator switches by adding
    `--multiproc-workers N`. Honors host/port/health-port/renderer; runs the **MPS preflight**;
    and exposes the effective config + runtime status via `get_server_config`. A glob matching
    **N** artifacts serves every scene lazily by `scene_id` with no header; on-demand scene
    download via ``x-nre-scene-url`` is wired using ``download_cache_dir``/``download_cache_size``;
    resident scenes are capped at ``cache_size`` with coordinated LRU eviction, mirroring serve's
    BackendCache. The fleet is zero-pinned (every path-registered scene is evictable).
    """
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    # Validate the admission options up front from the shared contract, so an
    # invalid combination fails the launch before any worker is spawned. The Click layer catches the same
    # violations as a usage error; this guards the programmatic/drop-in callers.
    admission_error = validate_multiproc_admission(
        max_in_flight=max_in_flight,
        admission_timeout_s=admission_timeout_s,
    )
    if admission_error is not None:
        raise ValueError(admission_error)
    harmonizer_error = validate_multiproc_harmonizer(grpc_config)
    if harmonizer_error is not None:
        raise ValueError(harmonizer_error)
    # A glob may match N artifacts; the first is the eager bootstrap, the rest are registered as
    # owner-loadable scenes served lazily by scene_id (no header), like serve. An empty/absent glob is a
    # valid zero-startup-scene launch: bootstrap_path is None and the pool comes up scene-less.
    # Additional scenes (and the first one, in the zero-scene case) arrive via the x-nre-scene-url header.
    # Discover through the SAME seam as single-process serve (`Artifact.discover_from_glob`): recursive `**`
    # matching + the `.usdz` suffix assertion, so a `**` glob serving nested scenes with plain serve serves
    # the exact same set here (drop-in parity). `discover_from_glob` asserts the glob ends in `.usdz`, so a
    # non-`.usdz` glob now fails loud at startup rather than silently serving fewer/other files.
    matches = (
        sorted(str(artifact.source) for artifact in Artifact.discover_from_glob(grpc_config.artifact_glob))
        if grpc_config.artifact_glob
        else []
    )
    # bootstrap_scene_id is intentionally unused: the bootstrap (if any) is validated by its eager
    # start_worker_pool load/attach, not re-driven through _validate_startup_scenes (see startup_scene_ids).
    bootstrap_path, _bootstrap_scene_id, extra_scene_artifacts = _plan_glob_scenes(matches)
    renderer_name = _renderer_name_from_backend(grpc_config.renderer)
    if renderer_name == "default":
        # The multi-process pool is gsplat-only; an unspecified (DEFAULT) renderer resolves to gsplat
        # here (rather than the unvalidated PyTorch-forward fallback). Unsupported renderer names still hard-fail in
        # `start_worker_pool` (it bypasses the IPC weight-share).
        print("multiproc: renderer 'default' -> 'gsplat' (the multi-process pool is validated only for gsplat)")
        renderer_name = _SUPPORTED_RENDERER

    # MPS must be started externally (operator / container entrypoint / initContainer) before launch, since
    # CUDA is initialized at import time. Here we only verify it: warn — or fail with --require-mps — if it
    # isn't up.
    mps_preflight(num_workers, require_mps, progress=print)

    # On-demand scene download cache (same knobs as serve), reused for remote x-nre-scene-url.
    # The on-disk cache must hold at least the resident-scene budget (cache_size) or an evicted remote
    # scene's file could be dropped before re-attach. Auto-align the capacities and report at INFO unless
    # --multiproc-require-scene-cache asks for a hard fail.
    download_cache_size = _resolve_download_cache_size(
        grpc_config.download_cache_size, grpc_config.cache_size, require_scene_cache=require_scene_cache, progress=print
    )
    scene_cache = SceneCache(cache_dir=Path(grpc_config.download_cache_dir).expanduser(), max_size=download_cache_size)
    downloader = Downloader()

    metrics_dir = grpc_config.metrics_output_dir
    pool = start_worker_pool(
        bootstrap_path,
        num_workers=num_workers,
        renderer_name=renderer_name,
        enable_editing_actors=grpc_config.enable_editing_actors,
        # Register every other glob scene as owner-loadable (served lazily by scene_id).
        extra_scene_artifacts=extra_scene_artifacts,
        # Cap resident scenes at cache_size (drop-in parity with serve's BackendCache LRU) and
        # size each worker's cache to match, so the coordinator's coordinated eviction — not the worker LRU
        # — bounds memory; idle LRU scenes are evicted (workers detach, owner releases) above the cap.
        worker_scene_cache_size=grpc_config.cache_size,
        max_scenes=grpc_config.cache_size,
        # Egocar hood masks + Harmonizer/Difix, wired into every worker exactly as serve.
        post_process=PostProcessConfig.from_grpc_config(grpc_config),
        metrics_output_dir=metrics_dir,
        ray_chunk_size=grpc_config.ray_chunk_size,
        # Bound the parallel worker startup VRAM peak (1 = serialized; <=0 = all at once).
        max_concurrent_bringup=max_concurrent_bringup,
        # Inference-only ray-stride (popped from the CLI before GrpcServerConfig, like single-process
        # serve): threaded to every owner + worker Backend.from_artifact so pool renders match
        # single-process renders at the same invocation (the drop-in guarantee).
        tracks_timestamp_ray_stride=tracks_timestamp_ray_stride,
        progress=print,
    )
    # Wire the on-disk download cache's eviction to the coordinator so evicting a remote scene's cached file
    # invalidates its reload mapping: an ID-only request for it then returns NOT_FOUND (never UNKNOWN
    # from a vanished file). Local file:// scenes are permanent registrations and are unaffected.
    coordinator = getattr(pool, "coordinator", None)
    if coordinator is not None:
        scene_cache.set_on_evict(coordinator.handle_cache_eviction)

    # Honor test_scenes_are_valid in multiprocess mode: before the proxy becomes SERVING, sequentially
    # load + attach every lazily-registered startup glob scene through the real owner/coordinator/all-worker
    # barrier. Any failure tears the partial fleet down and exits startup nonzero (no partially validated
    # proxy serves). The eager bootstrap (if any) is already loaded + attached on every worker by
    # start_worker_pool, so it is validated by that eager startup and is not re-driven here; this list is the
    # on-demand extras only (an empty glob -> [] -> a clean zero-scene no-op). On-demand
    # (x-nre-scene-url) scenes validate on their first scene-using RPC.
    startup_scene_ids = list(extra_scene_artifacts)
    validated_startup_scenes = 0
    if grpc_config.test_scenes_are_valid:
        try:
            validated_startup_scenes = _validate_startup_scenes(coordinator, startup_scene_ids, progress=print)
        except BaseException as exc:
            print(f"multiproc: startup scene validation failed ({exc!r}); tearing down the fleet")
            pool.shutdown()
            raise

    # Effective config + runtime status surfaced via get_server_config.
    config_map = _multiproc_server_config_map(
        grpc_config,
        num_workers=num_workers,
        renderer_name=renderer_name,
        max_concurrent_bringup=max_concurrent_bringup,
        max_in_flight=max_in_flight,
        admission_timeout_s=admission_timeout_s,
        startup_scenes=len(startup_scene_ids),
        validated_startup_scenes=validated_startup_scenes,
        download_cache_size=download_cache_size,
        tracks_timestamp_ray_stride=tracks_timestamp_ray_stride,
    )
    shutdown = PoolShutdown()
    # Default to a clean (requested) stop so a KeyboardInterrupt before serve_proxy returns exits 0.
    reason = PoolTerminalReason.REQUESTED
    try:
        reason = serve_proxy(
            pool,
            host=grpc_config.host or "localhost",
            port=grpc_config.port,
            shutdown_event=shutdown,
            health_port=grpc_config.health_port,
            # Admission (bounds active downstream renders), threaded without
            # renaming from serve-grpc → run_drop_in → serve_proxy → RenderProxy.
            max_in_flight=max_in_flight,
            admission_timeout_s=admission_timeout_s,
            server_config_map=config_map,
            metrics_output_dir=metrics_dir,
            scene_cache=scene_cache,  # Remote x-nre-scene-url downloads
            downloader=downloader,
            progress=print,
        )
    except KeyboardInterrupt:
        print("interrupted; shutting down")  # compatibility: exit 0, but this bypasses the normal proxy drain
    finally:
        pool.shutdown()
        # Pool the workers' render_time_ms + the proxy's e2e into one metrics.yaml the bench reads.
        if metrics_dir:
            sources = [_worker_metrics_dir(metrics_dir, i) for i in range(num_workers)] + [
                os.path.join(metrics_dir, "proxy")
            ]
            if merge_metrics_yaml(metrics_dir, sources):
                print(f"merged worker+proxy metrics -> {os.path.join(metrics_dir, 'metrics.yaml')}")
    # Map the typed terminal reason to an exit code: worker death / scene-pool degradation is nonzero so a
    # supervisor restarts the pod; a requested shutdown is zero.
    if not reason.is_clean:
        print(f"multiproc: exiting nonzero ({reason.value}) so the supervisor restarts the fleet")
    return reason.exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-process gRPC render serving (owner + workers + proxy)")
    parser.add_argument(
        "--artifact",
        required=True,
        help="Path to a single existing server-local .usdz artifact file to serve.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--renderer", choices=("gsplat",), default="gsplat")  # Gsplat-only pool
    parser.add_argument(
        "--enable-editing-actors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Honor per-request actor edits (transient + pool-safe; free for non-edit requests). "
        "Use --no-enable-editing-actors to reject edit requests.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=_DEFAULT_MAX_IN_FLIGHT,
        help="Admission control: cap concurrent in-flight renders fleet-wide (0 = unlimited). "
        "Over-limit requests are shed with RESOURCE_EXHAUSTED (or queued up to --admission-timeout-s).",
    )
    parser.add_argument(
        "--admission-timeout-s",
        type=float,
        default=_DEFAULT_ADMISSION_TIMEOUT_S,
        help="How long an over-limit render waits for a slot before shedding (0 = shed immediately).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Cap resident (attached) scenes fleet-wide (0 = unlimited). Above the cap, idle "
        "least-recently-used scenes are evicted (workers detach, then the owner frees the IPC weights); "
        "a new scene is shed RESOURCE_EXHAUSTED only if every resident scene is pinned or in-flight.",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=None,
        help="Serve gRPC health on this port (default: same as --port). Set a different port for "
        "a dedicated health-only server (parity with serve-grpc --health-port).",
    )
    parser.add_argument(
        "--max-concurrent-bringup",
        type=int,
        default=1,
        help="How many workers to bring up at once at startup (1 = serialized, the default; >1 overlaps "
        "startup; <=0 = all in parallel). Each worker transiently holds a full model copy on the GPU "
        "while loading (before aliasing onto the owner's IPC weights), so this caps the startup VRAM peak.",
    )
    args = parser.parse_args()
    # Block-buffered stdout (the common case when piped to a log) otherwise swallows the readiness
    # lines below, since the owner goes quiet once it is serving the proxy.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    artifact_path = Path(args.artifact)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"artifact not found: {artifact_path}")

    # MPS must be started externally before launch (CUDA initializes at import time); here we only verify
    # it. Without MPS, N workers silently time-slice one GPU and throughput does not scale — warn loudly.
    # This standalone entrypoint has no --require-mps knob, so it only warns (never hard-fails).
    mps_preflight(args.workers, require_mps=False, progress=print)

    pool = start_worker_pool(
        artifact_path,
        num_workers=args.workers,
        renderer_name=args.renderer,
        enable_editing_actors=args.enable_editing_actors,
        max_scenes=args.max_scenes,  # Resident-scene cap + coordinated LRU eviction (0 = off)
        max_concurrent_bringup=args.max_concurrent_bringup,  # Parallel startup cap (VRAM peak bound)
        progress=print,
    )
    shutdown = PoolShutdown()  # set by the proxy's shut_down RPC (requested) or the degraded callback
    reason = PoolTerminalReason.REQUESTED
    try:
        reason = serve_proxy(
            pool,
            host=args.host,
            port=args.port,
            shutdown_event=shutdown,
            health_port=args.health_port,
            max_in_flight=args.max_in_flight,
            admission_timeout_s=args.admission_timeout_s,
            progress=print,
        )
    except KeyboardInterrupt:
        print("interrupted; shutting down")  # compatibility: exit 0, but this bypasses the normal proxy drain
    finally:
        pool.shutdown()
    # Nonzero exit on worker death / scene-pool degradation so a supervisor restarts the fleet.
    if not reason.is_clean:
        print(f"multiproc: exiting nonzero ({reason.value}) so the supervisor restarts the fleet")
    return reason.exit_code
