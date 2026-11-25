# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-deep background prefetch with ordered handoff."""

from __future__ import annotations

import logging
import threading
import time

from collections.abc import Callable
from typing import Generic, TypeVar


log = logging.getLogger(__name__)

_WAIT_POLL_S = 1.0

K = TypeVar("K")
V = TypeVar("V")


class SinglePrefetch(Generic[K, V]):
    """Prepare one key and retain one queued successor.

    Duplicate keys are idempotent. Unused prepared values expire after
    ``ready_timeout_s`` seconds. Failed preparations release their slot.
    """

    def __init__(self, prepare: Callable[[K], V | None], *, ready_timeout_s: float = 120.0) -> None:
        self._prepare = prepare
        self._ready_timeout_s = ready_timeout_s
        self._condition = threading.Condition()
        self._active: K | None = None
        self._discard_active: K | None = None
        self._queued: K | None = None
        self._ready: tuple[K, V, float] | None = None
        self._admission_waiter: object | None = None
        self._closed = False
        self._thread: threading.Thread | None = None

    def submit(
        self,
        key: K,
        *,
        wait_for_capacity: bool = False,
        check_wait: Callable[[], None] | None = None,
    ) -> bool:
        """Admit a hint without replacing existing work.

        Return ``True`` if the key is admitted or already active, ready, or
        queued. Return ``False`` if closed or if a new hint cannot be admitted.

        With ``wait_for_capacity=True``, one caller may wait for space; a
        competing waiter returns ``False``. In this mode, ``check_wait`` runs
        before admission and while waiting, and its exceptions propagate.
        """
        with self._condition:
            if self._closed:
                return False
            if wait_for_capacity and check_wait is not None:
                check_wait()
            if self._try_submit_locked(key):
                return True
            if not wait_for_capacity or self._admission_waiter is not None:
                return False

            token = object()
            self._admission_waiter = token
            try:
                while not self._closed:
                    if check_wait is not None:
                        check_wait()
                    if self._try_submit_locked(key, admission_token=token):
                        return True
                    self._condition.wait(timeout=_WAIT_POLL_S if check_wait is not None else None)
                return False
            finally:
                if self._admission_waiter is token:
                    self._admission_waiter = None
                    self._condition.notify_all()

    def _try_submit_locked(self, key: K, *, admission_token: object | None = None) -> bool:
        """Schedule ``key`` without waiting. The condition lock must be held."""
        if self._closed:
            return False
        if self._ready is not None and self._ready[0] == key:
            return True
        if self._queued == key:
            return True
        if self._active == key and self._discard_active != key:
            return True
        if self._admission_waiter is not None and self._admission_waiter is not admission_token:
            return False
        if self._queued is not None:
            return False

        if self._active is not None:
            if self._active == key:
                self._discard_active = None
            else:
                self._queued = key
        elif self._ready is not None:
            self._queued = key
        else:
            self._start_locked()
            self._active = key
        self._condition.notify_all()
        return True

    def take(self, key: K, *, check_wait: Callable[[], None] | None = None) -> V | None:
        """Consume ``key`` in admission order, or return ``None`` for fallback.

        ``check_wait`` runs before each wait and before consumption, so an
        aborted caller leaves a ready value available to a later request.
        Taking a queued successor before its predecessor discards both hints.
        Missing, failed, unsupported, or closed preparation also returns ``None``.
        """
        with self._condition:
            while not self._closed:
                if self._ready is not None and self._ready[0] == key:
                    if check_wait is not None:
                        check_wait()
                    _, value, _ = self._ready
                    self._ready = None
                    self._promote_locked()
                    self._condition.notify_all()
                    return value
                if self._queued == key:
                    if check_wait is not None:
                        check_wait()
                    if self._ready is not None:
                        # Keep teardown outside the condition lock; a ready value may own a large backend.
                        discarded_ready = self._ready
                        skipped_key = discarded_ready[0]
                        self._ready = None
                    else:
                        assert self._active is not None
                        skipped_key = self._active
                        self._discard_active = self._active
                    self._queued = None
                    self._condition.notify_all()
                    log.warning("Out-of-order scene request; discarding prefetch hints %r and %r", skipped_key, key)
                    return None
                if self._active == key and self._discard_active != key:
                    if check_wait is not None:
                        check_wait()
                    self._condition.wait(timeout=_WAIT_POLL_S if check_wait is not None else None)
                    continue
                return None
            return None

    def discard(self, key: K) -> None:
        """Discard matching work without waiting for active preparation."""
        discarded_ready = None
        with self._condition:
            if self._ready is not None and self._ready[0] == key:
                discarded_ready = self._ready
                self._ready = None
                self._promote_locked()
            if self._queued == key:
                self._queued = None
            if self._active == key:
                self._discard_active = key
            self._condition.notify_all()
        del discarded_ready

    def close(self) -> None:
        """Stop accepting hints and discard queued and ready work.

        Return without waiting for active preparation; discard its result when it finishes.
        """
        discarded_ready = None
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._queued = None
            discarded_ready = self._ready
            self._ready = None
            self._condition.notify_all()
        del discarded_ready

    def _start_locked(self) -> None:
        if self._thread is not None:
            return
        thread = threading.Thread(target=self._run, name="nre-scene-prefetch", daemon=True)
        thread.start()
        self._thread = thread

    def _promote_locked(self) -> None:
        if self._queued is not None:
            self._active = self._queued
            self._queued = None

    def _run(self) -> None:
        while True:
            expired = None
            with self._condition:
                while self._active is None and not self._closed:
                    timeout = None
                    if self._ready is not None:
                        timeout = self._ready[2] - time.monotonic()
                        if timeout <= 0:
                            expired = self._ready
                            self._ready = None
                            self._promote_locked()
                            self._condition.notify_all()
                            break
                    self._condition.wait(timeout=timeout)
                if self._closed:
                    return
                key = self._active

            # Destruction can release a large model; keep it outside the lock.
            del expired
            if key is None:
                continue
            try:
                value = self._prepare(key)
            except Exception:
                log.exception("Scene prefetch failed for %r", key)
                value = None

            with self._condition:
                self._active = None
                discard = self._discard_active == key
                if discard:
                    self._discard_active = None
                if self._closed:
                    return
                if discard or value is None:
                    self._promote_locked()
                else:
                    self._ready = (key, value, time.monotonic() + self._ready_timeout_s)
                self._condition.notify_all()
            value = None
