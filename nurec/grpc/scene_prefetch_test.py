# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import weakref

import pytest

from nurec.grpc.scene_prefetch import SinglePrefetch


def test_take_returns_matching_preparation() -> None:
    prefetch = SinglePrefetch[str, str](lambda key: key.upper())
    assert prefetch.submit("scene")

    assert prefetch.take("scene") == "SCENE"
    assert prefetch.take("scene") is None
    prefetch.close()


def test_failed_worker_start_leaves_hint_available_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    prefetch = SinglePrefetch[str, str](lambda key: key.upper())
    start = threading.Thread.start
    attempts = 0

    def start_once(thread: threading.Thread) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cannot start worker")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_once)
    with pytest.raises(RuntimeError, match="cannot start worker"):
        prefetch.submit("scene")
    assert prefetch.take("scene") is None
    assert prefetch.submit("scene")
    assert prefetch.take("scene") == "SCENE"
    prefetch.close()


def test_worker_releases_consumed_value() -> None:
    released = threading.Event()

    class Value:
        pass

    references: list[weakref.ReferenceType[Value]] = []

    def prepare(_key: str) -> Value:
        value = Value()
        references.append(weakref.ref(value, lambda _ref: released.set()))
        return value

    prefetch = SinglePrefetch(prepare)
    assert prefetch.submit("scene")
    value = prefetch.take("scene")
    del value

    assert released.wait(timeout=2)
    prefetch.close()


@pytest.mark.parametrize(("operation", "admitted"), [("discard", True), ("close", False), ("expire", True)])
def test_ready_value_is_destroyed_outside_lock(monkeypatch: pytest.MonkeyPatch, operation: str, admitted: bool) -> None:
    now = [0.0]
    monkeypatch.setattr("nurec.grpc.scene_prefetch.time.monotonic", lambda: now[0])
    destroying = threading.Event()
    release = threading.Event()

    class Value:
        pass

    def block_destruction() -> None:
        destroying.set()
        release.wait(timeout=2)

    def prepare(_key: str) -> Value:
        value = Value()
        weakref.finalize(value, block_destruction)
        return value

    prefetch = SinglePrefetch(prepare)
    assert prefetch.submit("a")
    with prefetch._condition:
        assert prefetch._condition.wait_for(lambda: prefetch._ready is not None, timeout=2)

    def drop_ready() -> None:
        if operation == "discard":
            prefetch.discard("a")
        elif operation == "close":
            prefetch.close()
        else:
            with prefetch._condition:
                now[0] = 121.0
                prefetch._condition.notify_all()

    dropper = threading.Thread(target=drop_ready)
    dropper.start()
    assert destroying.wait(timeout=2)
    try:
        result: list[bool] = []
        submitter = threading.Thread(target=lambda: result.append(prefetch.submit("b")))
        submitter.start()
        submitter.join(timeout=1)
        assert not submitter.is_alive()
        assert result == [admitted]
    finally:
        release.set()
        dropper.join(timeout=2)
        prefetch.close()


def test_pending_hint_starts_after_current_is_consumed() -> None:
    started = threading.Event()
    release = threading.Event()
    next_started = threading.Event()
    calls: list[str] = []

    def prepare(key: str) -> str:
        calls.append(key)
        if key == "a":
            started.set()
            assert release.wait(timeout=2)
        else:
            next_started.set()
        return key.upper()

    prefetch = SinglePrefetch(prepare)
    assert prefetch.submit("a")
    assert started.wait(timeout=2)
    assert prefetch.submit("a")
    assert prefetch.submit("b")
    assert prefetch.submit("b")
    assert not prefetch.submit("c")
    release.set()
    with prefetch._condition:
        assert prefetch._condition.wait_for(lambda: prefetch._ready is not None, timeout=2)
    assert calls == ["a"]

    assert prefetch.take("a") == "A"
    assert next_started.wait(timeout=2)
    assert prefetch.take("b") == "B"
    assert calls == ["a", "b"]
    prefetch.close()


def test_blocking_submit_preserves_pending_order() -> None:
    admission_waiting = threading.Event()
    admitted: list[bool] = []
    prefetch = SinglePrefetch[str, str](lambda key: key.upper())
    assert prefetch.submit("a")
    assert prefetch.submit("b")

    def submit() -> None:
        admitted.append(
            prefetch.submit(
                "c",
                wait_for_capacity=True,
                check_wait=admission_waiting.set,
            )
        )

    thread = threading.Thread(target=submit)
    thread.start()
    try:
        assert admission_waiting.wait(timeout=2)
        assert not prefetch.submit("d")
        assert not prefetch.submit("d", wait_for_capacity=True)

        assert prefetch.take("a") == "A"
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert admitted == [True]
        assert prefetch.take("b") == "B"
        assert prefetch.take("c") == "C"
    finally:
        prefetch.close()
        with prefetch._condition:
            prefetch._condition.notify_all()
        thread.join(timeout=2)


def test_cancelled_blocking_submit_releases_admission_waiter() -> None:
    prefetch = SinglePrefetch[str, str](lambda key: key.upper())
    assert prefetch.submit("a")
    assert prefetch.submit("b")
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        prefetch.submit(
            "c",
            wait_for_capacity=True,
            check_wait=cancel,
        )

    assert checks == 2
    assert prefetch.take("a") == "A"
    assert prefetch.submit("c", wait_for_capacity=True)
    assert prefetch.take("b") == "B"
    assert prefetch.take("c") == "C"
    prefetch.close()


def test_close_unblocks_admission_waiter() -> None:
    waiting = threading.Event()
    result: list[bool] = []
    prefetch = SinglePrefetch[str, str](lambda key: key.upper())
    assert prefetch.submit("a")
    assert prefetch.submit("b")

    thread = threading.Thread(
        target=lambda: result.append(prefetch.submit("c", wait_for_capacity=True, check_wait=waiting.set))
    )
    thread.start()
    try:
        assert waiting.wait(timeout=2)
        prefetch.close()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert result == [False]
    finally:
        prefetch.close()
        with prefetch._condition:
            prefetch._condition.notify_all()
        thread.join(timeout=2)


def test_discarded_active_work_promotes_pending_hint() -> None:
    started = threading.Event()
    release = threading.Event()
    next_started = threading.Event()
    calls: list[str] = []

    def prepare(key: str) -> str:
        calls.append(key)
        if key == "a":
            started.set()
            assert release.wait(timeout=2)
        else:
            next_started.set()
        return key.upper()

    prefetch = SinglePrefetch(prepare)
    assert prefetch.submit("a")
    assert started.wait(timeout=2)
    assert prefetch.submit("b")
    prefetch.discard("a")
    release.set()

    assert prefetch.take("a") is None
    assert next_started.wait(timeout=2)
    assert prefetch.take("b") == "B"
    assert calls == ["a", "b"]
    prefetch.close()


def test_discarded_ready_work_promotes_pending_hint() -> None:
    next_started = threading.Event()

    def prepare(key: str) -> str:
        if key == "b":
            next_started.set()
        return key.upper()

    prefetch = SinglePrefetch(prepare)
    assert prefetch.submit("a")
    assert prefetch.submit("b")
    with prefetch._condition:
        assert prefetch._condition.wait_for(lambda: prefetch._ready is not None, timeout=2)
    assert not next_started.is_set()

    prefetch.discard("a")
    assert next_started.wait(timeout=2)
    assert prefetch.take("b") == "B"
    prefetch.close()


def test_out_of_order_take_discards_active_predecessor() -> None:
    started = threading.Event()
    release = threading.Event()

    def prepare(key: str) -> str:
        if key == "a":
            started.set()
            assert release.wait(timeout=2)
        return key.upper()

    prefetch = SinglePrefetch(prepare)
    assert prefetch.submit("a")
    assert started.wait(timeout=2)
    assert prefetch.submit("b")

    assert prefetch.take("b") is None
    assert prefetch.submit("c")
    release.set()
    with prefetch._condition:
        assert prefetch._condition.wait_for(lambda: prefetch._ready is not None, timeout=2)
    assert prefetch.take("c") == "C"
    prefetch.close()


def test_out_of_order_take_discards_ready_predecessor() -> None:
    def cancel() -> None:
        raise RuntimeError("cancelled")

    prefetch = SinglePrefetch[str, str](lambda key: key.upper())
    assert prefetch.submit("a")
    with prefetch._condition:
        assert prefetch._condition.wait_for(lambda: prefetch._ready is not None, timeout=2)
    assert prefetch.submit("b")

    with pytest.raises(RuntimeError, match="cancelled"):
        prefetch.take("b", check_wait=cancel)
    assert prefetch.take("b") is None
    assert prefetch.submit("c")
    assert prefetch.take("c") == "C"
    prefetch.close()


@pytest.mark.parametrize("raises", [False, True])
def test_failed_preparation_releases_capacity_without_take(raises: bool) -> None:
    next_started = threading.Event()

    def prepare(key: str) -> str | None:
        if key == "a":
            if raises:
                raise ValueError("bad scene")
            return None
        next_started.set()
        return key.upper()

    prefetch = SinglePrefetch(prepare)
    assert prefetch.submit("a")
    assert prefetch.submit("b")

    assert next_started.wait(timeout=2)
    assert prefetch.take("a") is None
    assert prefetch.take("b") == "B"
    prefetch.close()


def test_failed_key_can_be_prepared_again() -> None:
    attempts = 0

    def prepare(key: str) -> str | None:
        nonlocal attempts
        attempts += 1
        return None if attempts == 1 else key

    prefetch = SinglePrefetch(prepare)
    try:
        assert prefetch.submit("scene")
        assert prefetch.take("scene") is None
        assert prefetch.submit("scene")
        assert prefetch.take("scene") == "scene"
        assert attempts == 2
    finally:
        prefetch.close()


@pytest.mark.parametrize("queued", [False, True])
def test_unused_ready_value_expires(monkeypatch: pytest.MonkeyPatch, queued: bool) -> None:
    now = [0.0]
    monkeypatch.setattr("nurec.grpc.scene_prefetch.time.monotonic", lambda: now[0])
    released = threading.Event()

    class Value:
        pass

    def prepare(key: str) -> Value:
        value = Value()
        if key == "a":
            weakref.finalize(value, released.set)
        return value

    prefetch = SinglePrefetch(prepare, ready_timeout_s=120.0)
    try:
        assert prefetch.submit("a")
        with prefetch._condition:
            assert prefetch._condition.wait_for(lambda: prefetch._ready is not None, timeout=2)
        if queued:
            assert prefetch.submit("b")
        assert prefetch.take("unrelated") is None
        assert not released.is_set()
        with prefetch._condition:
            now[0] = 121.0
            prefetch._condition.notify_all()
        assert released.wait(timeout=2)
        assert prefetch.take("a") is None
        if not queued:
            assert prefetch.submit("b")
        assert isinstance(prefetch.take("b"), Value)
    finally:
        prefetch.close()


def test_take_does_not_wait_for_discarded_predecessor() -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def prepare(key: str) -> str:
        calls.append(key)
        if key == "a":
            started.set()
            assert release.wait(timeout=2)
        return key

    prefetch = SinglePrefetch(prepare)
    try:
        assert prefetch.submit("a")
        assert started.wait(timeout=2)
        prefetch.discard("a")
        assert prefetch.submit("b")
        assert prefetch.take("b") is None
        assert calls == ["a"]
        assert prefetch.submit("c")
        release.set()
        with prefetch._condition:
            assert prefetch._condition.wait_for(lambda: prefetch._ready is not None, timeout=2)
        assert prefetch.take("c") == "c"
        assert calls == ["a", "c"]
    finally:
        release.set()
        prefetch.close()


def test_cancelled_wait_does_not_discard_ready_preparation() -> None:
    next_started = threading.Event()

    def prepare(key: str) -> str:
        if key == "next":
            next_started.set()
        return key.upper()

    def cancel() -> None:
        raise RuntimeError("cancelled")

    prefetch = SinglePrefetch(prepare)
    assert prefetch.submit("scene")
    assert prefetch.submit("next")
    with prefetch._condition:
        assert prefetch._condition.wait_for(lambda: prefetch._ready is not None, timeout=2)

    with pytest.raises(RuntimeError, match="cancelled"):
        prefetch.take("scene", check_wait=cancel)

    assert not next_started.is_set()
    assert prefetch.take("scene") == "SCENE"
    assert next_started.wait(timeout=2)
    assert prefetch.take("next") == "NEXT"
    prefetch.close()


def test_close_unblocks_waiter(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    result: list[str | None] = []

    def prepare(key: str) -> str:
        started.set()
        release.wait()
        return key

    prefetch = SinglePrefetch(prepare)
    condition_wait = prefetch._condition.wait

    def wait(timeout: float | None = None) -> bool:
        waiting.set()
        return condition_wait(timeout)

    thread = threading.Thread(target=lambda: result.append(prefetch.take("scene")))
    try:
        assert prefetch.submit("scene")
        assert started.wait(timeout=2)
        monkeypatch.setattr(prefetch._condition, "wait", wait)
        thread.start()
        assert waiting.wait(timeout=2)
        prefetch.close()
        prefetch.close()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert result == [None]
    finally:
        prefetch.close()
        release.set()
        # Also wake the reader when testing a missing close notification.
        with prefetch._condition:
            prefetch._condition.notify_all()
        if thread.is_alive():
            thread.join(timeout=2)
        if prefetch._thread is not None:
            prefetch._thread.join(timeout=2)
