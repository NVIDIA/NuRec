# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import threading
import time

from typing import Callable, List, Optional, Set, Union

import pytest
import torch

from nurec.grpc.cache_keys import BackendCacheKey, SceneCacheKey
from nurec.grpc.serve import (
    BACKEND_CACHE_WAIT_POLL_S,
    Backend,
    BackendCache,
    BackendCacheWaitPolicy,
    BackendInUseError,
    BackendReservation,
    CacheFullError,
    CacheWaitAbortedError,
    CacheWaitAbortedReason,
    NoSpareBackendsError,
)


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture
def make_backend() -> Callable[[str], Backend]:
    def _backend_maker(_scene_id: str) -> Backend:
        # The test doesn't need a fully configured backend, so we just return a dummy one with None values
        return Backend(
            renderable_model=None,  # type: ignore
            camera_bank=None,  # type: ignore
            world_to_nurec=None,  # type: ignore
            lidar_bank=None,  # type: ignore
            asset_bank=None,  # type: ignore
            source_artifact=None,  # type: ignore
            backend_cache_key=BackendCacheKey(SceneCacheKey(scene_id=_scene_id), "0" * 64),
        )

    return _backend_maker


def keys_in_use(cache: BackendCache) -> Set[Union[str, None]]:
    return set(backend.cache_key for backend in cache.in_use)


def keys_spare(cache: BackendCache) -> List[Union[str, None]]:
    return [backend.cache_key for backend in cache.spares]


def check_cache(cache: BackendCache, expected_in_use: Set[str], expected_spare: List[str]):
    assert keys_in_use(cache) == expected_in_use
    assert keys_spare(cache) == expected_spare


def check_cache_size(cache: BackendCache, expected_in_use_size: int, expected_spare_size: int):
    assert len(cache.in_use) == expected_in_use_size
    assert len(cache.spares) == expected_spare_size


def check_pending(cache: BackendCache, expected_by_key: dict[str, int]):
    assert cache.pending_by_key == expected_by_key
    assert cache.pending_global == sum(expected_by_key.values())


def check_no_reservations(cache: BackendCache):
    check_pending(cache, {})
    assert cache.pending_reservations == {}
    assert cache._claimed_spares_by_reservation == {}


def capture_cleanup(cache: BackendCache, monkeypatch) -> List[Backend]:
    """Record backends passed to _cleanup_evicted_backends instead of freeing them."""
    cleaned_backends: List[Backend] = []

    def record_cleanup(backends: List[Backend]) -> None:
        cleaned_backends.extend(backends)
        backends.clear()

    monkeypatch.setattr(cache, "_cleanup_evicted_backends", record_cleanup)
    return cleaned_backends


class _ConcurrentAcquire:
    """Captured state of a put_with_retries() call running on a background thread."""

    def __init__(self) -> None:
        self.factory_called = threading.Event()
        self.done = threading.Event()
        self.results: List[Backend] = []
        self.errors: List[BaseException] = []


@contextlib.contextmanager
def run_concurrent_acquire(
    cache: BackendCache,
    key: str,
    factory: Callable[[], Backend],
    wait_policy: BackendCacheWaitPolicy | None = None,
    cache_miss_validator: Callable[[], None] | None = None,
):
    """Run cache.put_with_retries(key, factory) on a daemon thread; join on exit."""
    handle = _ConcurrentAcquire()

    def _tracked_factory() -> Backend:
        handle.factory_called.set()
        return factory()

    def _run() -> None:
        try:
            handle.results.append(
                cache.put_with_retries(
                    key, _tracked_factory, wait_policy=wait_policy, cache_miss_validator=cache_miss_validator
                )
            )
        except Exception as exc:
            handle.errors.append(exc)
        finally:
            handle.done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        yield handle
    finally:
        thread.join(timeout=2.0)
        assert not thread.is_alive()


class TestBackendCache:
    def test_invalid_maxsize(self):
        """Test that BackendCache rejects invalid maxsize values"""
        with pytest.raises(ValueError, match="BackendCache maxsize must be positive, got 0"):
            BackendCache(maxsize=0)

        with pytest.raises(ValueError, match="BackendCache maxsize must be positive, got -1"):
            BackendCache(maxsize=-1)

        with pytest.raises(ValueError, match="BackendCache maxsize must be positive, got -10"):
            BackendCache(maxsize=-10)

    def test_invalid_max_per_key(self):
        """Test that BackendCache rejects invalid max_per_key values"""
        with pytest.raises(ValueError, match="BackendCache max_per_key must be positive when set, got 0"):
            BackendCache(maxsize=3, max_per_key=0)

        with pytest.raises(ValueError, match="BackendCache max_per_key must be positive when set, got -1"):
            BackendCache(maxsize=3, max_per_key=-1)

        with pytest.raises(ValueError, match="BackendCache max_per_key must be <= maxsize, got 4 > 3"):
            BackendCache(maxsize=3, max_per_key=4)

    def test_backend_hash_eq_contract(self, make_backend):
        """Test that Backend maintains hash/eq contract using object identity"""
        b1 = make_backend("scene-1")
        b2 = make_backend("scene-1")  # Different object, same scene_id

        # Different objects should not be equal, even with same cache_key
        assert b1 != b2
        assert hash(b1) != hash(b2)

        # Same object should be equal to itself
        assert b1 == b1
        assert hash(b1) == hash(b1)

        # Hash should be stable even when cache_key is mutated
        original_hash = hash(b1)
        b1.cache_key = "scene-1"
        assert hash(b1) == original_hash

        # Can be added to set multiple times (different objects)
        backend_set = {b1, b2}
        assert len(backend_set) == 2

        # Adding same object twice doesn't increase set size
        backend_set.add(b1)
        assert len(backend_set) == 2

    def test_put(self, make_backend):
        """Test adding a backend to the cache"""
        cache = BackendCache(maxsize=3)

        b = make_backend("scene-id-0")
        cache.put("scene-id-0", b)

        check_cache(cache, {"scene-id-0"}, [])

    def test_checkin(self, make_backend):
        """Test checking in a backend makes it spare"""
        cache = BackendCache(maxsize=3)

        b = make_backend("scene-id-0")
        cache.put("scene-id-0", b)
        cache.checkin(b)

        check_cache(cache, set(), ["scene-id-0"])

    def test_count_based_lru_eviction(self, make_backend):
        """Test count-based LRU eviction with spare backends"""
        # Max 3 backends in cache
        cache = BackendCache(maxsize=3)

        scene_ids = []

        # Add 3 backends and check in (all become spares)
        for i in range(3):
            scene_id = f"scene-id-{i}"
            scene_ids.append(scene_id)

            b = make_backend(scene_id)
            cache.put(scene_id, b)
            cache.checkin(b)

            check_cache_size(cache, 0, i + 1)
            check_cache(cache, set(), scene_ids)

        # Add more backends to cause eviction (oldest spares evicted due to count limit)
        for i in range(3, 6):
            scene_id = f"scene-id-{i}"
            scene_ids.append(scene_id)
            scene_ids.pop(0)  # Remove oldest

            b = make_backend(scene_id)
            cache.put(scene_id, b)
            cache.checkin(b)

            # Should maintain 3 backends (maxsize limit)
            check_cache_size(cache, 0, 3)
            check_cache(cache, set(), scene_ids)

    def test_in_use_backends_not_evicted(self, make_backend):
        """Test that in-use backends are not evicted when adding new backends"""
        cache = BackendCache(maxsize=3)

        # Add 3 spare backends
        for i in range(3):
            b = make_backend(f"spare-{i}")
            cache.put(f"spare-{i}", b)
            cache.checkin(b)

        check_cache_size(cache, 0, 3)

        # Add in-use backends - should evict spares, not exceed maxsize
        in_use_ids = set()
        for i in range(3):  # Only add up to maxsize
            scene_id = f"in-use-{i}"
            in_use_ids.add(scene_id)

            b = make_backend(scene_id)
            cache.put(scene_id, b)  # no checkin - stays in-use

            # Each new in-use backend evicts a spare
            expected_spares = max(0, 3 - (i + 1))
            check_cache_size(cache, i + 1, expected_spares)

        # All spares should be evicted, only in-use remain
        check_cache(cache, in_use_ids, [])

    def test_checkout_on_empty_cache(self):
        """Test checkout returns None when cache is empty"""
        cache = BackendCache(maxsize=3)

        check_cache_size(cache, 0, 0)

        assert cache.checkout("scene-id-0") is None

    def test_checkout_used_backend(self, make_backend):
        """Test checkout returns None for in-use backends"""
        cache = BackendCache(maxsize=3)

        scene_id = "x-test-scene-id"

        b = make_backend(scene_id)
        cache.put(scene_id, b)  # b is in-use

        check_cache(cache, {"x-test-scene-id"}, [])

        assert cache.checkout(scene_id) is None
        check_cache(cache, {"x-test-scene-id"}, [])

    def test_checkout_unused_backend(self, make_backend):
        """Test checkout returns spare backend and marks as in-use"""
        cache = BackendCache(maxsize=3)

        scene_id = "scene-id"

        b = make_backend(scene_id)
        cache.put(scene_id, b)  # b is in-use
        cache.checkin(b)  # b is no longer in-use

        checked_out = cache.checkout(scene_id)
        assert id(checked_out) == id(b)
        check_cache(cache, {scene_id}, [])

    def test_evict_one_spare(self, make_backend):
        """Test evict_one_spare method used for OOM retry"""
        cache = BackendCache(maxsize=5)

        # Add 3 spare backends
        for i in range(3):
            b = make_backend(f"scene-{i}")
            cache.put(f"scene-{i}", b)
            cache.checkin(b)

        check_cache_size(cache, 0, 3)

        # Evict one spare
        assert cache.evict_one_spare() is True
        check_cache_size(cache, 0, 2)
        check_cache(cache, set(), ["scene-1", "scene-2"])  # scene-0 evicted (oldest)

        # Evict another
        assert cache.evict_one_spare() is True
        check_cache_size(cache, 0, 1)
        check_cache(cache, set(), ["scene-2"])  # scene-1 evicted

        # Evict last
        assert cache.evict_one_spare() is True
        check_cache_size(cache, 0, 0)
        check_cache(cache, set(), [])

        # No more spares to evict
        assert cache.evict_one_spare() is False

    def test_evict_one_spare_skips_in_use(self, make_backend):
        """Test evict_one_spare only evicts spares, not in-use backends"""
        cache = BackendCache(maxsize=5)

        # Add in-use backend
        b1 = make_backend("in-use")
        cache.put("in-use", b1)  # stays in-use

        # Add spare backend
        b2 = make_backend("spare")
        cache.put("spare", b2)
        cache.checkin(b2)

        check_cache_size(cache, 1, 1)

        # Evict one spare
        assert cache.evict_one_spare() is True
        check_cache_size(cache, 1, 0)
        check_cache(cache, {"in-use"}, [])

        # No more spares, in-use backend remains
        assert cache.evict_one_spare() is False
        check_cache(cache, {"in-use"}, [])

    def test_loaded_scene_counts_empty_cache(self):
        cache = BackendCache(maxsize=3)

        capacity, entries = cache.loaded_scene_counts()

        assert capacity == 3
        assert entries == []

    def test_loaded_scene_counts_single_spare(self, make_backend):
        cache = BackendCache(maxsize=3)
        backend = make_backend("scene-a")
        cache.put("scene-a", backend)
        cache.checkin(backend)

        capacity, entries = cache.loaded_scene_counts()

        assert capacity == 3
        assert entries == [("scene-a", 1, 1)]

    def test_loaded_scene_counts_single_in_use(self, make_backend):
        cache = BackendCache(maxsize=3)
        cache.put("scene-a", make_backend("scene-a"))

        capacity, entries = cache.loaded_scene_counts()

        assert capacity == 3
        assert entries == [("scene-a", 1, 0)]

    def test_loaded_scene_counts_duplicate_in_use_scene_ids(self, make_backend):
        cache = BackendCache(maxsize=3)
        cache.put("scene-a", make_backend("scene-a"))
        cache.put("scene-a", make_backend("scene-a"))

        capacity, entries = cache.loaded_scene_counts()

        assert capacity == 3
        assert entries == [("scene-a", 2, 0)]

    def test_loaded_scene_counts_mixed_in_use_and_spare_for_scene(self, make_backend):
        cache = BackendCache(maxsize=3)
        cache.put("scene-a", make_backend("scene-a"))
        spare = make_backend("scene-a")
        cache.put("scene-a", spare)
        cache.checkin(spare)

        capacity, entries = cache.loaded_scene_counts()

        assert capacity == 3
        assert entries == [("scene-a", 2, 1)]

    def test_loaded_scene_counts_multiple_scenes_sorted(self, make_backend):
        cache = BackendCache(maxsize=3)
        cache.put("A", make_backend("A"))
        spare_a = make_backend("A")
        cache.put("A", spare_a)
        cache.checkin(spare_a)
        spare_b = make_backend("B")
        cache.put("B", spare_b)
        cache.checkin(spare_b)

        capacity, entries = cache.loaded_scene_counts()

        assert capacity == 3
        assert entries == [("A", 2, 1), ("B", 1, 1)]

    def test_loaded_scene_counts_capacity(self):
        cache = BackendCache(maxsize=3)

        capacity, _ = cache.loaded_scene_counts()

        assert capacity == 3

    def test_loaded_scene_counts_after_lru_eviction(self, make_backend):
        cache = BackendCache(maxsize=3)

        for scene_id in ("A", "B", "C"):
            backend = make_backend(scene_id)
            cache.put(scene_id, backend)
            cache.checkin(backend)

        backend = make_backend("D")
        cache.put("D", backend)
        cache.checkin(backend)

        capacity, entries = cache.loaded_scene_counts()

        assert capacity == 3
        assert entries == [("B", 1, 1), ("C", 1, 1), ("D", 1, 1)]
        assert sum(loaded_count for _, loaded_count, _ in entries) == 3

    def test_lru_order_preserved(self, make_backend):
        """Test that LRU order is maintained correctly"""
        cache = BackendCache(maxsize=3)

        # Add and check in backends in order
        for i in range(5):
            b = make_backend(f"scene-{i}")
            cache.put(f"scene-{i}", b)
            cache.checkin(b)

        # Only last 3 should remain (scene-2, scene-3, scene-4)
        check_cache(cache, set(), ["scene-2", "scene-3", "scene-4"])

    def test_checkout_after_eviction(self, make_backend):
        """Test that checkout returns None for evicted backends"""
        cache = BackendCache(maxsize=2)

        # Add and checkin backend
        b = make_backend("scene-old")
        cache.put("scene-old", b)
        cache.checkin(b)

        # Add two more to trigger eviction of scene-old
        b1 = make_backend("scene-1")
        cache.put("scene-1", b1)
        cache.checkin(b1)

        b2 = make_backend("scene-2")
        cache.put("scene-2", b2)
        cache.checkin(b2)

        # scene-old should be evicted
        check_cache(cache, set(), ["scene-1", "scene-2"])

        # Checkout of evicted backend returns None
        assert cache.checkout("scene-old") is None

    def test_multiple_checkins_checkout_cycles(self, make_backend):
        """Test multiple checkin/checkout cycles"""
        cache = BackendCache(maxsize=3)

        b = make_backend("scene-id")
        cache.put("scene-id", b)

        # Checkin and checkout multiple times
        for _ in range(3):
            cache.checkin(b)
            check_cache(cache, set(), ["scene-id"])

            checked_out = cache.checkout("scene-id")
            assert id(checked_out) == id(b)
            check_cache(cache, {"scene-id"}, [])

    def test_maxsize_enforced_when_all_in_use(self, make_backend):
        """Test that maxsize cannot be exceeded even when all backends are in-use.

        Also verifies that the rejected backend's GPU memory is freed before
        CacheFullError is raised (internal cleanup via del + torch.cuda.empty_cache).
        """
        cache = BackendCache(maxsize=3)

        # Add 3 backends, all in-use (no checkin)
        for i in range(3):
            b = make_backend(f"scene-{i}")
            cache.put(f"scene-{i}", b)

        check_cache_size(cache, 3, 0)
        check_cache(cache, {"scene-0", "scene-1", "scene-2"}, [])

        # Try to add a 4th backend - should fail because all 3 are in-use
        # The rejected backend's GPU memory will be freed internally before the exception
        b4 = make_backend("scene-3")
        with pytest.raises(CacheFullError, match="Cannot add backend 'scene-3': all 3 cache slots are in-use"):
            cache.put("scene-3", b4)

        # Backend's cache_key should be reset to None for clean disposal
        assert b4.cache_key is None

        # Cache should still have only 3 backends
        check_cache_size(cache, 3, 0)
        check_cache(cache, {"scene-0", "scene-1", "scene-2"}, [])

        # Now if we check in one backend, we should be able to add a new one
        b0 = next(b for b in cache.in_use if b.cache_key == "scene-0")
        cache.checkin(b0)
        check_cache_size(cache, 2, 1)

        # Now we can add scene-3, which will evict scene-0
        b4 = make_backend("scene-3")
        cache.put("scene-3", b4)
        check_cache_size(cache, 3, 0)
        check_cache(cache, {"scene-1", "scene-2", "scene-3"}, [])

    def test_direct_put_enforces_max_per_key(self, make_backend):
        """Test direct put cannot bypass the per-key backend limit."""
        cache = BackendCache(maxsize=5, max_per_key=2)

        first = make_backend("scene")
        second = make_backend("scene")
        cache.put("scene", first)
        cache.put("scene", second)

        rejected = make_backend("scene")
        with pytest.raises(CacheFullError, match="per-key limit 2 is full"):
            cache.put("scene", rejected)

        assert rejected.cache_key is None
        check_cache(cache, {"scene"}, [])
        assert len(cache.in_use) == 2

    def test_reservation_counts_against_max_per_key(self, make_backend):
        """Test pending reservations count against max_per_key until released."""
        cache = BackendCache(maxsize=3, max_per_key=1)

        reservation = cache.reserve("scene")
        check_pending(cache, {"scene": 1})

        with pytest.raises(CacheFullError, match="per-key limit 1 is full"):
            cache.reserve("scene")

        cache.release_reservation(reservation)
        check_pending(cache, {})

        next_reservation = cache.reserve("scene")
        backend = make_backend("scene")
        cache.put_reserved(next_reservation, backend)

        check_pending(cache, {})
        check_cache(cache, {"scene"}, [])

    def test_reservation_token_must_be_active_and_owned(self, make_backend):
        """Test reservations cannot be consumed by another key or reused."""
        cache = BackendCache(maxsize=3)

        scene_reservation = cache.reserve("scene")
        wrong_key_reservation = BackendReservation(
            key="other-scene",
            token=scene_reservation.token,
            scene_id="other-scene",
        )

        with pytest.raises(ValueError, match="Backend reservation key mismatch"):
            cache.put_reserved(wrong_key_reservation, make_backend("other-scene"))

        backend = make_backend("scene")
        cache.put_reserved(scene_reservation, backend)

        with pytest.raises(ValueError, match="Unknown or already consumed backend reservation"):
            cache.release_reservation(scene_reservation)

        with pytest.raises(ValueError, match="Unknown or already consumed backend reservation"):
            cache.put_reserved(scene_reservation, make_backend("scene"))

    def test_put_with_retries_reserves_before_factory(self, make_backend):
        """Test factory runs while its cache slot is already reserved."""
        cache = BackendCache(maxsize=3, max_per_key=1)

        def factory():
            check_pending(cache, {"scene": 1})
            with pytest.raises(CacheFullError, match="per-key limit 1 is full"):
                cache.reserve("scene")
            return make_backend("scene")

        backend = cache.put_with_retries("scene", factory)

        assert backend.cache_key == "scene"
        check_pending(cache, {})
        check_cache(cache, {"scene"}, [])

    def test_put_with_retries_success(self, make_backend):
        """Test put_with_retries with successful backend creation on first attempt"""
        cache = BackendCache(maxsize=3)

        # Simple factory that succeeds
        def factory():
            return make_backend("test-scene")

        backend = cache.put_with_retries("test-scene", factory)

        assert backend is not None
        assert backend.cache_key == "test-scene"
        check_cache_size(cache, 1, 0)
        check_cache(cache, {"test-scene"}, [])

    def test_put_with_retries_oom_with_eviction_succeeds(self, make_backend):
        """Test put_with_retries handles OOM by evicting spares and retrying"""
        cache = BackendCache(maxsize=3)

        # Add 2 spare backends
        for i in range(2):
            b = make_backend(f"spare-{i}")
            cache.put(f"spare-{i}", b)
            cache.checkin(b)

        check_cache_size(cache, 0, 2)

        # Factory that fails once with OOM, then succeeds
        call_count = [0]

        def factory_with_oom():
            call_count[0] += 1
            if call_count[0] == 1:
                import torch

                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return make_backend("new-scene")

        # Should evict a spare and retry successfully
        backend = cache.put_with_retries("new-scene", factory_with_oom, enable_eviction=True)

        assert backend is not None
        assert backend.cache_key == "new-scene"
        assert call_count[0] == 2  # Called twice (failed once, succeeded once)
        check_cache_size(cache, 1, 1)  # One spare evicted

    def test_put_with_retries_oom_no_spares_raises(self, make_backend):
        """Test put_with_retries raises NoSpareBackendsError when OOM and no spares available"""
        cache = BackendCache(maxsize=3)

        # Leave one global slot available so reservation succeeds, then fail
        # during backend construction with no spare backends to evict.
        for i in range(2):
            b = make_backend(f"in-use-{i}")
            cache.put(f"in-use-{i}", b)

        check_cache_size(cache, 2, 0)

        # Factory that always fails with OOM
        def factory_oom():
            import torch

            raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # Should raise NoSpareBackendsError since no spares to evict
        with pytest.raises(NoSpareBackendsError, match="No spare backends available to evict"):
            cache.put_with_retries("new-scene", factory_oom, enable_eviction=True)

    def test_put_with_retries_oom_eviction_disabled(self, make_backend):
        """Test put_with_retries with eviction disabled still retries on OOM"""
        cache = BackendCache(maxsize=3)

        # Add a spare backend
        b = make_backend("spare")
        cache.put("spare", b)
        cache.checkin(b)

        # Factory that fails twice with OOM, then succeeds
        call_count = [0]

        def factory_with_oom():
            call_count[0] += 1
            if call_count[0] <= 2:
                import torch

                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return make_backend("new-scene")

        # Should retry without evicting the spare
        backend = cache.put_with_retries("new-scene", factory_with_oom, max_retries=5, enable_eviction=False)

        assert backend is not None
        assert call_count[0] == 3  # Called 3 times (failed twice, succeeded on 3rd)
        check_cache_size(cache, 1, 1)  # Spare still present (not evicted)

    def test_reserve_raises_when_global_capacity_full(self, make_backend):
        """Test direct reservations fail fast when no cache capacity is available."""
        cache = BackendCache(maxsize=2)

        # Fill cache with in-use backends
        for i in range(2):
            b = make_backend(f"in-use-{i}")
            cache.put(f"in-use-{i}", b)

        with pytest.raises(CacheFullError, match="all 2 cache slots are in-use"):
            cache.reserve("new-scene")

    def test_put_with_retries_propagates_other_exceptions(self):
        """Test put_with_retries propagates non-OOM exceptions immediately"""
        cache = BackendCache(maxsize=3)

        # Factory that raises a different exception
        def factory_error():
            raise ValueError("Simulated error")

        # Should raise ValueError immediately without retrying
        with pytest.raises(ValueError, match="Simulated error"):
            cache.put_with_retries("new-scene", factory_error)

        # Cache should still be empty
        check_cache_size(cache, 0, 0)

    def test_direct_put_rejects_when_per_key_capacity_full(self, make_backend):
        cache = BackendCache(maxsize=3, max_per_key=1)

        b1 = make_backend("scene-1")
        cache.put("scene-1", b1)

        b2 = make_backend("scene-1")
        with pytest.raises(CacheFullError, match="per-key limit 1 is full"):
            cache.put("scene-1", b2)

        assert b2.cache_key is None
        check_cache(cache, {"scene-1"}, [])

    def test_per_key_limit_counts_spares(self, make_backend):
        cache = BackendCache(maxsize=3, max_per_key=1)

        b1 = make_backend("scene-1")
        cache.put("scene-1", b1)
        cache.checkin(b1)

        b2 = make_backend("scene-1")
        with pytest.raises(CacheFullError, match="per-key limit 1 is full"):
            cache.put("scene-1", b2)

        check_cache(cache, set(), ["scene-1"])

    def test_scene_id_replaces_spare_from_another_lookup_key(self, make_backend, monkeypatch):
        cache = BackendCache(maxsize=3, max_per_key=1)
        old_backend = make_backend("scene")
        cache.put("backend-v1", old_backend)
        cache.checkin(old_backend)
        cleaned_backends = capture_cleanup(cache, monkeypatch)

        new_backend = cache.put_with_retries(
            "backend-v2",
            lambda: make_backend("scene"),
            scene_id="scene",
        )

        assert new_backend.cache_key == "backend-v2"
        assert new_backend.scene_id == "scene"
        assert cleaned_backends == [old_backend]
        check_cache(cache, {"backend-v2"}, [])

    def test_reservation_counts_toward_global_capacity(self):
        cache = BackendCache(maxsize=1)

        reservation = cache.reserve("scene-1")

        with pytest.raises(CacheFullError, match="all 1 cache slots"):
            cache.reserve("scene-2")

        cache.release_reservation(reservation)
        check_no_reservations(cache)

    def test_reservation_counts_toward_per_key_capacity(self):
        cache = BackendCache(maxsize=3, max_per_key=1)

        reservation = cache.reserve("scene-1")

        with pytest.raises(CacheFullError, match="per-key limit 1 is full"):
            cache.reserve("scene-1")

        cache.release_reservation(reservation)

    def test_put_reserved_consumes_own_pending_slot(self, make_backend):
        cache = BackendCache(maxsize=1, max_per_key=1)

        reservation = cache.reserve("scene-1")
        backend = make_backend("scene-1")
        cache.put_reserved(reservation, backend)

        check_cache(cache, {"scene-1"}, [])
        check_no_reservations(cache)

    def test_put_reserved_rejects_consumed_token(self, make_backend):
        cache = BackendCache(maxsize=2)

        reservation = cache.reserve("scene-1")
        cache.put_reserved(reservation, make_backend("scene-1"))

        with pytest.raises(ValueError, match="Unknown or already consumed backend reservation"):
            cache.put_reserved(reservation, make_backend("scene-1"))

    def test_release_reservation_rejects_consumed_token(self):
        cache = BackendCache(maxsize=2)

        reservation = cache.reserve("scene-1")
        cache.release_reservation(reservation)

        with pytest.raises(ValueError, match="Unknown or already consumed backend reservation"):
            cache.release_reservation(reservation)

    def test_reservation_rejects_wrong_key_token(self):
        cache = BackendCache(maxsize=2)

        reservation = cache.reserve("scene-1")
        wrong_key_reservation = type(reservation)(
            key="scene-2",
            token=reservation.token,
            scene_id=reservation.scene_id,
        )

        with pytest.raises(ValueError, match="Backend reservation key mismatch"):
            cache.release_reservation(wrong_key_reservation)

        cache.release_reservation(reservation)

    def test_put_with_retries_releases_reservation_on_factory_exception(self):
        cache = BackendCache(maxsize=2)

        def factory_error():
            raise RuntimeError("factory failed")

        with pytest.raises(RuntimeError, match="factory failed"):
            cache.put_with_retries("scene-1", factory_error)

        check_no_reservations(cache)

    def test_failed_replacement_restores_identical_spare_and_lru(self, make_backend):
        cache = BackendCache(maxsize=2, max_per_key=2)
        healthy_backend = make_backend("scene-a")
        cache.put("scene-a", healthy_backend)
        cache.checkin(healthy_backend)
        newer_backend = make_backend("scene-c")
        cache.put("scene-c", newer_backend)
        cache.checkin(newer_backend)

        def factory_error() -> Backend:
            # The claimed spare stays reusable while its replacement builds.
            assert cache.loaded_scene_counts() == (2, [("scene-a", 1, 1), ("scene-c", 1, 1)])
            raise RuntimeError("factory failed")

        with pytest.raises(RuntimeError, match="factory failed"):
            cache.put_with_retries("scene-b", factory_error)

        assert cache.spares == [healthy_backend, newer_backend]
        check_no_reservations(cache)
        assert cache.checkout("scene-a") is healthy_backend

    def test_cancelled_replacement_before_factory_restores_identical_spare(self, make_backend):
        cache = BackendCache(maxsize=1, max_per_key=1)
        healthy_backend = make_backend("scene-a")
        cache.put("scene-a", healthy_backend)
        cache.checkin(healthy_backend)
        factory_called = threading.Event()

        with pytest.raises(CacheWaitAbortedError) as exc_info:
            cache.put_with_retries(
                "scene-b",
                lambda: factory_called.set() or make_backend("scene-b"),
                wait_policy=BackendCacheWaitPolicy(
                    should_continue=lambda: False,
                    abort_reason=lambda: CacheWaitAbortedReason.CANCELLED,
                ),
            )

        assert exc_info.value.reason == CacheWaitAbortedReason.CANCELLED
        assert not factory_called.is_set()
        assert cache.spares == [healthy_backend]
        check_no_reservations(cache)

    def test_successful_replacement_cleans_claimed_spare_once(self, make_backend, monkeypatch):
        cache = BackendCache(maxsize=1, max_per_key=1)
        claimed_spare = make_backend("scene-a")
        cache.put("scene-a", claimed_spare)
        cache.checkin(claimed_spare)
        cleaned_backends = capture_cleanup(cache, monkeypatch)

        reservation = cache.reserve("scene-b")
        assert cache.spares == [claimed_spare]
        assert cache._claimed_spares_by_reservation == {reservation.token: claimed_spare}

        replacement = make_backend("scene-b")
        cache.put_reserved(reservation, replacement)

        assert cleaned_backends == [claimed_spare]
        check_cache(cache, {"scene-b"}, [])
        check_no_reservations(cache)

    @pytest.mark.parametrize("reservation_action", ["release", "commit"])
    def test_explicit_eviction_of_claimed_spare_keeps_reservation_valid(
        self, make_backend, monkeypatch, reservation_action: str
    ):
        cache = BackendCache(maxsize=1, max_per_key=1)
        claimed_spare = make_backend("scene-a")
        cache.put("scene-a", claimed_spare)
        cache.checkin(claimed_spare)
        cleaned_backends = capture_cleanup(cache, monkeypatch)
        reservation = cache.reserve("scene-b")

        assert cache.evict_one_spare() is True
        assert cleaned_backends == [claimed_spare]
        assert cache.spares == []
        assert cache._claimed_spares_by_reservation == {}
        check_pending(cache, {"scene-b": 1})

        if reservation_action == "release":
            cache.release_reservation(reservation)
            check_cache(cache, set(), [])
        else:
            cache.put_reserved(reservation, make_backend("scene-b"))
            check_cache(cache, {"scene-b"}, [])

        assert cleaned_backends == [claimed_spare]
        check_no_reservations(cache)

    def test_replacement_reservations_claim_distinct_spares(self, make_backend):
        cache = BackendCache(maxsize=2, max_per_key=2)
        oldest_backend = make_backend("scene-a")
        cache.put("scene-a", oldest_backend)
        cache.checkin(oldest_backend)
        newest_backend = make_backend("scene-b")
        cache.put("scene-b", newest_backend)
        cache.checkin(newest_backend)

        first = cache.reserve("scene-c")
        second = cache.reserve("scene-d")

        assert cache._claimed_spares_by_reservation == {
            first.token: oldest_backend,
            second.token: newest_backend,
        }
        assert cache.spares == [oldest_backend, newest_backend]

        cache.release_reservation(first)
        cache.release_reservation(second)
        check_no_reservations(cache)

    def test_claimed_spare_stays_checkoutable_and_evicts_on_checkin(self, make_backend, monkeypatch):
        """A claimed spare is reusable while its replacement builds; evicted on checkin if used."""
        cache = BackendCache(maxsize=1)
        backend_a = make_backend("a")
        cache.put("a", backend_a)
        cache.checkin(backend_a)

        cleaned_backends = capture_cleanup(cache, monkeypatch)

        reservation = cache.reserve("b")
        assert cache._claimed_spares_by_reservation == {reservation.token: backend_a}

        # Warm reuse: the claimed spare must still be checkout-able while "b" builds.
        assert cache.checkout("a") is backend_a
        check_cache(cache, {"a"}, [])

        replacement = make_backend("b")
        cache.put_reserved(reservation, replacement)

        # Transient overshoot: both the superseded backend and its replacement are in use.
        check_cache(cache, {"a", "b"}, [])
        check_no_reservations(cache)
        assert cleaned_backends == []
        # Both slots deducted: the superseded backend collapses back onto its replacement's slot.
        assert cache._total_count_locked() == 1

        cache.checkin(backend_a)
        # Superseded backend is evicted on checkin, never re-entering spares.
        check_cache(cache, {"b"}, [])
        assert cleaned_backends == [backend_a]

        cache.checkin(replacement)
        check_cache(cache, set(), ["b"])

    @pytest.mark.parametrize("use_wait_policy", [False, True])
    def test_put_with_retries_waits_for_global_capacity_and_creates_after_checkin(
        self, make_backend, use_wait_policy: bool
    ):
        cache = BackendCache(maxsize=1, max_per_key=1)
        existing_backend = make_backend("scene-1")
        cache.put("scene-1", existing_backend)

        wait_policy = BackendCacheWaitPolicy(wait_deadline=time.monotonic() + 5.0) if use_wait_policy else None

        with run_concurrent_acquire(
            cache,
            "scene-2",
            lambda: make_backend("scene-2"),
            wait_policy=wait_policy,
        ) as call:
            assert not call.done.wait(0.1)
            assert not call.factory_called.is_set()
            cache.checkin(existing_backend)
            assert call.done.wait(1.0)

        assert call.errors == []
        assert len(call.results) == 1
        assert call.results[0].cache_key == "scene-2"
        assert call.factory_called.is_set()
        check_cache(cache, {"scene-2"}, [])

    def test_put_with_retries_waits_for_per_key_capacity_and_reuses_backend(self, make_backend):
        cache = BackendCache(maxsize=2, max_per_key=1)
        existing_backend = make_backend("scene-1")
        cache.put("scene-1", existing_backend)

        def factory():
            raise AssertionError("factory should not be called")

        with run_concurrent_acquire(cache, "scene-1", factory) as call:
            assert not call.done.wait(0.1)
            assert not call.factory_called.is_set()
            cache.checkin(existing_backend)
            assert call.done.wait(1.0)

        assert call.errors == []
        assert call.results == [existing_backend]
        assert not call.factory_called.is_set()
        check_cache(cache, {"scene-1"}, [])

    def test_put_with_retries_waits_for_pending_per_key_reservation(self, make_backend):
        cache = BackendCache(maxsize=2, max_per_key=1)
        reservation = cache.reserve("scene-1")

        with run_concurrent_acquire(cache, "scene-1", lambda: make_backend("scene-1")) as call:
            assert not call.done.wait(0.1)
            assert not call.factory_called.is_set()
            cache.release_reservation(reservation)
            assert call.done.wait(1.0)

        assert call.errors == []
        assert len(call.results) == 1
        assert call.results[0].cache_key == "scene-1"
        assert call.factory_called.is_set()
        check_pending(cache, {})
        check_cache(cache, {"scene-1"}, [])

    def test_pending_key_does_not_block_unrelated_replacement(self, make_backend, monkeypatch):
        cache = BackendCache(maxsize=2, max_per_key=2)
        capture_cleanup(cache, monkeypatch)

        a_in_use = make_backend("scene-A")
        cache.put("scene-A", a_in_use)

        a_reservation = cache.reserve("scene-A")
        cache.checkin(a_in_use)

        with run_concurrent_acquire(
            cache,
            "scene-B",
            lambda: make_backend("scene-B"),
            wait_policy=BackendCacheWaitPolicy(wait_deadline=time.monotonic() + 1.0),
        ) as call:
            assert call.factory_called.wait(1.0)
            assert call.done.wait(1.0)

        cache.release_reservation(a_reservation)
        assert call.errors == []
        assert len(call.results) == 1
        assert call.results[0].cache_key == "scene-B"
        check_cache(cache, {"scene-B"}, [])
        check_no_reservations(cache)

    def test_wait_times_out_without_calling_factory(self, make_backend):
        cache = BackendCache(maxsize=1, max_per_key=1)
        cache.put("scene-1", make_backend("scene-1"))

        def fail_factory() -> Backend:
            raise AssertionError("factory should not run")

        start = time.monotonic()
        with pytest.raises(CacheWaitAbortedError) as exc_info:
            cache.put_with_retries(
                "scene-2",
                fail_factory,
                wait_policy=BackendCacheWaitPolicy(wait_deadline=time.monotonic() + 0.25),
            )

        elapsed = time.monotonic() - start
        assert exc_info.value.reason == CacheWaitAbortedReason.SERVER_TIMEOUT
        assert elapsed < BACKEND_CACHE_WAIT_POLL_S + 0.5
        assert cache.pending_global == 0

    def test_wait_aborts_when_should_continue_becomes_false(self, make_backend):
        cache = BackendCache(maxsize=1, max_per_key=1)
        cache.put("scene-1", make_backend("scene-1"))

        allow_wait = [True]

        def should_continue() -> bool:
            return allow_wait[0]

        with run_concurrent_acquire(
            cache,
            "scene-2",
            lambda: make_backend("scene-2"),
            wait_policy=BackendCacheWaitPolicy(
                should_continue=should_continue,
                abort_reason=lambda: CacheWaitAbortedReason.CANCELLED,
                wait_deadline=time.monotonic() + 30.0,
            ),
        ) as call:
            assert not call.done.wait(0.2)
            assert not call.factory_called.is_set()
            allow_wait[0] = False
            assert call.done.wait(BACKEND_CACHE_WAIT_POLL_S + 1.0)

        assert call.results == []
        assert len(call.errors) == 1
        assert isinstance(call.errors[0], CacheWaitAbortedError)
        assert call.errors[0].reason == CacheWaitAbortedReason.CANCELLED
        assert not call.factory_called.is_set()
        assert cache.pending_global == 0

    def test_cache_miss_validator_runs_without_cache_lock(self, make_backend):
        """Test that a blocking cache_miss_validator does not stall unrelated cache operations."""
        cache = BackendCache(maxsize=3)
        other_backend = make_backend("scene-other")
        cache.put("scene-other", other_backend)

        validator_entered = threading.Event()
        release_validator = threading.Event()
        checkin_done = threading.Event()

        def blocking_validator() -> None:
            validator_entered.set()
            assert release_validator.wait(2.0)

        def concurrent_checkin() -> None:
            cache.checkin(other_backend)
            checkin_done.set()

        with run_concurrent_acquire(
            cache,
            "scene-cold",
            lambda: make_backend("scene-cold"),
            cache_miss_validator=blocking_validator,
        ) as call:
            assert validator_entered.wait(1.0)
            checkin_thread = threading.Thread(target=concurrent_checkin, daemon=True)
            checkin_thread.start()
            # Must complete while the validator is still blocked: the validator
            # may not be invoked with the cache lock held.
            assert checkin_done.wait(1.0)
            release_validator.set()
            assert call.done.wait(1.0)
            checkin_thread.join(timeout=1.0)
            assert not checkin_thread.is_alive()

        assert call.errors == []
        assert len(call.results) == 1
        assert call.results[0].cache_key == "scene-cold"
        check_cache(cache, {"scene-cold"}, ["scene-other"])

    def test_put_with_retries_aborts_before_factory_releases_reservation(self, make_backend):
        cache = BackendCache(maxsize=2)
        cache.put("scene-1", make_backend("scene-1"))

        def fail_factory() -> Backend:
            raise AssertionError("factory should not run")

        with pytest.raises(CacheWaitAbortedError) as exc_info:
            cache.put_with_retries(
                "scene-2",
                fail_factory,
                wait_policy=BackendCacheWaitPolicy(
                    should_continue=lambda: False,
                    abort_reason=lambda: CacheWaitAbortedReason.CANCELLED,
                ),
            )

        assert exc_info.value.reason == CacheWaitAbortedReason.CANCELLED
        assert cache.pending_global == 0
        check_cache(cache, {"scene-1"}, [])

    def test_cancellation_after_successful_factory_commits_constructed_backend(self, make_backend):
        # A client whose deadline expires while its backend builds must still warm
        # the cache: discarding a finished build would livelock retries that each
        # pay a full construction. should_continue stays True for the pre-factory
        # guard and flips to False only after the build completes.
        cache = BackendCache(maxsize=2, max_per_key=1)
        constructed = make_backend("scene-b")
        build_finished = threading.Event()

        def factory() -> Backend:
            build_finished.set()
            return constructed

        result = cache.put_with_retries(
            "scene-b",
            factory,
            wait_policy=BackendCacheWaitPolicy(
                should_continue=lambda: not build_finished.is_set(),
                abort_reason=lambda: CacheWaitAbortedReason.CANCELLED,
            ),
        )

        assert result is constructed
        assert constructed.cache_key == "scene-b"
        check_cache(cache, {"scene-b"}, [])
        check_no_reservations(cache)

    def test_cancellation_during_successful_factory_commits_over_claimed_spare(self, make_backend, monkeypatch):
        # Cancellation after a successful build still commits: the reservation's
        # claimed spare (scene-a) is evicted to make room for the finished
        # replacement (scene-b) rather than discarding the paid-for build.
        cache = BackendCache(maxsize=1, max_per_key=1)
        healthy_backend = make_backend("scene-a")
        cache.put("scene-a", healthy_backend)
        cache.checkin(healthy_backend)
        replacement = make_backend("scene-b")
        cancelled = threading.Event()
        finish_factory = threading.Event()

        def factory() -> Backend:
            assert finish_factory.wait(2.0)
            return replacement

        cleaned_backends = capture_cleanup(cache, monkeypatch)

        with run_concurrent_acquire(
            cache,
            "scene-b",
            factory,
            wait_policy=BackendCacheWaitPolicy(
                should_continue=lambda: not cancelled.is_set(),
                abort_reason=lambda: CacheWaitAbortedReason.CANCELLED,
            ),
        ) as call:
            assert call.factory_called.wait(1.0)
            # The claimed spare stays reusable while its replacement builds.
            assert cache.loaded_scene_counts() == (1, [("scene-a", 1, 1)])
            cancelled.set()
            finish_factory.set()
            assert call.done.wait(1.0)

        assert call.errors == []
        assert call.results == [replacement]
        assert replacement.cache_key == "scene-b"
        assert cleaned_backends == [healthy_backend]
        check_cache(cache, {"scene-b"}, [])
        check_no_reservations(cache)

    def test_oom_after_cancellation_does_not_mutate_cache(self, make_backend, monkeypatch):
        import torch

        cache = BackendCache(maxsize=2, max_per_key=2)
        oldest_backend = make_backend("scene-a")
        cache.put("scene-a", oldest_backend)
        cache.checkin(oldest_backend)
        newest_backend = make_backend("scene-c")
        cache.put("scene-c", newest_backend)
        cache.checkin(newest_backend)
        cancelled = threading.Event()
        finish_factory = threading.Event()
        checkout_calls: List[str] = []
        eviction_calls: List[Optional[Set[str]]] = []
        original_checkout = cache.checkout
        original_evict_one_spare = cache.evict_one_spare

        def factory_oom() -> Backend:
            assert finish_factory.wait(2.0)
            raise torch.cuda.OutOfMemoryError("Simulated OOM")

        def record_checkout(key: str) -> Backend | None:
            checkout_calls.append(key)
            return original_checkout(key)

        def record_eviction(excluded_keys: Set[str] | None = None) -> bool:
            eviction_calls.append(excluded_keys)
            return original_evict_one_spare(excluded_keys)

        monkeypatch.setattr(cache, "checkout", record_checkout)
        monkeypatch.setattr(cache, "evict_one_spare", record_eviction)
        cleaned_backends = capture_cleanup(cache, monkeypatch)

        with run_concurrent_acquire(
            cache,
            "scene-b",
            factory_oom,
            wait_policy=BackendCacheWaitPolicy(
                should_continue=lambda: not cancelled.is_set(),
                abort_reason=lambda: CacheWaitAbortedReason.CANCELLED,
            ),
        ) as call:
            assert call.factory_called.wait(1.0)
            cancelled.set()
            finish_factory.set()
            assert call.done.wait(1.0)

        assert call.results == []
        assert len(call.errors) == 1
        assert isinstance(call.errors[0], CacheWaitAbortedError)
        assert checkout_calls == []
        assert eviction_calls == []
        assert cleaned_backends == []
        assert cache.spares == [oldest_backend, newest_backend]
        check_no_reservations(cache)

    def test_unset_per_key_limit_fails_immediately_without_factory(self, make_backend):
        cache = BackendCache(maxsize=1)
        cache.put("scene-a", make_backend("scene-a"))
        factory_called = threading.Event()

        with pytest.raises(CacheFullError, match="all 1 cache slots are in-use"):
            cache.put_with_retries(
                "scene-b",
                lambda: factory_called.set() or make_backend("scene-b"),
                wait_policy=BackendCacheWaitPolicy(wait_deadline=time.monotonic() + 0.1),
            )

        assert not factory_called.is_set()
        check_no_reservations(cache)

    def test_oom_retry_reuses_same_key_spare_before_eviction(self, make_backend):
        cache = BackendCache(maxsize=3)

        same_key_backend = make_backend("scene-1")
        cache.put("scene-1", same_key_backend)
        other_backend = make_backend("scene-2")
        cache.put("scene-2", other_backend)
        cache.checkin(other_backend)

        def factory_oom():
            cache.checkin(same_key_backend)
            import torch

            raise torch.cuda.OutOfMemoryError("Simulated OOM")

        backend = cache.put_with_retries("scene-1", factory_oom)

        assert backend is same_key_backend
        check_cache(cache, {"scene-1"}, ["scene-2"])

    def test_evict_one_spare_honors_excluded_keys(self, make_backend):
        cache = BackendCache(maxsize=3)

        for scene_id in ("scene-1", "scene-2"):
            backend = make_backend(scene_id)
            cache.put(scene_id, backend)
            cache.checkin(backend)

        assert cache.evict_one_spare(excluded_keys={"scene-1"}) is True
        check_cache(cache, set(), ["scene-1"])


class TestBackendCacheWorkerSceneTableApi:
    """`remove_by_key` / `grow_maxsize`: the lock-and-notify surface the multiproc `_WorkerSceneTable` drives."""

    def test_remove_by_key_refuses_when_in_use(self, make_backend):
        # F001: a by-key eviction must NEVER unlink a backend a render still holds. An in-use backend
        # (checked out, never checked in) is refused with BackendInUseError; the backend stays in `in_use`
        # with its key intact, so the in-flight render can still `checkin` it and a later eviction (after
        # checkin makes it a spare) removes it cleanly.
        cache = BackendCache(maxsize=3)
        backend = make_backend("scene-a")
        cache.put("scene-a", backend)  # in-use, never checked in

        with pytest.raises(BackendInUseError, match="in use"):
            cache.remove_by_key("scene-a")

        assert backend in cache.in_use  # NOT removed
        assert backend.cache_key == "scene-a"  # key preserved
        check_cache(cache, {"scene-a"}, [])
        check_no_reservations(cache)

        # Once the render checks the backend back in (spare), removal succeeds.
        cache.checkin(backend)
        removed = cache.remove_by_key("scene-a")
        assert removed is backend
        assert removed.cache_key is None
        check_cache(cache, set(), [])

    def test_remove_by_key_removes_from_spares(self, make_backend):
        cache = BackendCache(maxsize=3)
        sibling = make_backend("scene-a")
        cache.put("scene-a", sibling)
        cache.checkin(sibling)
        target = make_backend("scene-b")
        cache.put("scene-b", target)
        cache.checkin(target)  # both scene-a and scene-b are spares

        removed = cache.remove_by_key("scene-b")

        assert removed is target
        assert removed.cache_key is None
        check_cache(cache, set(), ["scene-a"])  # the sibling spare is left intact

    def test_remove_by_key_unknown_key_returns_none(self, make_backend):
        cache = BackendCache(maxsize=3)
        resident = make_backend("scene-a")
        cache.put("scene-a", resident)
        cache.checkin(resident)

        assert cache.remove_by_key("does-not-exist") is None
        check_cache(cache, set(), ["scene-a"])  # nothing evicted on a miss
        assert cache.remove_by_key("scene-a") is resident  # the resident key still resolves

    def test_remove_by_key_refuses_in_use_occupant_and_waiter_wakes_on_checkin(self, make_backend):
        # F001 at the cache level: while an in-use occupant holds the only slot, a capacity waiter blocks.
        # remove_by_key REFUSES to evict the in-use occupant (a render is in flight against it), so the waiter
        # stays blocked — a live backend is never unlinked out from under a render to wake a waiter. Only when
        # the render checks the occupant in (freeing the slot) does the waiter proceed.
        cache = BackendCache(maxsize=1, max_per_key=1)
        occupant = make_backend("scene-1")
        cache.put("scene-1", occupant)  # in-use: the only slot, no spare to evict

        with run_concurrent_acquire(cache, "scene-2", lambda: make_backend("scene-2")) as call:
            assert not call.done.wait(0.1)  # blocked at global capacity
            assert not call.factory_called.is_set()

            with pytest.raises(BackendInUseError):
                cache.remove_by_key("scene-1")  # refused: the occupant's backend is in use
            assert not call.done.wait(0.1)  # still blocked — the in-use backend was NOT evicted
            assert occupant in cache.in_use

            cache.checkin(occupant)  # the render completes -> occupant becomes a spare, freeing the slot
            assert call.done.wait(2.0)  # the waiter woke and proceeded

        assert call.errors == []
        assert len(call.results) == 1
        assert call.results[0].cache_key == "scene-2"
        check_cache(cache, {"scene-2"}, [])

    def test_grow_maxsize_increments(self):
        cache = BackendCache(maxsize=2)

        cache.grow_maxsize(1)
        assert cache.maxsize == 3
        assert cache.loaded_scene_counts()[0] == 3  # the reported capacity grows too

        cache.grow_maxsize(5)
        assert cache.maxsize == 8

    def test_grow_maxsize_rejects_non_positive(self):
        cache = BackendCache(maxsize=2)

        with pytest.raises(ValueError, match="grow_maxsize requires n >= 1, got 0"):
            cache.grow_maxsize(0)
        with pytest.raises(ValueError, match="grow_maxsize requires n >= 1, got -3"):
            cache.grow_maxsize(-3)
        assert cache.maxsize == 2  # rejected grows leave capacity unchanged

    def test_grow_maxsize_wakes_capacity_waiter(self, make_backend):
        cache = BackendCache(maxsize=1, max_per_key=1)
        occupant = make_backend("scene-1")
        cache.put("scene-1", occupant)  # in-use: the only slot, no spare to evict

        with run_concurrent_acquire(cache, "scene-2", lambda: make_backend("scene-2")) as call:
            assert not call.done.wait(0.1)  # blocked at global capacity
            assert not call.factory_called.is_set()

            cache.grow_maxsize(1)  # opens a second slot and notifies capacity_changed

            assert call.done.wait(2.0)  # the waiter woke and proceeded

        assert call.errors == []
        assert len(call.results) == 1
        assert call.results[0].cache_key == "scene-2"
        assert cache.maxsize == 2
        check_cache(cache, {"scene-1", "scene-2"}, [])


class TestGetBackendContextManager:
    """Tests for the get_backend context manager OOM recovery behavior.
    These tests verify that:
    1. Backends are always returned to the cache (checkin called), even on exceptions
    2. OOM errors trigger memory cleanup and spare backend eviction
    3. Exceptions are properly re-raised after cleanup
    Note: These tests use a helper context manager that mirrors the pattern in
    SensorSimService.get_backend() to verify the behavior without needing to
    instantiate the full service with all its dependencies.
    """

    @staticmethod
    @contextlib.contextmanager
    def _get_backend_pattern(cache: BackendCache, key: str, make_backend: Callable[[str], Backend]):
        """
        A test helper that mirrors the get_backend pattern from serve.py.
        This allows us to test the context manager behavior without instantiating
        the full SensorSimService with all its dependencies.
        """
        import gc

        import torch

        # Try to get an available backend (mirrors get_backend logic)
        backend = cache.checkout(key)

        if backend is None:
            backend = cache.put_with_retries(key, lambda: make_backend(key))

        try:
            yield backend
        except torch.cuda.OutOfMemoryError:
            # OOM recovery - mirrors the fix in serve.py
            # In real code: torch.cuda.empty_cache() and gc.collect()
            gc.collect()

            # Evict spare backends to free memory
            while cache.evict_one_spare():
                pass

            raise
        finally:
            # Always return backend to cache - THE KEY FIX
            cache.checkin(backend)

    def test_checkin_called_on_normal_exit(self, make_backend):
        """Test that checkin is called when context manager exits normally."""
        cache = BackendCache(maxsize=3)

        # Pre-populate cache with a spare backend
        b = make_backend("scene-1")
        cache.put("scene-1", b)
        cache.checkin(b)
        check_cache(cache, set(), ["scene-1"])

        # Use the context manager - backend should be checked out then back in
        with self._get_backend_pattern(cache, "scene-1", make_backend) as backend:
            assert backend is b
            check_cache(cache, {"scene-1"}, [])  # In use during context

        # After context exits, backend should be back in spares
        check_cache(cache, set(), ["scene-1"])

    def test_checkin_called_on_exception(self, make_backend):
        """Test that checkin is called even when an exception occurs.
        This verifies the fix for the bug where backends got stuck in in_use
        when exceptions occurred during rendering.
        """
        cache = BackendCache(maxsize=3)

        # Pre-populate cache
        b = make_backend("scene-1")
        cache.put("scene-1", b)
        cache.checkin(b)
        check_cache(cache, set(), ["scene-1"])

        # Use context manager with an exception - backend should still be returned
        with pytest.raises(RuntimeError, match="Simulated rendering error"):
            with self._get_backend_pattern(cache, "scene-1", make_backend):
                check_cache(cache, {"scene-1"}, [])  # In use during context
                raise RuntimeError("Simulated rendering error")

        # Backend should be back in spares, not stuck in in_use
        check_cache(cache, set(), ["scene-1"])

    def test_backend_not_stuck_on_oom(self, make_backend):
        """Test that backends don't get stuck in in_use when OOM occurs.
        This is the main bug that was fixed - without try/finally, backends
        would remain in in_use forever after OOM, causing cache to fill up.
        """
        import torch

        cache = BackendCache(maxsize=3)

        # Add multiple spare backends
        for i in range(3):
            b = make_backend(f"scene-{i}")
            cache.put(f"scene-{i}", b)
            cache.checkin(b)

        check_cache(cache, set(), ["scene-0", "scene-1", "scene-2"])

        # Use context manager with OOM - backend should still be returned
        # and spare backends should be evicted
        with pytest.raises(torch.cuda.OutOfMemoryError):
            with self._get_backend_pattern(cache, "scene-1", make_backend):
                check_cache(cache, {"scene-1"}, ["scene-0", "scene-2"])
                raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # Backend should be back in spares (not stuck in in_use)
        # Other spares should have been evicted by OOM handler
        check_cache(cache, set(), ["scene-1"])

    def test_oom_evicts_all_spares(self, make_backend):
        """Test that OOM recovery evicts all spare backends to free memory."""
        import torch

        cache = BackendCache(maxsize=5)

        # Add 4 spare backends
        for i in range(4):
            b = make_backend(f"spare-{i}")
            cache.put(f"spare-{i}", b)
            cache.checkin(b)

        check_cache_size(cache, 0, 4)

        # Use context manager with OOM - all OTHER spares should be evicted
        with pytest.raises(torch.cuda.OutOfMemoryError):
            with self._get_backend_pattern(cache, "spare-0", make_backend):
                # spare-0 is now in-use, spare-1,2,3 are spares
                check_cache_size(cache, 1, 3)
                raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # After OOM: spare-1,2,3 evicted, spare-0 returned to spares
        check_cache_size(cache, 0, 1)
        check_cache(cache, set(), ["spare-0"])

    def test_exception_propagates_after_cleanup(self, make_backend):
        """Test that exceptions are re-raised after cleanup."""
        cache = BackendCache(maxsize=3)

        b = make_backend("scene-1")
        cache.put("scene-1", b)
        cache.checkin(b)

        # Exception should propagate after cleanup
        with pytest.raises(ValueError, match="Test error"):
            with self._get_backend_pattern(cache, "scene-1", make_backend):
                raise ValueError("Test error")

        # Backend should still be checked in despite the exception
        check_cache(cache, set(), ["scene-1"])

    def test_new_backend_created_when_not_in_cache(self, make_backend):
        """Test that a new backend is created when not found in cache."""
        cache = BackendCache(maxsize=3)

        check_cache_size(cache, 0, 0)

        # Use context manager for a scene not in cache - should create new backend
        with self._get_backend_pattern(cache, "new-scene", make_backend) as backend:
            assert backend.cache_key == "new-scene"
            check_cache(cache, {"new-scene"}, [])

        # After context, new backend should be in spares
        check_cache(cache, set(), ["new-scene"])

    # -------------------------------------------------------------------------
    # Tests demonstrating the bug that the try/finally fix prevents
    # -------------------------------------------------------------------------

    @staticmethod
    @contextlib.contextmanager
    def _get_backend_pattern_buggy(cache: BackendCache, key: str, make_backend: Callable[[str], Backend]):
        """
        Simulates the BUGGY version of get_backend (before the fix).
        Missing try/finally causes backend to get stuck in in_use on exception.
        DO NOT USE THIS PATTERN IN PRODUCTION CODE!
        """
        backend = cache.checkout(key)

        if backend is None:
            backend = make_backend(key)
            cache.put(key, backend)

        yield backend
        # BUG: This line is skipped if an exception occurs!
        cache.checkin(backend)

    def test_buggy_pattern_leaks_backend_on_exception(self, make_backend):
        """Demonstrate the bug: without try/finally, backend gets stuck in in_use.
        This test shows what happens WITHOUT the fix - backends leak into in_use
        and are never returned to spares when exceptions occur.
        """
        cache = BackendCache(maxsize=4)

        # Pre-populate cache with a spare backend
        b = make_backend("scene-1")
        cache.put("scene-1", b)
        cache.checkin(b)
        check_cache(cache, set(), ["scene-1"])

        # Use the BUGGY pattern with an exception
        with pytest.raises(ValueError):
            with self._get_backend_pattern_buggy(cache, "scene-1", make_backend):
                raise ValueError("Test exception")

        # BUG: Backend is stuck in in_use, never returned to spares!
        check_cache(cache, {"scene-1"}, [])  # LEAK - should be in spares!

    def test_buggy_pattern_repeated_oom_fills_in_use(self, make_backend):
        """Demonstrate: buggy version causes in_use to grow with each OOM.
        This was the root cause of the unrecoverable OOM state - each OOM
        left a backend stuck in in_use, eventually filling the cache.
        """
        import torch

        cache = BackendCache(maxsize=4)

        # Simulate multiple OOM errors with buggy version
        for i in range(3):
            with pytest.raises(torch.cuda.OutOfMemoryError):
                with self._get_backend_pattern_buggy(cache, f"scene-{i}", make_backend):
                    raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # BUG: All 3 backends are stuck in in_use!
        check_cache_size(cache, 3, 0)  # LEAK! Should be 0 in_use, 3 spares

    def test_fixed_pattern_repeated_oom_does_not_fill_in_use(self, make_backend):
        """Verify the fix: repeated OOMs don't cause in_use to grow unboundedly.
        With the try/finally fix, backends are always returned to spares,
        allowing the server to recover from OOM conditions.
        """
        import torch

        cache = BackendCache(maxsize=4)

        # Simulate multiple OOM errors with FIXED version
        for i in range(5):
            with pytest.raises(torch.cuda.OutOfMemoryError):
                with self._get_backend_pattern(cache, f"scene-{i}", make_backend):
                    raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # FIXED: in_use should be empty - no backends stuck
        # Note: spares are evicted during OOM recovery, so the last backend
        # is the only one remaining in spares
        check_cache_size(cache, 0, 1)  # Correct! No leaks


class TestConcurrentGetBackend:
    """Many simultaneous same-scene checkouts.

    Mirrors the get_backend pattern (checkout -> create+put when no spare -> checkin)
    driven from N threads to verify a burst of N same-scene requests creates up to N
    distinct in-use replicas concurrently, drains back to empty afterward, and that an
    over-subscribed cache (cache_size < workers) surfaces CacheFullError.
    """

    @staticmethod
    def _hold_backend(
        cache: BackendCache,
        key: str,
        make_backend: Callable[[str], Backend],
        ready: threading.Barrier,
        release: threading.Event,
        held: List[Backend],
        errors: List[Exception],
    ) -> None:
        """Check out (or create) a backend, hold it in-use until released, then check in."""
        try:
            backend = cache.checkout(key)
            if backend is None:
                backend = make_backend(key)
                cache.put(key, backend)
        except Exception as e:  # e.g. CacheFullError when the cache is over-subscribed
            errors.append(e)
            ready.wait()
            return

        try:
            held.append(backend)
            ready.wait()  # all live threads are now holding a backend
            release.wait(timeout=10)  # keep it in-use until the test releases us
        finally:
            cache.checkin(backend)

    def _run_burst(self, cache: BackendCache, n: int, make_backend: Callable[[str], Backend]):
        ready = threading.Barrier(n + 1)  # n workers + the test thread
        release = threading.Event()
        held: List[Backend] = []
        errors: List[Exception] = []

        threads = [
            threading.Thread(
                target=self._hold_backend,
                args=(cache, "scene", make_backend, ready, release, held, errors),
            )
            for _ in range(n)
        ]
        for t in threads:
            t.start()

        # Wait until every worker has either taken a slot or failed; they all hold now.
        ready.wait(timeout=10)
        return threads, release, held, errors

    def test_n_concurrent_same_scene_creates_n_replicas(self, make_backend):
        """cache_size == n: a burst of n same-scene requests holds n distinct replicas,
        raises nothing, and drains back to empty after all check in."""
        n = 8
        cache = BackendCache(maxsize=n)
        threads, release, held, errors = self._run_burst(cache, n, make_backend)

        # All n replicas are checked out simultaneously; distinct objects; no errors.
        assert errors == []
        assert len(held) == n
        assert len(cache.in_use) == n
        assert len({id(b) for b in cache.in_use}) == n

        release.set()
        for t in threads:
            t.join(timeout=10)

        # Everything returned to the cache.
        assert len(cache.in_use) == 0

    def test_burst_exceeding_cache_raises_cache_full(self, make_backend):
        """cache_size < n: only cache_size replicas fit; the overflow gets CacheFullError
        (the condition the startup warning is meant to flag)."""
        n = 8
        maxsize = 4
        cache = BackendCache(maxsize=maxsize)
        threads, release, held, errors = self._run_burst(cache, n, make_backend)

        assert len(held) == maxsize
        assert len(errors) == n - maxsize
        assert all(isinstance(e, CacheFullError) for e in errors)

        release.set()
        for t in threads:
            t.join(timeout=10)

        assert len(cache.in_use) == 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="per-backend lidar stream requires CUDA")
class TestLidarStream:
    """Per-backend lidar CUDA stream (migrated off the composite).

    The stream is owned by the Backend (request context) and threaded into the
    composite's forward_gaussians; concurrent lidar requests run on distinct replicas
    and therefore distinct streams.
    """

    class _StubModel:
        def __init__(self, device: torch.device) -> None:
            self.device = device

    def _backend_on(self, device: torch.device) -> Backend:
        return Backend(
            renderable_model=self._StubModel(device),  # type: ignore[arg-type]
            camera_bank=None,  # type: ignore
            world_to_nurec=None,  # type: ignore
            lidar_bank=None,  # type: ignore
            asset_bank=None,  # type: ignore
            source_artifact=None,  # type: ignore
            backend_cache_key=BackendCacheKey(SceneCacheKey(scene_id="lidar-stream-scene"), "0" * 64),
        )

    def test_stream_on_device_creates_and_reuses(self):
        device = torch.device("cuda", torch.cuda.current_device())
        stream = Backend._stream_on_device(None, device)
        assert isinstance(stream, torch.cuda.Stream)
        assert stream.device == device
        assert Backend._stream_on_device(stream, device) is stream  # reused for same device

    def test_lidar_stream_lazy_on_model_device_and_cached(self):
        device = torch.device("cuda", torch.cuda.current_device())
        backend = self._backend_on(device)

        assert backend._lidar_stream is None  # not created until first use
        stream = backend._get_lidar_stream()
        assert stream.device == device  # created on the model device
        assert backend._lidar_stream is stream
        assert backend._get_lidar_stream() is stream  # cached

    def test_two_backends_have_distinct_lidar_streams(self):
        device = torch.device("cuda", torch.cuda.current_device())
        b1 = self._backend_on(device)
        b2 = self._backend_on(device)

        assert b1._get_lidar_stream() is not b2._get_lidar_stream()
