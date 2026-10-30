# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import shutil
import tempfile
import time
import uuid

from pathlib import Path

import pytest

from nurec.grpc.scene_cache import SceneCache, SceneCacheEviction, SceneCacheEvictionError, SceneCacheValue


@pytest.fixture
def cache_dir():
    """
    Generate a temporary directory path (but not a directory)
    and delete it after the test.

    Note: don't use TemporaryDirectory() because we want to  check if the folder is created by the cache
    """

    temp_path = Path(tempfile.gettempdir()) / f"test-scene-cache-{uuid.uuid4()}"

    yield temp_path

    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def scene_cache(cache_dir: Path):
    assert not cache_dir.exists()
    cache = SceneCache(cache_dir=cache_dir, max_size=10)
    yield cache


@pytest.fixture
def make_temp_file():
    """
    Fixture for a file creation function with a given content.

    Returns:
        Callable[[str], Path]: function that creates a temporary file with the given content
    """

    def _create_temp_file(content: str):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(content.encode("utf-8"))
        temp_file.close()

        return Path(temp_file.name)

    return _create_temp_file


@pytest.fixture
def temp_scene_1(make_temp_file):
    """
    A shortcut fixuture to provide a dummy scene #1.
    """
    temp_file = make_temp_file("x-test-scene-1")

    yield temp_file

    # cleanup
    if Path(temp_file.name).exists():
        Path(temp_file.name).unlink()


@pytest.fixture
def temp_scene_2(make_temp_file):
    """
    A shortcut fixuture to provide a dummy scene #2.
    """
    temp_file = make_temp_file("x-test-scene-2")
    yield temp_file

    # cleanup
    if Path(temp_file.name).exists():
        Path(temp_file.name).unlink()


class TestSceneCache:
    """Test suite for SceneCache class functionality.

    Tests the initialization, scene management, and cache size enforcement
    of the SceneCache implementation.
    """

    def test_create_cache_dir(self, cache_dir: Path):
        """Test SceneCache initialization creates the cache directory correctly."""
        assert not cache_dir.exists()

        SceneCache(cache_dir=cache_dir, max_size=10)

        # check that the cache directory exists
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_new_cache_has_no_scene(self, scene_cache):
        """Test that a newly initialized cache correctly reports no scenes present."""
        assert not scene_cache.has_scene("x-test-scene-1")

    def test_basic_add_and_acquire_scene(self, scene_cache: SceneCache, temp_scene_1: Path):
        """Basic test adding a new scene to the cache.

        Verifies that:
        - Scene is correctly added to cache
        - Original file is removed
        - Scene can be retrieved
        - Content matches original
        """
        expected_path = scene_cache.cache_dir / "x-test-scene-1.usdz"

        lease, _ = scene_cache.add_and_acquire_scene("x-test-scene-1", temp_scene_1)
        lease.release()
        assert lease.path == expected_path
        assert not temp_scene_1.exists()  # the original file should be deleted

        assert scene_cache.has_scene("x-test-scene-1")
        assert scene_cache.get_scene_path("x-test-scene-1") == expected_path
        assert expected_path.read_bytes() == b"x-test-scene-1"

        # get a non-existent scene should raise an error
        with pytest.raises(KeyError) as e:
            scene_cache.get_scene_path("x-test-missing-scene")

        assert str(e.value) == "'Scene x-test-missing-scene not in cache'"

    def test_add_and_acquire_scene_with_same_id_raises_error(
        self, scene_cache: SceneCache, temp_scene_1: Path, temp_scene_2: Path
    ):
        """Test handling of attempts to add duplicate scenes.

        Verifies that:
        - Adding a scene with existing ID raises ValueError
        - Original cached content remains unchanged
        """
        expected_path = scene_cache.cache_dir / "x-test-scene-1.usdz"

        lease, _ = scene_cache.add_and_acquire_scene("x-test-scene-1", temp_scene_1)
        lease.release()
        assert lease.path == expected_path

        # Adding another scene with the same scene id should raise an error
        with pytest.raises(ValueError) as e:
            scene_cache.add_and_acquire_scene("x-test-scene-1", temp_scene_2)

        assert str(e.value) == "Scene x-test-scene-1 already in cache"

        # Make sure the original file is not deleted and the content is the same
        assert expected_path.exists()
        assert expected_path.read_bytes() == b"x-test-scene-1"

    def test_get_scene_path_not_found(self, scene_cache: SceneCache):
        """Test appropriate error handling when retrieving non-existent scene from the empty cache."""
        with pytest.raises(KeyError) as e:
            scene_cache.get_scene_path("x-test-scene-1")

        assert str(e.value) == "'Scene x-test-scene-1 not in cache'"

    def test_enforce_max_size(self, scene_cache: SceneCache, make_temp_file):
        """Test cache size limit enforcement.

        Verifies that:
        - Cache maintains maximum size limit
        - Oldest entries are removed when limit is reached
        - New entries can be added after limit is reached while old ones are removed
        """
        max_size = scene_cache.max_size

        for i in range(max_size * 2):
            temp_file = make_temp_file(f"x-test-scene-{i}")
            expected_path = scene_cache.cache_dir / f"x-test-scene-{i}.usdz"
            lease, _ = scene_cache.add_and_acquire_scene(f"x-test-scene-{i}", temp_file)
            lease.release()
            assert lease.path == expected_path
            assert expected_path.exists()

            # the cache should have at most max_size files
            assert len(scene_cache._cache) <= max_size

            # List files in the cache directory
            files = list(scene_cache.cache_dir.glob("*.usdz"))
            assert len(files) == min(i + 1, max_size)

    def test_pinned_oldest_does_not_block_newer_unpinned_evictions(self, cache_dir: Path, make_temp_file):
        cache = SceneCache(cache_dir=cache_dir, max_size=1)
        scene_a_lease, evicted_scene_ids = cache.add_and_acquire_scene("scene-a", make_temp_file("scene-a"))
        assert evicted_scene_ids == []
        assert scene_a_lease.path.exists()

        for scene_id in ("scene-b", "scene-c", "scene-d"):
            newer_lease, added_evicted_scene_ids = cache.add_and_acquire_scene(
                scene_id,
                make_temp_file(scene_id),
            )
            assert added_evicted_scene_ids == []
            assert len(cache._cache) == 2
            assert newer_lease.release() == [scene_id]
            assert not newer_lease.path.exists()
            assert list(cache._cache) == ["scene-a"]

        assert scene_a_lease.release() == []
        assert scene_a_lease.path.exists()

    def test_multiple_leases_require_final_release(self, cache_dir: Path, make_temp_file):
        cache = SceneCache(cache_dir=cache_dir, max_size=1)
        first_lease, _ = cache.add_and_acquire_scene("scene-a", make_temp_file("scene-a"))
        second_lease = cache.acquire_scene("scene-a")
        scene_b_lease, evicted_scene_ids = cache.add_and_acquire_scene("scene-b", make_temp_file("scene-b"))

        assert evicted_scene_ids == []
        assert first_lease.release() == []
        assert first_lease.path.exists()
        assert second_lease.release() == ["scene-a"]
        assert not first_lease.path.exists()
        assert first_lease.release() == []
        assert scene_b_lease.release() == []

    def test_discard_only_removes_scene_without_other_leases(self, cache_dir: Path, make_temp_file):
        cache = SceneCache(cache_dir=cache_dir, max_size=2)
        only_lease, _ = cache.add_and_acquire_scene("scene-a", make_temp_file("scene-a"))

        assert only_lease.discard()
        assert not only_lease.path.exists()
        assert not cache.has_scene("scene-a")
        assert not only_lease.discard()
        assert only_lease.release() == []

        first_lease, _ = cache.add_and_acquire_scene("scene-b", make_temp_file("scene-b"))
        other_lease = cache.acquire_scene("scene-b")

        assert not first_lease.discard()
        assert other_lease.path.exists()
        assert cache.has_scene("scene-b")
        assert other_lease.release() == []

    def test_discard_after_remove_scene_does_not_raise(self, cache_dir: Path, make_temp_file):
        # `remove_scene` drops the cache entry but deliberately leaves pin counts intact so an outstanding
        # lease still releases cleanly. Discarding that surviving lease must be a no-op (the generation is
        # already gone) rather than a KeyError from popping a missing cache key.
        cache = SceneCache(cache_dir=cache_dir, max_size=2)
        lease, _ = cache.add_and_acquire_scene("scene-a", make_temp_file("scene-a"))

        assert cache.remove_scene("scene-a")
        assert not cache.has_scene("scene-a")
        assert not lease.path.exists()

        assert lease.discard() is False
        assert not cache.has_scene("scene-a")
        # The lease is now consumed; a follow-up release is likewise a clean no-op.
        assert lease.release() == []

    @pytest.mark.parametrize("acquire_lease", [False, True])
    def test_access_updates_lru_order(self, cache_dir: Path, make_temp_file, acquire_lease: bool):
        cache = SceneCache(cache_dir=cache_dir, max_size=2)
        lease_a, _ = cache.add_and_acquire_scene("scene-a", make_temp_file("scene-a"))
        lease_a.release()
        scene_a_path = lease_a.path
        lease_b, _ = cache.add_and_acquire_scene("scene-b", make_temp_file("scene-b"))
        lease_b.release()
        scene_b_path = lease_b.path

        scene_a_lease = cache.acquire_scene("scene-a") if acquire_lease else None
        if scene_a_lease is None:
            assert cache.get_scene_path("scene-a") == scene_a_path
        scene_c_lease, evicted_scene_ids = cache.add_and_acquire_scene("scene-c", make_temp_file("scene-c"))

        assert evicted_scene_ids == ["scene-b"]
        assert scene_a_path.exists()
        assert not scene_b_path.exists()
        assert scene_c_lease.path.exists()
        assert list(cache._cache) == ["scene-a", "scene-c"]

        if scene_a_lease is not None:
            assert scene_a_lease.release() == []
        assert scene_c_lease.release() == []


class TestSceneCachePolicy:
    """Cache-policy behaviors: truthful LRU on access and disk-eviction callback.

    These lock in the two properties the multiprocess drop-in relies on: (1) a cache *hit* refreshes LRU
    recency so `_enforce_max_size` evicts the genuinely least-recently-*used* entry (not merely the least
    recently *inserted*), and (2) each disk eviction fires `on_evict(scene_id)` so the drop-in can invalidate
    the coordinator's reload mapping (an ID-only request for an evicted remote scene then returns NOT_FOUND).
    """

    def test_get_scene_path_hit_refreshes_lru_order(self, cache_dir: Path, make_temp_file):
        # max_size=2: touch the older entry via get_scene_path (default update_access_time=True), then insert a
        # third — the *touched* entry survives and the untouched middle entry is evicted (truthful LRU).
        cache = SceneCache(cache_dir=cache_dir, max_size=2)
        cache.add_scene("a", make_temp_file("a"))
        cache.add_scene("b", make_temp_file("b"))

        assert cache.get_scene_path("a") == cache_dir / "a.usdz"  # a hit -> a becomes most-recent

        cache.add_scene("c", make_temp_file("c"))  # over cap -> evicts the LRU, which is now "b"

        assert cache.has_scene("a") and cache.has_scene("c")
        assert not cache.has_scene("b")  # the untouched middle entry was evicted, not the touched "a"

    def test_get_scene_path_without_touch_keeps_lru_order(self, cache_dir: Path, make_temp_file):
        # update_access_time=False must NOT reorder: the peeked entry stays the LRU victim.
        cache = SceneCache(cache_dir=cache_dir, max_size=2)
        cache.add_scene("a", make_temp_file("a"))
        cache.add_scene("b", make_temp_file("b"))

        assert cache.get_scene_path("a", update_access_time=False) == cache_dir / "a.usdz"  # peek, no reorder

        cache.add_scene("c", make_temp_file("c"))  # over cap -> "a" is still the LRU and is evicted

        assert not cache.has_scene("a")
        assert cache.has_scene("b") and cache.has_scene("c")

    def test_add_scene_fires_on_evict_with_evicted_scene_id(self, cache_dir: Path, make_temp_file):
        # max_size=1: inserting "b" evicts "a" and must fire on_evict("a") exactly once, AFTER the entry is
        # gone from the cache (so the coordinator's invalidation sees a truthful post-eviction state).
        evicted: list[SceneCacheEviction] = []

        def _seen(eviction: SceneCacheEviction) -> None:
            assert not cache.has_scene(eviction.scene_id)  # eviction is applied before the callback fires
            evicted.append(eviction)

        cache = SceneCache(cache_dir=cache_dir, max_size=1, on_evict=_seen)
        cache.add_scene("a", make_temp_file("a"))
        assert evicted == []  # first insert fits; no eviction

        cache.add_scene("b", make_temp_file("b"))
        assert [(event.scene_id, event.generation) for event in evicted] == [("a", 1)]

    def test_set_on_evict_registers_callback_after_construction(self, cache_dir: Path, make_temp_file):
        # The drop-in builds the cache before the coordinator exists, then wires the callback via set_on_evict.
        evicted: list[SceneCacheEviction] = []
        cache = SceneCache(cache_dir=cache_dir, max_size=1)
        cache.add_scene("a", make_temp_file("a"))

        cache.set_on_evict(evicted.append)  # wired only now, after the cache already holds "a"

        cache.add_scene("b", make_temp_file("b"))
        assert [event.scene_id for event in evicted] == ["a"]

    def test_on_evict_failure_is_surfaced_after_cache_mutation(self, cache_dir: Path, make_temp_file):
        # A correctness-critical callback failure is surfaced; the cache state itself remains truthful.
        def _boom(_eviction: SceneCacheEviction) -> None:
            raise RuntimeError("callback boom")

        cache = SceneCache(cache_dir=cache_dir, max_size=1, on_evict=_boom)
        cache.add_scene("a", make_temp_file("a"))
        with pytest.raises(SceneCacheEvictionError, match=r"a@1.*callback boom"):
            cache.add_scene("b", make_temp_file("b"))
        assert cache.has_scene("b") and not cache.has_scene("a")

    def test_release_is_idempotent_when_eviction_callback_raises(self, cache_dir: Path, make_temp_file):
        # A lease release whose deferred eviction fires a raising on_evict must still CONSUME the lease:
        # the pin is deleted before the callback runs, so a retry of release() must be a clean no-op
        # (return []), never a KeyError from decrementing an already-deleted pin count.
        def _boom(_eviction: SceneCacheEviction) -> None:
            raise RuntimeError("callback boom")

        cache = SceneCache(cache_dir=cache_dir, max_size=1, on_evict=_boom)
        lease_a, _ = cache.add_and_acquire_scene("scene-a", make_temp_file("scene-a"))
        lease_b, evicted = cache.add_and_acquire_scene("scene-b", make_temp_file("scene-b"))
        assert evicted == []  # both pinned -> over-budget eviction is deferred, callback not yet fired

        # Releasing scene-b makes it the unpinned over-budget victim; its eviction callback raises.
        with pytest.raises(SceneCacheEvictionError, match=r"scene-b@2.*callback boom"):
            lease_b.release()
        # The pin was consumed despite the callback failure: a retry is a no-op, not a KeyError.
        assert lease_b.release() == []
        assert "scene-b" not in cache._pin_counts
        # scene-a's pin is untouched and still releases cleanly.
        assert lease_a.release() == []

    def test_add_and_acquire_scene_failure_from_eviction_leaks_no_pin(self, cache_dir: Path, make_temp_file):
        # If add_and_acquire_scene's post-insert on_evict raises, the freshly acquired lease never reaches
        # the caller, so nothing external can release its pin. The method must release that fresh lease
        # internally before propagating, leaving NO leaked pin (which would otherwise pin the new scene
        # forever, making it unevictable).
        def _boom(_eviction: SceneCacheEviction) -> None:
            raise RuntimeError("callback boom")

        cache = SceneCache(cache_dir=cache_dir, max_size=1, on_evict=_boom)
        cache.add_scene("scene-a", make_temp_file("scene-a"))  # unpinned resident filling the single slot

        # Adding scene-b evicts the unpinned scene-a, whose on_evict raises.
        with pytest.raises(SceneCacheEvictionError, match=r"scene-a@1.*callback boom"):
            cache.add_and_acquire_scene("scene-b", make_temp_file("scene-b"))

        assert "scene-b" not in cache._pin_counts  # the fresh lease's pin was rolled back
        assert cache._pin_counts == {}  # no pin leaked anywhere

    def test_source_binding_and_generation_change_on_replacement(self, cache_dir: Path, make_temp_file):
        cache = SceneCache(cache_dir=cache_dir, max_size=2)
        first_path = cache.add_scene("s", make_temp_file("first"))
        first = cache.bind_scene_source("s", first_path, "https://one/s.usdz")

        assert cache.is_current("s", first.generation, "https://one/s.usdz")
        with pytest.raises(ValueError, match="different source"):
            cache.bind_scene_source("s", first_path, "https://two/s.usdz")

        assert cache.remove_scene("s", generation=first.generation)
        second_path = cache.add_scene("s", make_temp_file("second"))
        second = cache.bind_scene_source("s", second_path, "https://two/s.usdz")
        assert second.generation > first.generation
        assert not cache.is_current("s", first.generation, "https://one/s.usdz")
        assert cache.is_current("s", second.generation, "https://two/s.usdz")


class TestSceneCacheValue:
    """Test suite for SceneCacheValue named tuple implementation."""

    def test_scene_cache_value_named_tuple(self):
        """Test SceneCacheValue named tuple creation and attribute access."""
        path = Path("/test/path")
        timestamp = time.time()

        value = SceneCacheValue(path, timestamp)

        assert value.path == path
        assert value.timestamp == timestamp
        assert isinstance(value, tuple)  # namedtuple is still a tuple


class TestSceneCacheLock:
    """Test suite for SceneCache lock functionality."""

    def test_lock_scene_path_keeps_lock_file(self, scene_cache: SceneCache):
        """
        Simple test to check that the lock file is created and locked

        Verifies that:
        - Lock file is created
        - Lock file is locked
        - Lock file is not deleted internally
        - Lock file is unlocked when the context manager is exited
        """

        expected_lock_file = scene_cache.cache_dir / "x-test-scene-1.lock"

        with scene_cache.lock_scene_path("x-test-scene-1"):
            assert expected_lock_file.exists()

            # Try to acquire low-level lock on the same file
            with open(expected_lock_file, "w") as f:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Lock file is NOT deleted internally
        assert expected_lock_file.exists()

        # Try to acquire low-level lock on the same file again
        with open(expected_lock_file, "w") as f:
            # should not raise an error
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_sequential_lock_file(self, scene_cache: SceneCache):
        """Test that the lock file can be acquired sequentially multiple times."""
        for _ in range(10):
            with scene_cache.lock_scene_path("x-test-scene-1"):
                pass
