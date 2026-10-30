# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import fcntl
import itertools
import logging
import os
import shutil
import threading
import time

from collections import OrderedDict
from pathlib import Path
from typing import Callable, NamedTuple, Optional


log = logging.getLogger(__name__)


class SceneCacheValue(NamedTuple):
    """Cache value containing scene path and timestamp."""

    path: Path
    """Path to the cached scene."""

    timestamp: float
    """Timestamp of the scene."""

    generation: int = 0
    """Monotonic identity for this particular cached file."""

    source: Optional[str] = None
    """Remote source URI whose bytes populate this entry, when known."""


class SceneCacheEviction(NamedTuple):
    """Generation-aware notification emitted after a cache entry is removed."""

    scene_id: str
    generation: int
    source: Optional[str]
    path: Path


class SceneCacheEvictionError(RuntimeError):
    """One or more correctness-critical eviction callbacks failed."""


class SceneCacheLease:
    """Pins a scene in the cache until the lease is released."""

    def __init__(self, scene_cache: SceneCache, scene_id: str, path: Path):
        self.scene_id = scene_id
        self.path = path
        self._scene_cache = scene_cache
        self._release_lock = threading.Lock()
        self._released = False

    def release(self) -> list[str]:
        """Release this lease and return scenes evicted by deferred enforcement."""
        with self._release_lock:
            if self._released:
                return []

            # The pin is consumed the instant ``_release_scene`` decrements/deletes the pin count; a
            # later ``_notify_evicted`` failure (only reachable in multiprocess, where it is fail-closed
            # anyway) must NOT leave this lease un-released. Otherwise a retry of ``release()`` would run
            # ``_release_scene`` again against an already-deleted pin and raise ``KeyError``. Mark released
            # in ``finally`` so the lease is strictly single-shot regardless of callback outcome.
            try:
                return self._scene_cache._release_scene(self.scene_id)
            finally:
                self._released = True

    def discard(self) -> bool:
        """Consume this lease and remove its scene if no other lease pins it.

        This is for rolling back a newly added scene before publishing it to other
        callers. A false result means another lease owns the scene, so its file was
        preserved. Calling ``release`` or ``discard`` again has no effect.
        """
        with self._release_lock:
            if self._released:
                return False

            # Consume the lease even if ``_discard_scene``'s post-lock ``_notify_evicted`` raises: the pin
            # is already gone, so a retry must be a no-op, not a double-consume. Mark released in ``finally``.
            try:
                return self._scene_cache._discard_scene(self.scene_id)
            finally:
                self._released = True


class SceneCache:
    """Manages an on-disk LRU cache of downloaded scenes."""

    def __init__(
        self,
        cache_dir: Path,
        max_size: int,
        on_evict: Optional[Callable[[SceneCacheEviction], None]] = None,
    ):
        """
        Initialize the scene cache.

        Args:
            cache_dir: Path to the cache directory
            max_size: Maximum number of scenes to keep in the cache
            on_evict: Optional callback invoked with a generation-aware record for each entry evicted from the
                on-disk cache (its file has been removed). The multiprocess drop-in wires this to the
                `SceneCoordinator` so a disk-cache eviction invalidates the scene's reload mapping — an
                ID-only request for it then returns ``NOT_FOUND`` instead of ``UNKNOWN`` from a vanished
                file. Fired **after** the internal lock is released, so the callback may take other
                locks without a nesting hazard.
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self._cache: OrderedDict[str, SceneCacheValue] = OrderedDict()
        self._pin_counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._on_evict = on_evict
        self._generations = itertools.count(1)

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Scene cache initialized in {self.cache_dir} with max size {self.max_size}")

    def set_on_evict(self, on_evict: Optional[Callable[[SceneCacheEviction], None]]) -> None:
        """Register (or clear) the disk-cache eviction callback after construction.

        The multiprocess drop-in builds the `SceneCache` before the `SceneCoordinator` exists, so it wires
        the invalidation callback here once the coordinator is available."""
        with self._lock:
            self._on_evict = on_evict

    def _enforce_max_size_locked(self) -> list[SceneCacheEviction]:
        """Remove unpinned oldest scenes until the cache reaches its maximum size (caller holds ``self._lock``).

        Leased (pinned) scenes are never evicted, so the cache may transiently exceed its maximum size until
        their leases are released. Returns generation-aware eviction records so the caller can fire
        ``on_evict`` **after** releasing the lock (the callback may take other locks, e.g. the coordinator's)."""
        evicted: list[SceneCacheEviction] = []
        while len(self._cache) > self.max_size:
            scene_id = next(
                (candidate for candidate in self._cache if self._pin_counts.get(candidate, 0) == 0),
                None,
            )
            if scene_id is None:
                break

            value = self._cache.pop(scene_id)
            try:
                os.remove(value.path)
                log.info(f"Removed {scene_id} from cache to enforce max size")
            except OSError as e:
                log.error(f"Failed to remove {value.path}: {e}")
            evicted.append(SceneCacheEviction(scene_id, value.generation, value.source, value.path))
        return evicted

    def _notify_evicted(self, evictions: list[SceneCacheEviction]) -> None:
        """Fire every eviction callback and surface failures after all notifications were attempted."""
        if not evictions:
            return
        with self._lock:
            on_evict = self._on_evict
        if on_evict is None:
            return
        failures: list[tuple[SceneCacheEviction, Exception]] = []
        for eviction in evictions:
            try:
                on_evict(eviction)
            except Exception as exc:  # noqa: BLE001 - notify every victim, then fail closed
                log.exception(
                    f"on_evict callback failed for scene {eviction.scene_id} generation {eviction.generation}"
                )
                failures.append((eviction, exc))
        if failures:
            details = ", ".join(f"{eviction.scene_id}@{eviction.generation}: {exc!r}" for eviction, exc in failures)
            raise SceneCacheEvictionError(f"scene-cache eviction callback failure(s): {details}")

    def _acquire_scene_locked(self, scene_id: str) -> SceneCacheLease:
        if scene_id not in self._cache:
            raise KeyError(f"Scene {scene_id} not in cache")

        value = self._cache[scene_id]
        self._cache[scene_id] = SceneCacheValue(value.path, time.time(), value.generation, value.source)
        self._cache.move_to_end(scene_id)
        self._pin_counts[scene_id] = self._pin_counts.get(scene_id, 0) + 1
        return SceneCacheLease(self, scene_id, value.path)

    def _release_scene(self, scene_id: str) -> list[str]:
        with self._lock:
            pin_count = self._pin_counts[scene_id]
            if pin_count > 1:
                self._pin_counts[scene_id] = pin_count - 1
                return []

            del self._pin_counts[scene_id]
            evicted = self._enforce_max_size_locked()
        # Deferred enforcement is still an eviction: publish it after releasing the lock so the multiprocess
        # coordinator invalidates the reload mapping for lease-triggered victims too.
        self._notify_evicted(evicted)
        return [eviction.scene_id for eviction in evicted]

    def _discard_scene(self, scene_id: str) -> bool:
        with self._lock:
            pin_count = self._pin_counts[scene_id]
            if pin_count > 1:
                self._pin_counts[scene_id] = pin_count - 1
                return False

            del self._pin_counts[scene_id]
            value = self._cache.pop(scene_id, None)
            if value is None:
                # ``remove_scene`` already evicted this generation: it drops the cache entry but leaves the
                # pin counts intact so outstanding leases still release cleanly. There is nothing left to
                # delete or to notify about, so this discard is a no-op rather than a KeyError.
                return False
            try:
                os.remove(value.path)
            except OSError as e:
                log.error(f"Failed to remove {value.path}: {e}")
            eviction = SceneCacheEviction(scene_id, value.generation, value.source, value.path)
        self._notify_evicted([eviction])
        return True

    def has_scene(self, scene_id: str) -> bool:
        """Check if a scene is in the cache."""
        with self._lock:
            return scene_id in self._cache

    def get_scene_value(self, scene_id: str, update_access_time: bool = True) -> SceneCacheValue:
        """Return the generation/source-bearing cache value for ``scene_id``."""
        with self._lock:
            if scene_id not in self._cache:
                raise KeyError(f"Scene {scene_id} not in cache")

            value = self._cache[scene_id]
            if update_access_time:
                value = SceneCacheValue(value.path, time.time(), value.generation, value.source)
                self._cache[scene_id] = value
                self._cache.move_to_end(scene_id)
            return value

    def get_scene_path(self, scene_id: str, update_access_time: bool = True) -> Path:
        """Get the path to a cached scene with proper locking."""
        return self.get_scene_value(scene_id, update_access_time).path

    def bind_scene_source(self, scene_id: str, path: Path, source: str) -> SceneCacheValue:
        """Bind a freshly resolved cache generation to its source URI.

        A generation already bound to another URI can never be reused for ``source``. The returned value is
        the token the registrar stores in the coordinator and later revalidates before publishing.
        """
        with self._lock:
            value = self._cache.get(scene_id)
            if value is None or value.path != path:
                raise KeyError(f"Scene {scene_id} cache generation changed before source binding")
            if value.source is not None and value.source != source:
                raise ValueError(
                    f"scene {scene_id!r} cached from a different source (existing: {value.source}, requested: {source})"
                )
            bound = SceneCacheValue(value.path, value.timestamp, value.generation, source)
            self._cache[scene_id] = bound
            return bound

    def is_current(self, scene_id: str, generation: int, source: str) -> bool:
        """Whether the exact source-bound cache generation is still usable."""
        with self._lock:
            value = self._cache.get(scene_id)
            return (
                value is not None and value.generation == generation and value.source == source and value.path.exists()
            )

    def remove_scene(self, scene_id: str, *, generation: Optional[int] = None) -> bool:
        """Remove one exact generation and publish its eviction after releasing the cache lock.

        Removes unconditionally, ignoring pin counts: the caller (the multiprocess registrar) has already
        determined the generation is unusable (wrong source or vanished file). Outstanding pin counts are
        left intact so their leases still release cleanly."""
        with self._lock:
            value = self._cache.get(scene_id)
            if value is None or (generation is not None and value.generation != generation):
                return False
            self._cache.pop(scene_id)
            try:
                os.remove(value.path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.error(f"Failed to remove {value.path}: {exc}")
            eviction = SceneCacheEviction(scene_id, value.generation, value.source, value.path)
        self._notify_evicted([eviction])
        return True

    def acquire_scene(self, scene_id: str) -> SceneCacheLease:
        """Acquire a lease for a cached scene and update its LRU position."""
        with self._lock:
            return self._acquire_scene_locked(scene_id)

    def add_and_acquire_scene(self, scene_id: str, temp_path: Path) -> tuple[SceneCacheLease, list[str]]:
        """Atomically add and acquire a scene, returning any scenes evicted by insertion."""
        with self._lock:
            if scene_id in self._cache:
                raise ValueError(f"Scene {scene_id} already in cache")

            dest_path = self.cache_dir / f"{scene_id}.usdz"
            shutil.move(temp_path, dest_path)
            self._cache[scene_id] = SceneCacheValue(dest_path, time.time(), next(self._generations))
            lease = self._acquire_scene_locked(scene_id)
            evicted = self._enforce_max_size_locked()

        # Fire eviction callbacks after releasing the lock (the callback may take the coordinator lock).
        # If notification fails, the freshly acquired lease never reaches the caller, so nothing would ever
        # release its pin — the just-added scene would be pinned (unevictable) forever. Release it here so a
        # failed add leaks no pin, suppressing any secondary failure from that rollback so the original
        # eviction error propagates.
        try:
            self._notify_evicted(evicted)
        except Exception:
            with contextlib.suppress(Exception):
                lease.release()
            raise
        return lease, [eviction.scene_id for eviction in evicted]

    def add_scene(self, scene_id: str, temp_path: Path) -> Path:
        """
        Add a scene to the cache with proper locking.
        Adding a new scene will raise an error if the scene is already in the cache.
        """
        # Check again under lock if the scene was added while we were downloading
        with self._lock:
            if scene_id in self._cache:
                raise ValueError(f"Scene {scene_id} already in cache")

            # Generate destination path
            dest_path = self.cache_dir / f"{scene_id}.usdz"

            # Move the file to the cache directory
            shutil.move(temp_path, dest_path)

            # Update LRU with current timestamp
            now = time.time()
            self._cache[scene_id] = SceneCacheValue(dest_path, now, next(self._generations))

            # Enforce max size
            evicted = self._enforce_max_size_locked()

        # Fire eviction callbacks after releasing the lock (the callback may take the coordinator lock).
        self._notify_evicted(evicted)
        return dest_path

    @contextlib.contextmanager
    def lock_scene_path(self, scene_id: str):
        """
        Context manager to lock the downloading of a scene_id to avoid concurrent downloads.

        This is a simplified version of the filelock library.

        - Open file for writing
        - Lock the file
        - Write the pid of the current process to the file
        - Yield the lock file to do the download under the lock
        - Unlock the file
        - Close the file

        Do not remove the lockfile:
        * https://github.com/tox-dev/py-filelock/issues/31
        * https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
        """

        assert self.cache_dir is not None
        assert self.cache_dir.exists()

        lock_file = self.cache_dir / f"{scene_id}.lock"

        with open(lock_file, "w") as f:
            log.info(f"Locking scene {scene_id} for download")

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(f"{os.getpid()}\r\n")
            yield lock_file
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            log.info(f"Unlocked scene {scene_id} for download")
