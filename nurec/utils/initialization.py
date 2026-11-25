# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources shared only during one scene initialization, never between runs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np
import torch


_active: ContextVar[InitializationResources | None] = ContextVar("initialization_resources", default=None)
# Torch's CPU thread count is process-wide, so only one open scope may own it.
_torch_thread_budget = Lock()


def current_initialization() -> InitializationResources | None:
    resources = _active.get()
    return resources if resources is not None and not resources.closed else None


def close_current_initialization() -> None:
    """Close the active scope, if any, re-raising worker failures."""
    if (resources := current_initialization()) is not None:
        resources.close()


class InitializationResources:
    """Run startup preparation jobs on background threads, scoped to one initialization.

    - Jobs must not mutate the model or draw from global torch/NumPy random state; a job-local generator is fine.
    - While open, the scope exclusively owns the process-wide Torch CPU thread count:
      it caps the count at 2, close() restores it, and opening a second scope raises RuntimeError.
    - Workers see the active scope through context propagation, so one PLY read is shared without a process-global cache.
    """

    def __init__(self) -> None:
        self.closed = False
        self._closing = False
        self._lock = Lock()
        self._pools: dict[str, ThreadPoolExecutor] = {}
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._jobs: dict[Any, Future] = {}
        self.on_datasource_loading: list[Callable] = []
        self.on_datasource_ready: list[Callable] = []
        self.prepared_initializers: dict[str, Any] = {}
        self._previous_threads: int | None = None
        self._owns_thread_budget = False

    def __enter__(self) -> InitializationResources:
        if not _torch_thread_budget.acquire(blocking=False):
            raise RuntimeError("Another initialization scope already owns the Torch thread budget")
        self._owns_thread_budget = True
        self._token = _active.set(self)
        self._previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(self._previous_threads, 2))
        return self

    def submit(self, key: Any, pool: str, fn: Callable, *args, **kwargs) -> Future:
        """Run ``fn`` once per ``key`` and return its future.

        - A repeated ``key`` returns the first future, even a failed one.
        - A repeated ``key`` ignores its new ``fn`` and arguments.
        - Each ``pool`` has one worker, so its jobs run in submission order.
        - While closing, a new ``key`` raises RuntimeError; a known one still returns its future.
        - After close() completes, every call raises RuntimeError.
        """
        with self._lock:
            if self.closed:
                raise RuntimeError("Initialization resources have already been released")
            if key not in self._jobs:
                if self._closing:
                    raise RuntimeError("Cannot start new preparation while initialization is closing")
                if pool not in self._pools:
                    self._pools[pool] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=pool)
                executor = self._pools[pool]
                self._jobs[key] = executor.submit(copy_context().run, fn, *args, **kwargs)
            return self._jobs[key]

    def executor(self, name: str, max_workers: int) -> ThreadPoolExecutor:
        """Return the scope's ``name`` thread pool, created on first use; close() shuts it down."""
        with self._lock:
            if self.closed or self._closing:
                raise RuntimeError("Initialization resources are closing or already released")
            if name not in self._executors:
                self._executors[name] = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=name)
            return self._executors[name]

    def get(self, key: Any) -> Future | None:
        """Return the future submitted under ``key``, or None if there is none."""
        return self._jobs.get(key)

    def close(self) -> None:
        if self.closed:
            return
        with self._lock:
            self._closing = True
        try:
            # Drain consumers before releasing shared CPU/CUDA storage. Surface
            # worker failures even if a model layer never consumed its future.
            for pool in [*self._executors.values(), *reversed(list(self._pools.values()))]:
                pool.shutdown(wait=True)
            for future in self._jobs.values():
                future.result()
        finally:
            self._jobs.clear()
            self._pools.clear()
            self._executors.clear()
            self.on_datasource_loading.clear()
            self.on_datasource_ready.clear()
            self.prepared_initializers.clear()
            self.closed = True
            if self._previous_threads is not None:
                torch.set_num_threads(self._previous_threads)
            if self._owns_thread_budget:
                self._owns_thread_budget = False
                _torch_thread_budget.release()

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            # The caller's exception stays primary; keep the worker failure visible on it.
            exc.add_note(f"Initialization worker cleanup also failed: {cleanup_error!r}")
        finally:
            _active.reset(self._token)


def prefetch_initialization_ply(resources: InitializationResources, path: str) -> Future:
    return resources.submit(("ply", str(Path(path).resolve())), "init-ply", _read_fused_ply, path)


def load_initialization_ply(path: str):
    """Read an immutable fused point cloud, sharing it only inside the active scope."""
    resources = current_initialization()
    return prefetch_initialization_ply(resources, path).result() if resources is not None else _read_fused_ply(path)


def _read_fused_ply(path: str):
    """Bulk-read the fixed vertex layout emitted by fusion; delegate other PLYs."""
    import point_cloud_utils as pcu

    properties = [
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property int segment_id",
    ]
    record = np.dtype([("xyz", "<f4", (3,)), ("normal", "<f4", (3,)), ("rgb", "u1", (3,)), ("segment_id", "<i4")])
    with open(path, "rb") as stream:
        header = []
        for _ in range(100):
            try:
                line = stream.readline(4096).decode("ascii").strip()
            except UnicodeDecodeError:
                break
            header.append(line)
            if line == "end_header":
                break
        elements = [line for line in header if line.startswith("element ")]
        supported = (
            header
            and header[0] == "ply"
            and header[-1] == "end_header"
            and "format binary_little_endian 1.0" in header
            and len(elements) == 1
            and elements[0].startswith("element vertex ")
            and [line for line in header if line.startswith("property ")] == properties
        )
        if not supported:
            return pcu.load_triangle_mesh(path)
        try:
            count = int(elements[0].split()[-1])
        except ValueError:
            return pcu.load_triangle_mesh(path)
        offset = stream.tell()
        if count < 0 or Path(path).stat().st_size != offset + count * record.itemsize:
            return pcu.load_triangle_mesh(path)
        records = np.fromfile(stream, dtype=record, count=count)
    if len(records) != count:
        raise OSError(f"Truncated point cloud: {path}")
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = records["xyz"].copy()
    mesh.vertex_data.normals = records["normal"].copy()
    mesh.vertex_data.colors = np.full((count, 4), 255, dtype=np.uint8)
    mesh.vertex_data.colors[:, :3] = records["rgb"]
    mesh.vertex_data.custom_attributes["segment_id"] = records["segment_id"].copy()
    return mesh
