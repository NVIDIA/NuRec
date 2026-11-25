# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Event

import numpy as np
import point_cloud_utils as pcu
import pytest
import torch

import nurec.utils.initialization as initialization


def test_shares_one_read_and_releases_between_initializations(monkeypatch, tmp_path):
    reads = []

    def read(path):
        reads.append(path)
        return object()

    monkeypatch.setattr(initialization, "_read_fused_ply", read)
    path = str(tmp_path / "cloud.ply")
    previous_threads = torch.get_num_threads()
    with initialization.InitializationResources() as resources:
        with ThreadPoolExecutor(max_workers=4) as callers:
            futures = list(callers.map(partial(initialization.prefetch_initialization_ply, resources), [path] * 8))
        first = futures[0].result()
        assert all(future.result() is first for future in futures)
        # Context follows scheduled workers, including nested PLY consumers.
        nested = resources.submit("consumer", "consumer", initialization.load_initialization_ply, path)
        assert nested.result() is first
        assert len(reads) == 1
    assert resources.closed and not resources._jobs and not resources._pools
    assert initialization.current_initialization() is None
    assert torch.get_num_threads() == previous_threads
    with initialization.InitializationResources():
        assert initialization.load_initialization_ply(path) is not first
    assert len(reads) == 2


def test_worker_failure_is_propagated_and_resources_restored():
    previous_threads = torch.get_num_threads()
    finished = Event()

    def fail():
        finished.set()
        raise ValueError("decode failed")

    with pytest.raises(ValueError, match="decode failed"):
        with initialization.InitializationResources() as resources:
            resources.submit("failure", "worker", fail)
    assert finished.is_set() and resources.closed
    assert torch.get_num_threads() == previous_threads
    assert initialization.current_initialization() is None
    resources.close()  # Idempotent after failure.


def test_scope_owns_its_thread_pools():
    with initialization.InitializationResources() as resources:
        pool = resources.executor("workers", 2)
        assert resources.executor("workers", 2) is pool
        assert list(pool.map(abs, [-2, 1])) == [2, 1]
    with pytest.raises(RuntimeError, match="after shutdown"):
        pool.submit(abs, 1)
    with pytest.raises(RuntimeError, match="already released"):
        resources.executor("workers", 2)


def test_only_one_scope_owns_the_torch_thread_budget():
    with initialization.InitializationResources() as first:
        with pytest.raises(RuntimeError, match="already owns the Torch thread budget"):
            with initialization.InitializationResources():
                pass
        assert not first.closed
    # Closing the owner frees the budget for the next scope.
    with initialization.InitializationResources() as second:
        assert not second.closed


def test_close_current_initialization_closes_only_the_active_scope():
    initialization.close_current_initialization()

    def fail():
        raise ValueError("worker")

    with initialization.InitializationResources() as resources:
        resources.submit("failure", "worker", fail)
        with pytest.raises(ValueError, match="worker"):
            initialization.close_current_initialization()
        assert resources.closed
        assert initialization.current_initialization() is None
        initialization.close_current_initialization()


def test_cleanup_preserves_original_caller_exception():
    def fail():
        raise ValueError("worker")

    with pytest.raises(RuntimeError, match="caller") as caught:
        with initialization.InitializationResources() as resources:
            resources.submit("failure", "worker", fail)
            raise RuntimeError("caller")
    assert resources.closed
    assert any("ValueError('worker')" in note for note in getattr(caught.value, "__notes__", []))


def test_bulk_ply_matches_general_reader(tmp_path):
    path = tmp_path / "fused.ply"
    header = """ply
format binary_little_endian 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property int segment_id
end_header
"""
    dtype = np.dtype([("xyz", "<f4", (3,)), ("normal", "<f4", (3,)), ("rgb", "u1", (3,)), ("segment_id", "<i4")])
    values = np.zeros(3, dtype=dtype)
    values["xyz"] = [[1.25, -2.5, 3.75], [0, 0, 0], [1e-8, 1e6, -12.5]]
    values["normal"] = [0, 0, 1]
    values["rgb"] = [[1, 2, 3], [0, 255, 7], [128, 42, 13]]
    values["segment_id"] = [1, 255, -1]
    path.write_bytes(header.encode("ascii") + values.tobytes())
    expected = pcu.load_triangle_mesh(str(path)).vertex_data
    actual = initialization.load_initialization_ply(str(path)).vertex_data
    for field in ("positions", "normals"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field), strict=True)
    # The file contains RGB only. PCU leaves its synthetic fourth channel
    # uninitialized; all three fused-cloud consumers explicitly use [:, :3].
    np.testing.assert_array_equal(actual.colors[:, :3], expected.colors[:, :3], strict=True)
    np.testing.assert_array_equal(actual.colors[:, 3], np.full(3, 255, dtype=np.uint8), strict=True)
    np.testing.assert_array_equal(
        actual.custom_attributes["segment_id"], expected.custom_attributes["segment_id"], strict=True
    )


def test_other_ply_layout_uses_general_reader(tmp_path, monkeypatch):
    path = tmp_path / "ascii.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    expected = object()
    calls = []
    monkeypatch.setattr(pcu, "load_triangle_mesh", lambda filename: calls.append(filename) or expected)
    assert initialization.load_initialization_ply(str(path)) is expected
    assert calls == [str(path)]
