# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from nurec.utils.selective_adam_visibility import prepare_async_visibility_union
from nurec.utils.trainer import BroadcastExceptions


_COLLECTIVE_TIMEOUT_SECONDS = 20
# Include cold Lightning imports before Gloo initialization.
_WORKER_TIMEOUT_SECONDS = 120


class _FakeWork:
    def __init__(self, reduction: torch.Tensor, remote: torch.Tensor) -> None:
        self.reduction = reduction
        self.remote = remote
        self.waited = False

    def wait(self) -> None:
        self.waited = True
        torch.maximum(self.reduction, self.remote, out=self.reduction)


def test_local_camera_lidar_union_compacts_without_mutating_sources() -> None:
    camera = torch.tensor([0.0, 1.0, 0.0, 0.0])
    lidar = torch.tensor([0.0, 0.0, 0.5, 0.0])
    camera_before = camera.clone()
    lidar_before = lidar.clone()

    pending = prepare_async_visibility_union(
        [camera, lidar],
        element_count=4,
        device=torch.device("cpu"),
        local_ranges=((1, 2),),
        positive_only=False,
    )

    pending.launch()
    assert pending._work is None
    torch.testing.assert_close(pending.resolve(), torch.tensor([True, True]))
    torch.testing.assert_close(camera, camera_before)
    torch.testing.assert_close(lidar, lidar_before)


def test_missing_or_malformed_source_resolves_to_dense_fallback() -> None:
    for sources in (
        [],
        [torch.zeros(2)],
    ):
        pending = prepare_async_visibility_union(
            sources,
            element_count=3,
            device=torch.device("cpu"),
            local_ranges=((0, 1), (2, 1)),
            positive_only=False,
        )
        assert pending.local_source_missing
        pending.launch()
        torch.testing.assert_close(pending.resolve(), torch.ones(2, dtype=torch.bool))


def test_resolve_preserves_multi_range_order() -> None:
    pending = prepare_async_visibility_union(
        [torch.tensor([1, 0, 0])],
        element_count=3,
        device=torch.device("cpu"),
        local_ranges=((0, 1), (2, 1)),
        positive_only=False,
    )

    pending.launch()
    torch.testing.assert_close(pending.resolve(), torch.tensor([True, False]))


def test_remote_dense_fallback_shares_the_visibility_async_max() -> None:
    pending = prepare_async_visibility_union(
        [torch.tensor([0.0, 1.0, 0.0])],
        element_count=3,
        device=torch.device("cpu"),
        local_ranges=((0, 1), (2, 1)),
        positive_only=False,
    )
    work = _FakeWork(pending.reduction, torch.ones(3, dtype=torch.uint8))
    pending._work = work
    pending.launch()
    assert not work.waited
    # A remote rank without a usable source contributes an all-visible mask,
    # including an owner row that the local rank did not mark visible.
    torch.testing.assert_close(pending.resolve(), torch.tensor([True, True]))
    assert work.waited


def test_launch_and_resolve_state_machine_fails_closed() -> None:
    pending = prepare_async_visibility_union(
        [torch.ones(1)],
        element_count=1,
        device=torch.device("cpu"),
        local_ranges=None,
        positive_only=False,
    )

    with pytest.raises(RuntimeError, match="must be launched before it is resolved"):
        pending.resolve()

    pending.launch()
    with pytest.raises(RuntimeError, match="launched more than once"):
        pending.launch()

    first = pending.resolve()
    assert pending.resolve() is first


@pytest.mark.parametrize(
    "sources, element_count",
    [([], 0), ([], 3), ([torch.zeros(2)], 0), ([torch.ones(3), torch.zeros(2)], 3)],
)
def test_strict_missing_or_malformed_source_fails_before_launch(sources, element_count) -> None:
    pending = prepare_async_visibility_union(
        sources,
        element_count=element_count,
        device=torch.device("cpu"),
        local_ranges=None,
        positive_only=False,
    )
    with pytest.raises(RuntimeError, match="complete global per-Gaussian signal") as error:
        pending.require_complete_sources()
    message = str(error.value)
    assert ("no visibility sources" if not sources else "size mismatch") in message
    assert f"element_count={element_count}" in message
    assert f"source_numels={[source.numel() for source in sources]}" in message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="multi-rank visibility launch requires CUDA")
def test_strict_union_launches_only_visibility_collective(monkeypatch) -> None:
    source = torch.tensor([0, 1, 0], dtype=torch.bool, device="cuda")
    calls = []

    def all_reduce(value: torch.Tensor, *, op, async_op: bool):
        assert async_op and op == torch.distributed.ReduceOp.MAX
        calls.append(value)
        return _FakeWork(value, torch.ones_like(value))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: torch.distributed.Backend.NCCL)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        pending = prepare_async_visibility_union(
            [source],
            element_count=3,
            device=torch.device("cuda"),
            local_ranges=None,
            positive_only=False,
        )
        pending.require_complete_sources()
        assert not calls
        pending.launch()
        visibility = pending.resolve()
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    assert len(calls) == 1 and calls[0] is pending.reduction
    torch.testing.assert_close(visibility, torch.ones(3, device="cuda", dtype=torch.bool))


def test_empty_global_and_owner_counts_skip_collective_when_dist_is_initialized(monkeypatch) -> None:
    pending = prepare_async_visibility_union(
        [torch.empty(0)],
        element_count=0,
        device=torch.device("cpu"),
        local_ranges=(),
        positive_only=True,
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "get_backend",
        lambda: pytest.fail("empty visibility union queried the distributed backend"),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("empty visibility union launched a collective"),
    )

    pending.launch()
    assert pending._work is None
    result = pending.resolve()
    assert result.shape == (0,)
    assert result.dtype == torch.bool


def _strict_source_failure_worker(rank: int, init_file: str) -> None:
    """Exercise a component-only coverage collective before strict validation."""
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=_COLLECTIVE_TIMEOUT_SECONDS),
    )
    try:

        def reduce_boolean_decision(decision: bool, all: bool) -> bool:
            assert not all
            flag = torch.tensor(int(decision))
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            return bool(flag.item())

        for failing_rank in (0, 1):
            trainer = SimpleNamespace(
                strategy=SimpleNamespace(reduce_boolean_decision=reduce_boolean_decision),
                global_rank=rank,
                should_stop=False,
            )
            message = "complete global per-Gaussian signal" if rank == failing_rank else "exception in another rank"
            with pytest.raises(RuntimeError, match=message):
                with BroadcastExceptions(cast(Any, trainer)):
                    pending = prepare_async_visibility_union(
                        [] if rank == failing_rank else [torch.ones(1)],
                        element_count=1,
                        device=torch.device("cpu"),
                        local_ranges=None,
                        positive_only=False,
                    )
                    # The 16th component-only probe reduces its coverage count
                    # before the first forward-error gate exits.
                    coverage = torch.tensor([int(rank == 0)])
                    torch.distributed.all_reduce(coverage, op=torch.distributed.ReduceOp.SUM)
                    assert coverage.item() == 1
                    pending.require_complete_sources()
                pending.launch()
                pytest.fail("forward-error gate allowed visibility launch/backward after a rank failed")
            assert trainer.should_stop
    finally:
        torch.distributed.destroy_process_group()


def test_strict_missing_source_after_component_only_probe_aborts_both_ranks(tmp_path: Path) -> None:
    def run_worker(rank: int) -> None:
        result = subprocess.run(
            [sys.executable, __file__, str(rank), str(tmp_path / "visibility-init")],
            env=os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            timeout=_WORKER_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(run_worker, range(2)))


if __name__ == "__main__":
    _strict_source_failure_worker(int(sys.argv[1]), sys.argv[2])
