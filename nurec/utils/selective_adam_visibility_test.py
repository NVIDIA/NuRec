# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nurec.utils.selective_adam_visibility import prepare_async_visibility_union


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
