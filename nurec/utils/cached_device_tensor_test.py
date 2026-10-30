# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for ``nurec.utils.cached_device_tensor``."""

from __future__ import annotations

import pytest
import torch

from nurec.utils.cached_device_tensor import (
    BoundedCacheFullError,
    CachedDeviceTensorPool,
    bounded_cache,
)


def test_pool_keys_on_inference_mode():
    pool = CachedDeviceTensorPool(maxsize=4, name="test_inference_mode")
    values = (1.25, 0.0, 3.5)
    device = torch.device("cpu")

    with torch.inference_mode():
        primed = pool.get(values, device)
        assert primed.is_inference()

    cached = pool.get(values, device)
    assert not cached.is_inference()
    assert cached.untyped_storage().data_ptr() != primed.untyped_storage().data_ptr()


def test_pool_reuses_storage_within_the_same_mode():
    pool = CachedDeviceTensorPool(maxsize=4, name="test_reuse")
    values = (2.5, -1.0)
    device = torch.device("cpu")

    first = pool.get(values, device)
    second = pool.get(values, device)
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def test_pools_do_not_share_capacity():
    device = torch.device("cpu")
    pool_a = CachedDeviceTensorPool(maxsize=1, name="pool_a")
    pool_b = CachedDeviceTensorPool(maxsize=1, name="pool_b")

    pool_a.get((1.0,), device)
    pool_b.get((2.0,), device)
    with pytest.raises(BoundedCacheFullError, match="pool_a bounded cache capacity exceeded"):
        pool_a.get((3.0,), device)


def test_pool_overflow_appends_extra_message():
    pool = CachedDeviceTensorPool(
        maxsize=1,
        name="named_pool",
        overflow_message="Hint from the declaring site.",
    )
    device = torch.device("cpu")
    pool.get((1.0,), device)
    with pytest.raises(BoundedCacheFullError, match="Hint from the declaring site"):
        pool.get((2.0,), device)


def test_bounded_cache_error_is_generic_without_overflow_message():
    @bounded_cache(maxsize=1, name="boxed")
    def boxed(value: int) -> list[int]:
        return [value]

    boxed(1)
    with pytest.raises(
        BoundedCacheFullError,
        match=(
            r"boxed bounded cache capacity exceeded: the next insertion requires at least 2 entries, "
            r"but configured capacity is 1\. Configure at least 2 entries before retrying\.$"
        ),
    ) as caught:
        boxed(2)
    assert "sensor" not in str(caught.value).lower()


def test_bounded_cache_hits_reuse_the_factory_result():
    calls = {"n": 0}

    @bounded_cache(maxsize=2)
    def boxed(value: int) -> list[int]:
        calls["n"] += 1
        return [value]

    first = boxed(1)
    second = boxed(1)
    assert first is second
    assert calls["n"] == 1


def test_bounded_cache_raises_on_overflow_and_keeps_existing_entries():
    calls = {"n": 0}

    @bounded_cache(maxsize=2)
    def boxed(value: int) -> list[int]:
        calls["n"] += 1
        return [value]

    first = boxed(1)
    second = boxed(2)
    with pytest.raises(BoundedCacheFullError, match="requires at least 3 entries, but configured capacity is 2"):
        boxed(3)
    assert calls["n"] == 2
    assert boxed(1) is first
    assert boxed(2) is second


def test_bounded_cache_clear_allows_a_new_key_after_overflow():
    @bounded_cache(maxsize=1)
    def boxed(value: int) -> list[int]:
        return [value]

    boxed(1)
    with pytest.raises(BoundedCacheFullError):
        boxed(2)
    boxed.cache_clear()
    assert boxed(2) == [2]


def test_bounded_cache_rejects_non_positive_maxsize():
    with pytest.raises(ValueError, match="maxsize"):

        @bounded_cache(maxsize=0)
        def _fn(value: int) -> int:
            return value
