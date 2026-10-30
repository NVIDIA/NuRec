# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded memoization that fails out loud instead of evicting.

``bounded_cache`` memoizes a function up to ``maxsize`` keys. A miss on a full
cache raises ``BoundedCacheFullError`` instead of dropping an entry.

``CachedDeviceTensorPool`` is the object callers declare for session-constant
float tensors. Construct one pool per population (with its own ``maxsize`` and
optional overflow hint) and allocate through ``get``. There is no process-wide
tensor cache: occupancy is tracked on the pool you hold.

Returned tensors are **shared for a given** ``(values, device, inference_mode)``
— treat them as read-only. Any in-place mutation (``.add_``, ``.zero_``,
``.copy_``, autograd through a mutated alias) corrupts every subsequent cache
hit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

import torch


P = ParamSpec("P")
R = TypeVar("R")


class BoundedCacheFullError(RuntimeError):
    """Raised when a bounded cache would have to evict to store a new key."""


class _BoundedCache(Generic[P, R]):
    """Callable memoizer: hits reuse the stored result; overflow raises."""

    def __init__(
        self,
        fn: Callable[P, R],
        *,
        maxsize: int,
        name: str,
        overflow_message: str | None,
    ) -> None:
        self._fn = fn
        self._maxsize = maxsize
        self._name = name
        self._overflow_message = overflow_message
        self._store: dict[object, R] = {}

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key: object = args if not kwargs else (args, frozenset(kwargs.items()))
        if key in self._store:
            return self._store[key]
        if len(self._store) >= self._maxsize:
            required_entries = len(self._store) + 1
            message = (
                f"{self._name} bounded cache capacity exceeded: the next insertion requires at least "
                f"{required_entries} entries, but configured capacity is {self._maxsize}. "
                f"Configure at least {required_entries} entries before retrying."
            )
            if self._overflow_message:
                message = f"{message} {self._overflow_message}"
            raise BoundedCacheFullError(message)
        result = self._fn(*args, **kwargs)
        self._store[key] = result
        return result

    def cache_clear(self) -> None:
        self._store.clear()


def bounded_cache(
    maxsize: int,
    *,
    name: str | None = None,
    overflow_message: str | None = None,
) -> Callable[[Callable[P, R]], _BoundedCache[P, R]]:
    """Memoize a function up to ``maxsize`` keys; a miss on a full cache raises.

    ``name`` labels the cache in ``BoundedCacheFullError`` (defaults to the
    wrapped function's ``__qualname__``). The error reports the minimum capacity
    needed for the rejected miss, not a prediction of future occupancy.
    ``overflow_message``, if given, is appended so the declaring site can
    explain occupancy for that pool.

    The wrapped callable exposes ``cache_clear``.
    """
    if maxsize < 1:
        raise ValueError(f"bounded_cache maxsize must be >= 1, got {maxsize}")

    def decorator(fn: Callable[P, R]) -> _BoundedCache[P, R]:
        return _BoundedCache(
            fn,
            maxsize=maxsize,
            name=name or fn.__qualname__,
            overflow_message=overflow_message,
        )

    return decorator


class CachedDeviceTensorPool:
    """Named, bounded cache of session-constant 1-D float32 tensors.

    Declare one pool per population so ``maxsize`` is local and overflow names
    this object. ``get`` keys on ``(values, device, inference_mode)`` because a
    tensor created under ``torch.inference_mode()`` cannot be used for backward.
    """

    def __init__(
        self,
        maxsize: int,
        *,
        name: str,
        overflow_message: str | None = None,
    ) -> None:
        self.maxsize = maxsize
        self.name = name
        self.overflow_message = overflow_message

        def allocate(values: tuple[float, ...], device: torch.device, _inference_mode: bool) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=device)

        self._allocate = bounded_cache(maxsize=maxsize, name=name, overflow_message=overflow_message)(allocate)

    def get(self, values: tuple[float, ...], device: torch.device) -> torch.Tensor:
        """Return a cached 1-D float32 tensor on ``device``. Treat it as read-only."""
        return self._allocate(values, device, torch.is_inference_mode_enabled())

    def cache_clear(self) -> None:
        self._allocate.cache_clear()
