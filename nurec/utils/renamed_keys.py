# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readers for on-disk keys whose spelling changed in the nre -> nurec rename."""

import operator

from collections.abc import Callable, Mapping
from typing import Any, TypeVar


T = TypeVar("T")


def normalize_renamed_key(
    mapping: Mapping[str, T], new_key: str, old_key: str, *, equal: Callable[[T, T], bool] = operator.eq
) -> dict[str, T]:
    """Return a copy of `mapping` that carries the value under `new_key` only.

    Accepts the old or the new spelling. If both are present they must compare
    equal under `equal`; conflicting values are rejected so an ambiguous file
    never silently selects one binding.

    Raises:
        ValueError: If both keys are present with values that are not equal.
    """
    out = dict(mapping)
    if old_key not in out:
        return out
    old_value = out.pop(old_key)
    if new_key in out:
        if not equal(out[new_key], old_value):
            raise ValueError(f"conflicting values for {new_key!r} and its legacy spelling {old_key!r}")
        return out
    out[new_key] = old_value
    return out


def adopt_renamed_key(
    mapping: dict[str, T], new_key: str, old_key: str, *, equal: Callable[[T, T], bool] = operator.eq
) -> dict[str, T]:
    """Move `old_key` onto `new_key` inside `mapping`.

    Same conflict rule as `normalize_renamed_key`. Returns `mapping`.
    """
    normalized = normalize_renamed_key(mapping, new_key, old_key, equal=equal)
    mapping.clear()
    mapping.update(normalized)
    return mapping


def normalize_renamed_value(value: Any, new_value: str, old_value: str) -> Any:
    """Map the legacy spelling `old_value` to `new_value`; every other value passes through."""
    return new_value if value == old_value else value
