# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest entry point for pytest_test targets.

Responsibilities:
- Advertise Bazel sharding support (writes TEST_SHARD_STATUS_FILE).
- Honour TEST_SHARD_INDEX / TEST_TOTAL_SHARDS via a pytest plugin that selects
  the slice of collected items belonging to this shard.
- Dispatch to pytest.main() and propagate its return code.
"""

import os
import sys

import pytest


class _ShardPlugin:
    """Distribute collected test items across Bazel shards."""

    def __init__(self, shard_index: int, total_shards: int) -> None:
        self._shard_index = shard_index
        self._total_shards = total_shards

    def pytest_collection_modifyitems(self, items: list) -> None:  # type: ignore[override]
        items.sort(key=lambda item: item.nodeid)
        items[:] = [item for i, item in enumerate(items) if i % self._total_shards == self._shard_index]


def main() -> None:
    # Advertise sharding support so Bazel does not reject shard_count targets.
    shard_status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status_file:
        with open(shard_status_file, "w"):
            pass

    args = list(sys.argv[1:])
    plugins: list = []

    shard_index_str = os.environ.get("TEST_SHARD_INDEX")
    total_shards_str = os.environ.get("TEST_TOTAL_SHARDS")
    if shard_index_str is not None and total_shards_str is not None:
        shard_index = int(shard_index_str)
        total_shards = int(total_shards_str)
        if total_shards > 1:
            plugins.append(_ShardPlugin(shard_index, total_shards))

    # Two-statement form is intentional: sys.exit is evaluated after pytest.main()
    # so that monkey-patches installed during test collection (e.g. from
    # register_pytest_finalization_bypass()) are in effect when the call dispatches.
    exit_code = pytest.main(args, plugins=plugins)
    sys.exit(exit_code)


main()
