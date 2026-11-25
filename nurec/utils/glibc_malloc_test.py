# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

from unittest.mock import Mock, call

import pytest

from nurec.utils import glibc_malloc
from nurec.utils.glibc_malloc import TUNE_GLIBC_MALLOC_ENV, tune_glibc_malloc


MMAP_BYTES = 32 * 1024 * 1024
TRIM_BYTES = 256 * 1024 * 1024
MMAP = call(-3, MMAP_BYTES)
TRIM = call(-1, TRIM_BYTES)


@pytest.fixture
def mallopt(monkeypatch):
    """Pretend to be glibc on Linux and record every mallopt call instead of changing the test's allocator."""
    for name in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "GLIBC_TUNABLES", TUNE_GLIBC_MALLOC_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(glibc_malloc.platform, "libc_ver", lambda: ("glibc", "2.39"))
    libc = Mock()
    libc.mallopt.return_value = 1
    cdll = Mock(return_value=libc)
    monkeypatch.setattr(glibc_malloc.ctypes, "CDLL", cdll)
    return libc.mallopt


def test_sets_both_thresholds(mallopt, caplog):
    with caplog.at_level(logging.INFO, logger=glibc_malloc.__name__):
        tune_glibc_malloc()

    assert mallopt.call_args_list == [MMAP, TRIM]
    assert f"mmap_threshold={MMAP_BYTES}, trim_threshold={TRIM_BYTES}" in caplog.text


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"MALLOC_MMAP_THRESHOLD_": "131072"}, [TRIM]),
        ({"MALLOC_TRIM_THRESHOLD_": "0"}, [MMAP]),
        ({"GLIBC_TUNABLES": "glibc.malloc.trim_threshold=0"}, [MMAP]),
        ({"GLIBC_TUNABLES": "glibc.malloc.mmap_threshold=1048576"}, [TRIM]),
        ({"GLIBC_TUNABLES": "glibc.malloc.arena_max=2:glibc.malloc.mmap_threshold=1048576"}, [TRIM]),
        ({"MALLOC_MMAP_THRESHOLD_": "131072", "MALLOC_TRIM_THRESHOLD_": "0"}, []),
    ],
)
def test_leaves_user_set_knobs_alone(mallopt, monkeypatch, caplog, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with caplog.at_level(logging.INFO, logger=glibc_malloc.__name__):
        tune_glibc_malloc()

    assert mallopt.call_args_list == expected
    assert ("Set glibc malloc" in caplog.text) == bool(expected)


@pytest.mark.parametrize("value, expected", [("0", []), ("false", []), ("1", [MMAP, TRIM]), ("True", [MMAP, TRIM])])
def test_kill_switch(mallopt, monkeypatch, caplog, value, expected):
    monkeypatch.setenv(TUNE_GLIBC_MALLOC_ENV, value)

    with caplog.at_level(logging.INFO, logger=glibc_malloc.__name__):
        tune_glibc_malloc()

    assert mallopt.call_args_list == expected
    assert ("Set glibc malloc" in caplog.text) == bool(expected)


@pytest.mark.parametrize(
    "platform_name, libc",
    [("linux", ("", "")), ("linux", ("musl", "")), ("darwin", ("glibc", "2.39"))],
)
def test_skips_without_glibc(mallopt, monkeypatch, platform_name, libc):
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(glibc_malloc.platform, "libc_ver", lambda: libc)

    tune_glibc_malloc()

    glibc_malloc.ctypes.CDLL.assert_not_called()
    mallopt.assert_not_called()


def test_rejected_value_warns_and_continues(mallopt, caplog):
    mallopt.side_effect = [0, 1]

    with caplog.at_level(logging.INFO, logger=glibc_malloc.__name__):
        tune_glibc_malloc()

    assert mallopt.call_args_list == [MMAP, TRIM]
    assert f"mallopt rejected glibc malloc mmap_threshold={MMAP_BYTES}" in caplog.text
    assert f"Set glibc malloc trim_threshold={TRIM_BYTES}" in caplog.text


@pytest.mark.parametrize(
    "cdll_config",
    [
        pytest.param({"side_effect": OSError("cannot load libc")}, id="libc"),
        pytest.param({"return_value": Mock(spec=[])}, id="mallopt"),
    ],
)
def test_missing_mallopt_warns_and_keeps_defaults(mallopt, monkeypatch, caplog, cdll_config):
    monkeypatch.setattr(glibc_malloc.ctypes, "CDLL", Mock(**cdll_config))

    with caplog.at_level(logging.INFO, logger=glibc_malloc.__name__):
        tune_glibc_malloc()

    mallopt.assert_not_called()
    assert "glibc mallopt is unavailable" in caplog.text
    assert "Set glibc malloc" not in caplog.text


@pytest.mark.skipif(sys.platform != "linux" or glibc_malloc.platform.libc_ver()[0] != "glibc", reason="needs glibc")
def test_real_glibc_accepts_both_values(monkeypatch, caplog):
    for name in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "GLIBC_TUNABLES", TUNE_GLIBC_MALLOC_ENV):
        monkeypatch.delenv(name, raising=False)

    with caplog.at_level(logging.INFO, logger=glibc_malloc.__name__):
        tune_glibc_malloc()

    assert "rejected" not in caplog.text
    assert f"Set glibc malloc mmap_threshold={MMAP_BYTES}, trim_threshold={TRIM_BYTES}" in caplog.text
