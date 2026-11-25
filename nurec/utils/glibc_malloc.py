# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only glibc malloc tuning, applied by the training entrypoints before they start allocating."""

import ctypes
import logging
import os
import platform
import sys


# mallopt parameter ids from glibc's malloc.h.
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3

_MMAP_THRESHOLD_BYTES = 32 * 1024 * 1024
_TRIM_THRESHOLD_BYTES = 256 * 1024 * 1024

TUNE_GLIBC_MALLOC_ENV = "NUREC_TUNE_GLIBC_MALLOC"


def tune_glibc_malloc() -> None:
    """Keep large host buffers on the heap so repeated decodes reuse pages instead of faulting in fresh ones.

    It mainly helps the vehicle-light layers in model initialization, whose per-frame loop is sensitive to page faults.
    glibc gives freed memory back to the kernel in two ways.
    A buffer above the mmap threshold gets its own mapping, which is unmapped when the buffer is freed.
    Free space at the top of the heap is trimmed once it grows past the trim threshold.
    By default glibc adjusts both thresholds on the fly, so per-frame buffers can keep going back to the kernel.
    The next frame then page-faults on every page it touches, which slows initialization by a varying amount per run.
    A fixed 32 MiB mmap threshold keeps buffers smaller than 32 MiB on the heap.
    A 256 MiB trim threshold stops the heap from being trimmed between frames.
    A threshold already set through MALLOC_*_ or GLIBC_TUNABLES is left alone, so the user's value wins.
    NUREC_TUNE_GLIBC_MALLOC set to anything but 1 or true (such as 0 or false) keeps glibc's defaults.
    """
    if os.environ.get(TUNE_GLIBC_MALLOC_ENV, "1").lower() not in ("1", "true"):
        return
    if sys.platform != "linux" or platform.libc_ver()[0] != "glibc":
        return
    logger = logging.getLogger(__name__)
    try:
        mallopt = ctypes.CDLL(None).mallopt
    except (OSError, AttributeError) as error:
        logger.warning("glibc mallopt is unavailable, keeping the default malloc settings: %s", error)
        return
    mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
    mallopt.restype = ctypes.c_int
    applied = []
    for name, param, value in (
        ("mmap_threshold", _M_MMAP_THRESHOLD, _MMAP_THRESHOLD_BYTES),
        ("trim_threshold", _M_TRIM_THRESHOLD, _TRIM_THRESHOLD_BYTES),
    ):
        if f"MALLOC_{name.upper()}_" in os.environ or f"glibc.malloc.{name}" in os.environ.get("GLIBC_TUNABLES", ""):
            continue
        if mallopt(param, value) != 1:
            logger.warning("mallopt rejected glibc malloc %s=%d, keeping the default", name, value)
            continue
        applied.append(f"{name}={value}")
    if applied:
        logger.info("Set glibc malloc %s", ", ".join(applied))
