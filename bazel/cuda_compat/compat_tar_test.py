# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks the compat archive keeps the .deb's SONAME symlinks.

Flattening them into copies makes libnvidia-container mount the compat
libcuda.so.1 over the host driver, which fails every GPU call with CUDA error
803 on hosts whose driver is newer than the compat libraries.
"""

import pathlib
import re
import tarfile


# libcuda.so, libnvidia-nvvm.so.4: a bare name or a single-component suffix.
_SONAME = re.compile(r"lib.*\.so(\.\d+)?$")
# libcuda.so.570.211.01: the full version.
_VERSIONED = re.compile(r"lib.*\.so\.\d+\.\d+.*$")


_ARCHIVE = pathlib.Path(__file__).with_name("compat_tar.tar.gz")


def _members():
    with tarfile.open(_ARCHIVE) as tar:
        members = {m.name.lstrip("./"): m for m in tar.getmembers()}
    return {name.rsplit("/", 1)[-1]: m for name, m in members.items() if not m.isdir()}


def test_archive_is_not_empty():
    assert _members(), "compat archive contains no libraries"


def test_sonames_are_symlinks():
    libs = _members()
    sonames = [n for n in libs if _SONAME.match(n)]
    assert sonames, "no SONAME entries found"
    for name in sonames:
        assert libs[name].issym(), (
            f"{name} is a copy, not a symlink: the container toolkit will mount "
            f"it over the host driver (CUDA error 803)"
        )


def test_versioned_libraries_are_regular_files():
    libs = _members()
    versioned = [n for n in libs if _VERSIONED.match(n)]
    assert versioned, "no versioned libraries found"
    for name in versioned:
        assert libs[name].isreg(), f"{name} is not a regular file"


def test_symlinks_resolve_inside_the_archive():
    libs = _members()
    for name, member in libs.items():
        if member.issym():
            target = member.linkname.rsplit("/", 1)[-1]
            assert target in libs, f"{name} -> {target} dangles"
