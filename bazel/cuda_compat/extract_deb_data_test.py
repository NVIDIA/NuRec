# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the .deb extraction used for the CUDA compat libraries."""

import io
import tarfile

from pathlib import Path
from typing import Literal

import pytest

from bazel.cuda_compat.extract_deb_data import extract_deb_data, read_data_archive


def _ar_member(name: str, content: bytes) -> bytes:
    header = f"{name + '/':<16}{0:<12}{0:<6}{0:<6}{100644:<8}{len(content):<10}`\n".encode("ascii")
    return header + content + (b"\n" if len(content) % 2 else b"")


def _deb(data_name: str = "data.tar.xz", mode: Literal["w:xz", "w:gz"] = "w:xz", extra: str | None = None) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode=mode) as tar:
        lib = tarfile.TarInfo("usr/local/cuda-12.8/compat/libcuda.so.570")
        content = b"lib"
        lib.size = len(content)
        lib.mode = 0o644
        tar.addfile(lib, io.BytesIO(content))
        link = tarfile.TarInfo("usr/local/cuda-12.8/compat/libcuda.so.1")
        link.type = tarfile.SYMTYPE
        link.linkname = "libcuda.so.570"
        tar.addfile(link)
        if extra:
            escape = tarfile.TarInfo(extra)
            tar.addfile(escape, io.BytesIO())
    return (
        b"!<arch>\n"
        + _ar_member("debian-binary", b"2.0\n")
        + _ar_member("control.tar.xz", b"odd")
        + _ar_member(data_name, payload.getvalue())
    )


def test_extracts_files_and_keeps_symlinks(tmp_path: Path):
    deb = tmp_path / "pkg.deb"
    deb.write_bytes(_deb())
    out = tmp_path / "out"
    extract_deb_data(deb, out)
    compat = out / "usr/local/cuda-12.8/compat"
    assert (compat / "libcuda.so.570").read_bytes() == b"lib"
    assert (compat / "libcuda.so.1").is_symlink()
    assert (compat / "libcuda.so.1").readlink().as_posix() == "libcuda.so.570"


def test_skips_members_before_data_including_odd_sized_ones():
    assert read_data_archive(io.BytesIO(_deb())).startswith(b"\xfd7zXZ")


def test_other_compression_is_detected_from_the_content(tmp_path: Path):
    deb = tmp_path / "pkg.deb"
    deb.write_bytes(_deb("data.tar.gz", "w:gz"))
    extract_deb_data(deb, tmp_path / "out")
    assert (tmp_path / "out/usr/local/cuda-12.8/compat/libcuda.so.570").exists()


def test_rejects_non_ar_input():
    with pytest.raises(ValueError, match="not an ar archive"):
        read_data_archive(io.BytesIO(b"nope"))


def test_requires_a_data_member():
    with pytest.raises(ValueError, match="no data.tar"):
        read_data_archive(io.BytesIO(b"!<arch>\n" + _ar_member("debian-binary", b"2.0\n")))


def test_refuses_members_outside_the_output_dir(tmp_path: Path):
    deb = tmp_path / "pkg.deb"
    deb.write_bytes(_deb(extra="../escaped"))
    with pytest.raises(tarfile.OutsideDestinationError):
        extract_deb_data(deb, tmp_path / "out")
    assert not (tmp_path / "escaped").exists()
