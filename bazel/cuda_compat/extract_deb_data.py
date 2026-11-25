# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extracts the data archive of a .deb package.

A .deb is an `ar` archive that holds a compressed tar of the installed files. The standard library
reads both layers, so neither the ar nor the xz host tool is needed.

Usage: extract_deb_data.py <deb> <output dir>
"""

import io
import sys
import tarfile

from pathlib import Path
from typing import BinaryIO


_AR_MAGIC = b"!<arch>\n"
_AR_HEADER_SIZE = 60


def read_data_archive(deb: BinaryIO) -> bytes:
    """Return the content of the data.tar.* member of a .deb."""
    if deb.read(len(_AR_MAGIC)) != _AR_MAGIC:
        raise ValueError("not an ar archive")
    while header := deb.read(_AR_HEADER_SIZE):
        if len(header) < _AR_HEADER_SIZE:
            break
        name = header[:16].decode("ascii").strip().rstrip("/")
        size = int(header[48:58].decode("ascii"))
        if name.startswith("data.tar"):
            return deb.read(size)
        # Members are padded to an even length.
        deb.seek(size + (size % 2), io.SEEK_CUR)
    raise ValueError("no data.tar member found")


def extract_deb_data(deb_path: Path, output_dir: Path) -> None:
    """Extract the files a .deb installs into output_dir, keeping symlinks and permissions."""
    with deb_path.open("rb") as deb:
        data = read_data_archive(deb)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        tar.extractall(output_dir, filter="tar")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    extract_deb_data(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
