# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a NuRec checkout and its pinned build environment."""

from __future__ import annotations

import os
import re
import subprocess
import sys

from pathlib import Path

from lib.parse_input import parse_input


def release_image(source: Path) -> str | None:
    """Derive the public dependency-image tag from the checkout's release version."""
    version_file = source / "bazel/version/VERSION_FILE"
    if not version_file.is_file():
        return None
    version = {}
    for line in version_file.read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key in {"VERSION_MAJOR", "VERSION_MINOR"} and value.isascii() and value.isdecimal():
            version[key] = value
    if "VERSION_MAJOR" not in version or "VERSION_MINOR" not in version:
        return None
    tag = f"{int(version['VERSION_MAJOR']):02d}.{int(version['VERSION_MINOR']):02d}"
    return f"ghcr.io/nvidia/nurec/nurec_dev_deps:{tag}"


def development_image(source: Path) -> str | None:
    """Read the development image CI pins for a development checkout, if this is one."""
    ci = source / ".gitlab-ci.yml"
    if not ci.is_file():
        return None
    match = re.search(
        r"^default:\s*\n\s+image:\s*\n(?:\s*#[^\n]*\n)*\s+name:\s*(\S+)",
        ci.read_text(),
        re.MULTILINE,
    )
    if not match:
        return None
    if match[1].endswith(":latest"):
        raise ValueError("The checkout development image must not use latest")
    return match[1]


def resolve_runtime(input_path: str, enclosing_root: str) -> tuple[Path, str]:
    """Resolve the selected checkout and require its documented dev dependencies image pin."""
    settings = parse_input(input_path).get("nurec", {})
    source = Path(settings.get("source_dir") or enclosing_root).expanduser()
    if not source.is_absolute():
        raise ValueError("nurec.source_dir must be an absolute NuRec checkout path")
    source = source.resolve()
    if source in {Path("/"), Path.home(), Path("/root"), Path("/etc"), Path("/usr"), Path("/var")}:
        raise ValueError("nurec.source_dir must be a dedicated checkout")
    if (
        not all((source / name).is_file() for name in ("MODULE.bazel", "run.py", ".bazelversion"))
        or not (source / "configs").is_dir()
    ):
        raise ValueError("NuRec source needs MODULE.bazel, run.py, .bazelversion and configs/")
    pinned = development_image(source)
    if pinned:
        image = settings.get("dev_deps_image") or pinned
        if image != pinned:
            raise ValueError(f"nurec.dev_deps_image must match this checkout's .gitlab-ci.yml pin: {pinned}")
        return source, pinned
    pins = set()
    guides = [source / "docs/basics/software.rst", source / "docs/basics/software.md"]
    for path in guides:
        if path.is_file():
            pins.update(
                re.findall(
                    r"ghcr\.io/nvidia/nurec/nurec_dev_deps"
                    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]*(?:@sha256:[0-9a-f]{64})?|@sha256:[0-9a-f]{64})"
                    r"(?![A-Za-z0-9_.:@-])",
                    path.read_text(),
                )
            )
    if any(path.is_file() for path in guides):
        if len(pins) != 1:
            raise ValueError("The selected source must document one pinned nurec_dev_deps image in its software guide")
        pinned = pins.pop()
    else:
        pinned = settings.get("dev_deps_image") or release_image(source)
    if not pinned:
        raise ValueError(
            "Could not read VERSION_MAJOR and VERSION_MINOR from bazel/version/VERSION_FILE; "
            "set nurec.dev_deps_image to the image for this release"
        )
    image = settings.get("dev_deps_image") or pinned
    if ":" not in image.rsplit("/", 1)[-1] or image.split("@")[0].endswith(":latest") or image != pinned:
        raise ValueError(
            f"nurec.dev_deps_image must match this checkout's documented image: {pinned}; latest is not supported"
        )
    return source, image


if __name__ == "__main__":
    try:
        source, image = resolve_runtime(*sys.argv[1:])
        netrc = None
        if development_image(source):
            netrc = Path.home() / ".netrc"
            if not netrc.is_file() or not os.access(netrc, os.R_OK):
                raise ValueError(
                    "This development checkout needs a readable ~/.netrc; follow its README authentication setup"
                )
        git_common = ""
        if (source / ".git").is_file():
            git_common = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True
            ).strip()
        print(source)
        print(image)
        print(netrc or "")
        print(git_common)
    except (ValueError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
