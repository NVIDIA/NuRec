# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

from dataclasses import dataclass
from pathlib import Path


IMAGEIO_FFMPEG_EXE_ENV = "IMAGEIO_FFMPEG_EXE"
USER_FFMPEG_EXE_ENV = "USER_FFMPEG_EXE"

DEFAULT_FFMPEG_CACHE_DIR = "~/.cache/nurec/ffmpeg"
DEFAULT_FFMPEG_DOWNLOAD_VERSION = "0.6.0"
DEFAULT_FFMPEG_DOWNLOAD_URL: str | None = None
FFMPEG_DOWNLOAD_NOTICE = """--allow-ffmpeg-download was provided.

This container includes an LGPL-compatible limited-codec build of FFmpeg 9.0.1.
The application will now download the full Linux FFmpeg 7.0.2 binary distributed with
upstream imageio-ffmpeg==0.6.0 from ImageIO's imageio-binaries repository, available at https://github.com/imageio/imageio-binaries/tree/master/ffmpeg.

Your use of the FFmpeg binary is governed by the GPLv3 license, available at: https://www.gnu.org/licenses/gpl-3.0.html.

As a reminder, you are responsible for complying with any applicable codec, patent, and redistribution
requirements. Please review this page on FFmpeg licensing for additional details: https://ffmpeg.org/legal.html"""

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FfmpegDownloadConfig:
    """Optional ffmpeg download settings supplied by NuRec configuration.

    Download approval is intentionally not configurable here. Callers must pass
    allow_download=True from an explicit command-line approval path.
    """

    cache_dir: str | os.PathLike[str] | None = None
    download_url: str | os.PathLike[str] | None = None
    download_sha256: str | None = None


def configure_ffmpeg(
    ffmpeg_exe: str | os.PathLike[str] | None = None,
    *,
    download_config: FfmpegDownloadConfig | None = None,
    allow_download: bool = False,
) -> str | None:
    """Configure imageio-ffmpeg from explicit NuRec runtime options.

    Returns the executable path selected for imageio-ffmpeg, or None when no
    override was requested. The allow_download flag must come from an explicit
    command-line approval to download an ffmpeg binary when no explicit
    executable override is configured.
    """
    explicit_exe = _as_nonempty_str(ffmpeg_exe)
    explicit_exe = explicit_exe or _as_nonempty_str(os.getenv(USER_FFMPEG_EXE_ENV))
    explicit_exe = explicit_exe or _as_nonempty_str(os.getenv(IMAGEIO_FFMPEG_EXE_ENV))

    if explicit_exe is not None:
        os.environ[IMAGEIO_FFMPEG_EXE_ENV] = explicit_exe
        return explicit_exe

    download_config = download_config or FfmpegDownloadConfig()
    if not allow_download:
        return None

    downloaded_exe = download_ffmpeg_binary(
        cache_dir=download_config.cache_dir,
        download_url=download_config.download_url,
        download_sha256=download_config.download_sha256,
    )
    os.environ[IMAGEIO_FFMPEG_EXE_ENV] = downloaded_exe
    return downloaded_exe


def download_ffmpeg_binary(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    download_url: str | os.PathLike[str] | None = None,
    download_sha256: str | None = None,
) -> str:
    """Download and cache an ffmpeg executable selected for this platform."""
    version = DEFAULT_FFMPEG_DOWNLOAD_VERSION
    binary_name = _target_binary_name(version)
    source_url, expected_sha256 = _resolve_download_source(
        version=version,
        download_url=download_url,
        download_sha256=download_sha256,
    )
    cache_root = Path(_as_nonempty_str(cache_dir) or DEFAULT_FFMPEG_CACHE_DIR).expanduser()
    target = cache_root / _cache_key(version, source_url, expected_sha256) / binary_name

    if target.exists() and _is_valid_exe(str(target)):
        return str(target)

    _emit_download_notice()
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading ffmpeg binary from %s", source_url)

    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as temp_download:
        temp_download_path = Path(temp_download.name)

    try:
        _download_source_to_file(source_url, temp_download_path, expected_sha256=expected_sha256)
        _extract_or_copy_ffmpeg(temp_download_path, target, binary_name)
    finally:
        temp_download_path.unlink(missing_ok=True)

    if not _is_valid_exe(str(target)):
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded ffmpeg binary is not executable: {target}")

    return str(target)


def _as_nonempty_str(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    value_str = os.fspath(value).strip()
    return value_str or None


# Maps upstream imageio-ffmpeg wheel version -> the FFmpeg binary version it
# bundles. We avoid reading imageio_ffmpeg._definitions.FNAME_PER_PLATFORM
# because the locally installed wheel may be the NuRec-patched LGPL build
# (the wheel build sed-rewrites that table to point at the LGPL FFmpeg version),
# which would not match what is actually inside the upstream wheel we are downloading.
_UPSTREAM_FFMPEG_BINARY_VERSION: dict[str, str] = {
    "0.6.0": "7.0.2",
}


def _target_platform_tag() -> str:
    import platform

    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if machine in ("x86_64", "amd64"):
            return "linux-x86_64"
        if machine in ("aarch64", "arm64"):
            return "linux-aarch64"
    raise RuntimeError(f"No imageio-ffmpeg binary is known for platform {sys.platform}/{machine}")


def _target_binary_name(version: str) -> str:
    ffmpeg_version = _UPSTREAM_FFMPEG_BINARY_VERSION.get(version)
    if ffmpeg_version is None:
        raise RuntimeError(
            f"Unknown upstream FFmpeg binary version for imageio-ffmpeg=={version}; "
            f"add an entry to _UPSTREAM_FFMPEG_BINARY_VERSION."
        )
    return f"ffmpeg-{_target_platform_tag()}-v{ffmpeg_version}"


def _target_wheel_tag() -> str:
    from imageio_ffmpeg._definitions import WHEEL_BUILDS, get_platform  # type: ignore[import-not-found]

    platform_name = get_platform()
    for wheel_tag, wheel_platform in WHEEL_BUILDS.items():
        if wheel_platform == platform_name:
            return wheel_tag
    raise RuntimeError(f"No imageio-ffmpeg wheel is known for platform {platform_name!r}")


def _resolve_download_source(
    *,
    version: str,
    download_url: str | os.PathLike[str] | None,
    download_sha256: str | None,
) -> tuple[str, str | None]:
    source_url = _as_nonempty_str(download_url) or DEFAULT_FFMPEG_DOWNLOAD_URL
    source_sha256 = _as_nonempty_str(download_sha256)
    if source_url is not None:
        # Require a SHA256 pin for remote URLs so an unverified payload cannot
        # be downloaded and then executed by _is_valid_exe(). Local file:// and
        # bare-path sources are exempt; the upstream-default path falls through
        # to _imageio_ffmpeg_wheel_url() which always returns a SHA-pinned pair.
        parsed = urllib.parse.urlparse(source_url)
        if parsed.scheme not in ("", "file") and source_sha256 is None:
            raise RuntimeError("download_sha256 is required for remote ffmpeg downloads")
        return source_url, source_sha256

    return _imageio_ffmpeg_wheel_url(version)


def _imageio_ffmpeg_wheel_url(version: str) -> tuple[str, str]:
    release = _pypi_release_json(version)
    wheel_tag = _target_wheel_tag()
    for release_file in release.get("urls", []):
        filename = release_file.get("filename", "")
        if release_file.get("packagetype") == "bdist_wheel" and filename.endswith(f"-{wheel_tag}.whl"):
            digest = release_file.get("digests", {}).get("sha256")
            url = release_file.get("url")
            if not digest or not url:
                raise RuntimeError(f"PyPI metadata for {filename} is missing URL or SHA256 digest")
            return url, digest
    raise RuntimeError(f"No imageio-ffmpeg {version} wheel found for tag {wheel_tag!r}")


def _pypi_release_json(version: str) -> dict:
    url = f"https://pypi.org/pypi/imageio-ffmpeg/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _cache_key(version: str, source_url: str, expected_sha256: str | None = None) -> str:
    # Pin SHA256 into the cache key so an updated SHA256 invalidates the cached entry.
    source_hash = hashlib.sha256(f"{source_url}\0{expected_sha256 or ''}".encode("utf-8")).hexdigest()[:12]
    return f"imageio-ffmpeg-{version}-{source_hash}"


def _emit_download_notice() -> None:
    print(FFMPEG_DOWNLOAD_NOTICE, file=sys.stderr)


def _download_source_to_file(url_or_path: str, output_path: Path, *, expected_sha256: str | None) -> None:
    digest = hashlib.sha256()
    parsed_url = urllib.parse.urlparse(url_or_path)
    if parsed_url.scheme in ("", "file"):
        source_path = Path(urllib.request.url2pathname(parsed_url.path) if parsed_url.scheme == "file" else url_or_path)
        with source_path.open("rb") as source, output_path.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
    else:
        with urllib.request.urlopen(url_or_path, timeout=30) as response, output_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)

    if expected_sha256 is None:
        return

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded ffmpeg payload hash mismatch: expected {expected_sha256}, got {actual_sha256}")


def _extract_or_copy_ffmpeg(download_path: Path, target: Path, binary_name: str) -> None:
    # Use a unique temp file per writer so concurrent extractions to the same
    # cache key cannot clobber each other and leave a corrupted target.
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f"{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        temp_target = Path(tmp.name)
    try:
        if zipfile.is_zipfile(download_path):
            member_name = f"imageio_ffmpeg/binaries/{binary_name}"
            with zipfile.ZipFile(download_path) as archive:
                with archive.open(member_name) as member, temp_target.open("wb") as output:
                    shutil.copyfileobj(member, output)
        else:
            shutil.copyfile(download_path, temp_target)

        mode = temp_target.stat().st_mode
        temp_target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        temp_target.replace(target)
    finally:
        temp_target.unlink(missing_ok=True)


def _is_valid_exe(exe: str) -> bool:
    try:
        with open(os.devnull, "w") as null:
            subprocess.check_call(
                [exe, "-version"],
                stdout=null,
                stderr=subprocess.STDOUT,
                timeout=15,
            )
        return True
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
