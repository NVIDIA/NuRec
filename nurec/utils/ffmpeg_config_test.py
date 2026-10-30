# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import zipfile

from pathlib import Path

import pytest

from nurec.utils import ffmpeg_config


def _clear_ffmpeg_env(monkeypatch):
    monkeypatch.delenv(ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV, raising=False)
    monkeypatch.delenv(ffmpeg_config.USER_FFMPEG_EXE_ENV, raising=False)


def test_configure_ffmpeg_prefers_explicit_argument(monkeypatch, tmp_path: Path) -> None:
    _clear_ffmpeg_env(monkeypatch)
    ffmpeg_exe = tmp_path / "ffmpeg"
    monkeypatch.setenv(ffmpeg_config.USER_FFMPEG_EXE_ENV, "/env/ffmpeg")

    result = ffmpeg_config.configure_ffmpeg(ffmpeg_exe=ffmpeg_exe)

    assert result == str(ffmpeg_exe)
    assert os.environ[ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV] == str(ffmpeg_exe)


def test_configure_ffmpeg_uses_user_env_alias(monkeypatch) -> None:
    _clear_ffmpeg_env(monkeypatch)
    monkeypatch.setenv(ffmpeg_config.USER_FFMPEG_EXE_ENV, "/opt/ffmpeg")

    result = ffmpeg_config.configure_ffmpeg()

    assert result == "/opt/ffmpeg"
    assert os.environ[ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV] == "/opt/ffmpeg"


def test_configure_ffmpeg_uses_existing_imageio_env(monkeypatch) -> None:
    _clear_ffmpeg_env(monkeypatch)
    monkeypatch.setenv(ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV, "/imageio/ffmpeg")

    result = ffmpeg_config.configure_ffmpeg()

    assert result == "/imageio/ffmpeg"
    assert os.environ[ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV] == "/imageio/ffmpeg"


def test_configure_ffmpeg_leaves_environment_unchanged_without_override(monkeypatch) -> None:
    _clear_ffmpeg_env(monkeypatch)

    assert ffmpeg_config.configure_ffmpeg() is None
    assert ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV not in os.environ


def test_configure_ffmpeg_does_not_download_without_permission(monkeypatch) -> None:
    _clear_ffmpeg_env(monkeypatch)

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(ffmpeg_config, "download_ffmpeg_binary", fail_download)

    assert ffmpeg_config.configure_ffmpeg(allow_download=False) is None
    assert ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV not in os.environ


def test_configure_ffmpeg_download_config_does_not_approve_download(monkeypatch) -> None:
    _clear_ffmpeg_env(monkeypatch)

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(ffmpeg_config, "download_ffmpeg_binary", fail_download)

    result = ffmpeg_config.configure_ffmpeg(
        download_config=ffmpeg_config.FfmpegDownloadConfig(download_url="https://example.invalid/ffmpeg.whl")
    )

    assert result is None
    assert ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV not in os.environ


def test_configure_ffmpeg_downloads_when_approved(monkeypatch) -> None:
    _clear_ffmpeg_env(monkeypatch)

    def fake_download(**kwargs):
        assert kwargs == {
            "cache_dir": None,
            "download_url": None,
            "download_sha256": None,
        }
        return "/downloaded/ffmpeg"

    monkeypatch.setattr(ffmpeg_config, "download_ffmpeg_binary", fake_download)

    assert ffmpeg_config.configure_ffmpeg(allow_download=True) == "/downloaded/ffmpeg"
    assert os.environ[ffmpeg_config.IMAGEIO_FFMPEG_EXE_ENV] == "/downloaded/ffmpeg"


def test_configure_ffmpeg_accepts_download_config(monkeypatch) -> None:
    _clear_ffmpeg_env(monkeypatch)
    captured_kwargs = {}

    def fake_download(**kwargs):
        captured_kwargs.update(kwargs)
        return "/configured/ffmpeg"

    monkeypatch.setattr(ffmpeg_config, "download_ffmpeg_binary", fake_download)

    result = ffmpeg_config.configure_ffmpeg(
        allow_download=True,
        download_config=ffmpeg_config.FfmpegDownloadConfig(
            cache_dir="/cache",
            download_url="https://example.invalid/ffmpeg.whl",
            download_sha256="abc123",
        ),
    )

    assert result == "/configured/ffmpeg"
    assert captured_kwargs == {
        "cache_dir": "/cache",
        "download_url": "https://example.invalid/ffmpeg.whl",
        "download_sha256": "abc123",
    }


def test_imageio_ffmpeg_wheel_url_selects_current_platform_wheel(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg_config, "_target_wheel_tag", lambda: "py3-none-manylinux2014_x86_64")
    monkeypatch.setattr(
        ffmpeg_config,
        "_pypi_release_json",
        lambda version: {
            "urls": [
                {
                    "filename": "imageio_ffmpeg-0.6.0.tar.gz",
                    "packagetype": "sdist",
                    "url": "https://example.invalid/sdist",
                    "digests": {"sha256": "sdist-sha"},
                },
                {
                    "filename": "imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl",
                    "packagetype": "bdist_wheel",
                    "url": "https://example.invalid/wheel",
                    "digests": {"sha256": "wheel-sha"},
                },
            ]
        },
    )

    assert ffmpeg_config._imageio_ffmpeg_wheel_url("0.6.0") == ("https://example.invalid/wheel", "wheel-sha")


def test_download_ffmpeg_extracts_binary_and_prints_download_notice(monkeypatch, tmp_path: Path, capsys) -> None:
    _clear_ffmpeg_env(monkeypatch)
    binary_name = "ffmpeg-linux-x86_64-v8.1.1"
    wheel_path = tmp_path / "imageio_ffmpeg.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(f"imageio_ffmpeg/binaries/{binary_name}", b"binary")

    monkeypatch.setattr(ffmpeg_config, "_target_binary_name", lambda version: binary_name)
    monkeypatch.setattr(ffmpeg_config, "_is_valid_exe", lambda exe: True)

    result = ffmpeg_config.download_ffmpeg_binary(cache_dir=tmp_path / "cache", download_url=wheel_path)

    result_path = Path(result)
    assert result_path.read_bytes() == b"binary"
    assert result_path.stat().st_mode & stat.S_IXUSR
    captured = capsys.readouterr()
    assert captured.err.strip() == ffmpeg_config.FFMPEG_DOWNLOAD_NOTICE


def test_download_ffmpeg_uses_upstream_imageio_version(monkeypatch, tmp_path: Path, capsys) -> None:
    _clear_ffmpeg_env(monkeypatch)
    payload_path = tmp_path / "ffmpeg"
    payload_path.write_bytes(b"binary")
    captured_kwargs = {}
    notice_seen_during_resolve = False

    def fake_resolve_download_source(**kwargs):
        nonlocal notice_seen_during_resolve
        notice_seen_during_resolve = capsys.readouterr().err.strip() == ffmpeg_config.FFMPEG_DOWNLOAD_NOTICE
        captured_kwargs.update(kwargs)
        return str(payload_path), None

    monkeypatch.setattr(ffmpeg_config, "_target_binary_name", lambda version: "ffmpeg")
    monkeypatch.setattr(ffmpeg_config, "_resolve_download_source", fake_resolve_download_source)
    monkeypatch.setattr(ffmpeg_config, "_is_valid_exe", lambda exe: True)

    ffmpeg_config.download_ffmpeg_binary(cache_dir=tmp_path / "cache")

    assert captured_kwargs == {
        "version": "0.6.0",
        "download_url": None,
        "download_sha256": None,
    }
    assert not notice_seen_during_resolve


def test_download_ffmpeg_cache_hit_does_not_emit_notice(monkeypatch, tmp_path: Path, capsys) -> None:
    """Cached binary: download_ffmpeg_binary should return immediately without printing the GPL notice."""
    _clear_ffmpeg_env(monkeypatch)
    binary_name = "ffmpeg-linux-x86_64-v8.1.1"
    source_url = "https://example.invalid/imageio_ffmpeg.whl"
    cache_root = tmp_path / "cache"
    target = cache_root / ffmpeg_config._cache_key("0.6.0", source_url) / binary_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"cached binary")

    monkeypatch.setattr(ffmpeg_config, "_target_binary_name", lambda version: binary_name)
    monkeypatch.setattr(ffmpeg_config, "_resolve_download_source", lambda **kw: (source_url, None))
    monkeypatch.setattr(ffmpeg_config, "_is_valid_exe", lambda exe: True)

    result = ffmpeg_config.download_ffmpeg_binary(cache_dir=cache_root)

    assert result == str(target)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_resolve_download_source_requires_sha256_for_remote_url() -> None:
    """Remote URLs without an explicit SHA256 pin must fail fast."""
    with pytest.raises(RuntimeError, match="download_sha256 is required"):
        ffmpeg_config._resolve_download_source(
            version="0.6.0",
            download_url="https://example.invalid/ffmpeg.whl",
            download_sha256=None,
        )


def test_resolve_download_source_local_path_skips_sha256_check(tmp_path: Path) -> None:
    """Local file paths (and file:// URLs) are exempt from the SHA256-pin requirement."""
    local_path = tmp_path / "ffmpeg.whl"
    local_path.write_bytes(b"")
    url, sha = ffmpeg_config._resolve_download_source(
        version="0.6.0",
        download_url=str(local_path),
        download_sha256=None,
    )
    assert url == str(local_path)
    assert sha is None


def test_target_binary_name_returns_upstream_v7_0_2_for_0_6_0() -> None:
    # _target_binary_name must compute the binary name from the requested
    # upstream wheel version, NOT from the locally installed imageio_ffmpeg
    # module — which under the NuRec LGPL build is sed-rewritten to point at
    # the LGPL FFmpeg version and would mismatch the upstream wheel's contents.
    name = ffmpeg_config._target_binary_name("0.6.0")
    assert name.endswith("-v7.0.2"), name
    assert name in {"ffmpeg-linux-x86_64-v7.0.2", "ffmpeg-linux-aarch64-v7.0.2"}, name


def test_target_binary_name_unknown_version_raises() -> None:
    with pytest.raises(RuntimeError, match="Unknown upstream FFmpeg"):
        ffmpeg_config._target_binary_name("99.99.99")


def test_download_ffmpeg_extracts_upstream_v7_0_2_binary(monkeypatch, tmp_path: Path) -> None:
    # End-to-end: production-shaped upstream wheel contains the v7.0.2 binary
    # (NOT the patched LGPL v8.1.1). Regression for the cross-version KeyError
    # that --allow-ffmpeg-download hit before _target_binary_name was decoupled
    # from imageio_ffmpeg._definitions.FNAME_PER_PLATFORM.
    _clear_ffmpeg_env(monkeypatch)
    binary_name = ffmpeg_config._target_binary_name("0.6.0")
    assert binary_name.endswith("-v7.0.2"), binary_name
    wheel_path = tmp_path / "imageio_ffmpeg.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(f"imageio_ffmpeg/binaries/{binary_name}", b"binary")
    monkeypatch.setattr(ffmpeg_config, "_is_valid_exe", lambda exe: True)

    result = ffmpeg_config.download_ffmpeg_binary(cache_dir=tmp_path / "cache", download_url=wheel_path)

    assert Path(result).name == binary_name
    assert Path(result).read_bytes() == b"binary"


def test_download_ffmpeg_rejects_hash_mismatch(monkeypatch, tmp_path: Path) -> None:
    _clear_ffmpeg_env(monkeypatch)
    payload_path = tmp_path / "ffmpeg"
    payload_path.write_bytes(b"binary")
    monkeypatch.setattr(ffmpeg_config, "_target_binary_name", lambda version: "ffmpeg")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        ffmpeg_config.download_ffmpeg_binary(
            cache_dir=tmp_path / "cache",
            download_url=payload_path,
            download_sha256="not-the-real-hash",
        )
