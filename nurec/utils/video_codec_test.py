# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codec selection in the imageio-ffmpeg wheel's `_video_codec`, with the NVENC probe's ffmpeg mocked."""

import subprocess

import pytest

from imageio_ffmpeg import _video_codec as video_codec


@pytest.fixture(autouse=True)
def _unprobed(monkeypatch):
    monkeypatch.setattr(video_codec, "_nvenc_state", None)
    monkeypatch.setattr(video_codec, "_fallback_warned", set())
    monkeypatch.setattr(video_codec, "_small_frame_warned", set())
    video_codec._nvenc_accepts.cache_clear()


def _probe(monkeypatch, returncode: int) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, "", "")

    monkeypatch.setattr(video_codec.subprocess, "run", run)
    return calls


def test_probe_encodes_a_frame_nvenc_accepts(monkeypatch):
    calls = _probe(monkeypatch, returncode=0)

    assert video_codec.pick_codec("out.mp4") == "h264_nvenc"
    cmd = calls[0]
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
    size = next(arg for arg in cmd if arg.startswith("nullsrc=")).split("s=")[1].split(":")[0]
    width, height = map(int, size.split("x"))
    # NVENC rejects frames below its minimum size: 16x16 fails on every GPU, 256x256 passes.
    assert width >= 256 and height >= 256


def test_failed_probe_falls_back_to_av1(monkeypatch):
    _probe(monkeypatch, returncode=1)

    assert video_codec.pick_codec("out.mp4") == "libsvtav1"


def test_probe_runs_once(monkeypatch):
    calls = _probe(monkeypatch, returncode=0)

    video_codec.pick_codec("a.mp4")
    video_codec.pick_codec("b.mp4")

    assert len(calls) == 1


def test_non_mp4_is_left_to_ffmpeg(monkeypatch):
    calls = _probe(monkeypatch, returncode=0)

    assert video_codec.pick_codec("out.mkv") is None
    assert calls == []


def _write(monkeypatch, size, **kwargs) -> dict:
    written = {}

    def orig(path, size, **kw):
        written.update(kw)

    monkeypatch.setattr(video_codec, "_orig_write_frames", orig)
    video_codec._io_module.write_frames("out.mp4", size, **kwargs)
    return written


def test_implicit_mp4_codec_is_av1_even_with_nvenc(monkeypatch):
    calls = _probe(monkeypatch, returncode=0)

    # imageio passes codec=None when the caller leaves it unset.
    assert _write(monkeypatch, (1920, 1080), codec=None)["codec"] == "libsvtav1"
    assert calls == []


def test_explicit_codec_is_passed_through(monkeypatch):
    _probe(monkeypatch, returncode=0)

    assert _write(monkeypatch, (1920, 1080), codec="h264_nvenc")["codec"] == "h264_nvenc"


def test_codec_config_asks_for_nvenc(monkeypatch):
    _probe(monkeypatch, returncode=0)

    assert video_codec.get_codec_config("out.mp4", crf=20) == ("h264_nvenc", ["-cq", "20", "-profile:v", "high"])


def test_codec_config_falls_back_with_matching_flags_for_frames_nvenc_rejects(monkeypatch):
    calls: list[list[str]] = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0 if "nullsrc=s=256x256:d=0.1" in cmd else 1, "", "")

    monkeypatch.setattr(video_codec.subprocess, "run", run)

    assert video_codec.get_codec_config("out.mp4", crf=20, size=(64, 128)) == ("libsvtav1", ["-crf", "20"])
    assert "nullsrc=s=64x128:d=0.1" in calls[-1]
    assert calls[-1][calls[-1].index("-pix_fmt") + 1] == "yuv420p"


def test_codec_config_keeps_nvenc_for_frames_it_accepts(monkeypatch):
    calls = _probe(monkeypatch, returncode=0)

    assert video_codec.get_codec_config("out.mp4", crf=20, size=(3840, 2160)) == (
        "h264_nvenc",
        ["-cq", "20", "-profile:v", "high"],
    )
    assert len(calls) == 2
