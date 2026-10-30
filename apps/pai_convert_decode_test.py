# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PAI converter decodes camera video in presentation order.

testdata/hevc_bframes_levels.mp4: 48 frames of 256x256 HEVC with B-frames (x265 bframes=3, keyint=24),
frame i a flat grey of level 20 + 4 * i, so a frame decoded out of place shows as a wrong level.
"""

from pathlib import Path

from python.runfiles import runfiles


def test_frames_in_presentation_order():
    from tools.data_converter.pai.converter import _decode_rgb_frames  # noqa: PLC0415

    r = runfiles.Create()
    video = r.Rlocation("_main/apps/testdata/hevc_bframes_levels.mp4", r.CurrentRepository())
    assert video is not None, "test video not in runfiles"
    levels = [float(frame.mean()) for frame in _decode_rgb_frames(Path(video))]
    assert len(levels) == 48
    for i, level in enumerate(levels):
        assert abs(level - (20 + 4 * i)) <= 3, f"frame {i}: level {level:.1f}, expected {20 + 4 * i}"
