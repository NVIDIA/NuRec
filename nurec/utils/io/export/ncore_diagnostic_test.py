# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from click.testing import CliRunner

from nurec.utils.io.export.ncore_diagnostic import (
    camera_sampling_from_timestamps,
    export_ncore_diagnostic,
    get_frame_id,
    ncore_export_camera_images,
)


@pytest.mark.parametrize(("source_fps", "expected_step"), ((30.0, 6), (15.0, 3)))
def test_camera_sampling_uses_each_camera_timestamp_rate(source_fps: float, expected_step: int) -> None:
    period_us = 1_000_000.0 / source_fps
    starts = np.arange(20, dtype=np.float64) * period_us
    timestamps = np.column_stack((starts, starts + period_us))

    sampling = camera_sampling_from_timestamps(timestamps, fallback_frame_step=50, fps_sample=5.0)

    assert sampling["frame_step"] == expected_step
    assert sampling["source_fps"] == pytest.approx(source_fps)
    assert sampling["sampled_fps"] == pytest.approx(5.0)


def test_camera_sampling_uses_fallback_when_rate_is_unavailable() -> None:
    timestamps = np.zeros((2, 2), dtype=np.float64)

    sampling = camera_sampling_from_timestamps(timestamps, fallback_frame_step=7, fps_sample=5.0)

    assert sampling["source_fps"] == 0.0
    assert sampling["frame_step"] == 7


@pytest.mark.parametrize(
    ("fallback_frame_step", "fps_sample"),
    ((0, None), (1, 0.0), (1, float("nan"))),
)
def test_camera_sampling_rejects_invalid_options(
    fallback_frame_step: int,
    fps_sample: float | None,
) -> None:
    with pytest.raises(ValueError):
        camera_sampling_from_timestamps(
            np.zeros((2, 2), dtype=np.float64),
            fallback_frame_step=fallback_frame_step,
            fps_sample=fps_sample,
        )


@pytest.mark.parametrize("jpeg_subsampling", [None, 0, 2])
def test_camera_image_export_only_overrides_subsampling_when_requested(
    tmp_path: Path,
    jpeg_subsampling: int | None,
) -> None:
    image = MagicMock()
    camera = MagicMock()
    camera.sensor_id = "camera_front"
    camera.frames_count = 1
    camera.frames_timestamps_us = np.array([[0, 1]], dtype=np.int64)
    camera.get_frame_image.return_value = image
    loader = MagicMock()
    loader.get_camera_sensor.return_value = camera

    ncore_export_camera_images(
        loader=loader,
        camera_ids=["camera_front"],
        output_dir=tmp_path,
        frame_step_camera=1,
        export_images=True,
        export_video=False,
        fps=15,
        name_by_timestamp=False,
        jpeg_quality=91,
        jpeg_subsampling=jpeg_subsampling,
    )

    save_options = image.save.call_args.kwargs
    assert save_options["quality"] == 91
    if jpeg_subsampling is None:
        assert "subsampling" not in save_options
    else:
        assert save_options["subsampling"] == jpeg_subsampling


def test_camera_jpeg_subsampling_cli_option_is_available() -> None:
    result = CliRunner().invoke(export_ncore_diagnostic, ["--help"])

    assert result.exit_code == 0
    assert "--camera-jpeg-subsampling" in result.output


def test_frame_id_zero_pads_timestamps_only_when_asked() -> None:
    """Padding is opt-in, and width 20 matches what the render command writes."""
    sensor = MagicMock()
    sensor.frames_timestamps_us = np.array([[100, 36797]], dtype=np.int64)

    assert get_frame_id(sensor, 0, name_by_timestamp=True) == "36797"
    assert get_frame_id(sensor, 0, name_by_timestamp=True, frame_name_width=20) == "00000000000000036797"
    # Index naming is unaffected by the width.
    assert get_frame_id(sensor, 0, name_by_timestamp=False, frame_name_width=20) == "000000"
