# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import gc
import threading
import weakref

from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import pytest
import torch

from nurec.lidar.firing_patterns.templates import LidarFiringPatternTemplate
from nurec.render.lidar_tiling import build_unstructured_tiling, firing_pattern_fov
from nurec.render.render import LidarRayBundle, PoseRange, _crop_to_sweep
from nurec.utils.types import FrameConversion


def _template(
    n: int = 8,
    *,
    span_us: int = 70_000,
    azimuth_offset: float = 0.0,
) -> LidarFiringPatternTemplate:
    azimuths = np.linspace(-0.5, 0.5, n) + azimuth_offset
    directions = np.stack([np.cos(azimuths), np.sin(azimuths), np.zeros(n)], axis=-1).astype(np.float32)
    return LidarFiringPatternTemplate(
        directions_sensor=directions,
        timestamp_offsets_us=np.linspace(0, span_us, n).astype(np.int64),
        packet_ray_offsets=np.array([0, n // 2, n], dtype=np.int64),
    )


def test_zero_duration_flash_pattern_keeps_the_start_pose() -> None:
    start = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    end = torch.tensor([4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0])

    cropped = _crop_to_sweep(PoseRange(start, end, 100, 100), sweep_us=0)

    torch.testing.assert_close(cropped.start_pose_tquat_sensor_world, start)
    torch.testing.assert_close(cropped.end_pose_tquat_sensor_world, start)
    assert (cropped.start_timestamp_us, cropped.end_timestamp_us) == (100, 100)


class TestFovBounds:
    def test_brackets_every_ray(self):
        template = _template(n=16)
        fov = firing_pattern_fov(template)
        azimuth = np.arctan2(template.directions_sensor[:, 1], template.directions_sensor[:, 0])

        assert fov.periodic_az is False
        assert fov.az_lo < azimuth.min() and fov.az_hi > azimuth.max()

    def test_a_full_circle_template_is_periodic(self):
        angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
        template = LidarFiringPatternTemplate(
            directions_sensor=np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1).astype(
                np.float32
            ),
            timestamp_offsets_us=np.arange(64, dtype=np.int64),
            packet_ray_offsets=np.array([0, 32, 64], dtype=np.int64),
        )

        assert firing_pattern_fov(template).periodic_az is True

    def test_limited_rear_arc_crossing_the_seam_is_rejected(self):
        template = LidarFiringPatternTemplate(
            directions_sensor=np.array([[-1.0, -1e-6, 0.0], [-1.0, 1e-6, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([0, 1], dtype=np.int64),
            packet_ray_offsets=np.array([0, 2], dtype=np.int64),
        )

        with pytest.raises(ValueError, match=r"crosses the \+/-180 degree seam"):
            firing_pattern_fov(template)


@dataclasses.dataclass
class _FakeTiling:
    directions: object
    meta: dict

    def to(self, _device):
        return self


def test_tiling_is_cached_per_template_and_restamped_per_request():
    template = _template(n=6)
    calls = []

    def fake_build_tiling(directions, timestamps, fov, **frame_bounds):
        calls.append((directions, timestamps, fov, frame_bounds))
        return _FakeTiling(
            directions=object(),
            meta={"frame_start_us": 0.0, "frame_end_us": 0.0},
        )

    with mock.patch("nurec.utils.unstructured_lidar.build_tiling", fake_build_tiling):
        first = build_unstructured_tiling(template, 1_000, "cpu")
        second = build_unstructured_tiling(template, 7_000, "cpu")

    assert len(calls) == 1
    assert first.directions is second.directions
    assert first.meta["frame_start_us"] == 1_000
    assert second.meta["frame_start_us"] == 7_000


@pytest.mark.parametrize("first_offset_us", [0, 100])
def test_tiling_timestamps_match_ray_motion_from_physical_frame_start(first_offset_us: int) -> None:
    """Check tiling and motion use the same time origin, including cached requests.

    Args:
        first_offset_us: First firing's offset from physical frame start, in microseconds.
    """
    template = dataclasses.replace(
        _template(n=3, span_us=500),
        timestamp_offsets_us=np.array([first_offset_us, 300, 500], dtype=np.int64),
    )
    start = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    end = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    bundle = LidarRayBundle.build_model_free(
        template,
        500,
        PoseRange(start, end, 1_000, 1_500),
        FrameConversion(matrix=np.eye(4, dtype=np.float32)),
    )

    tiling = bundle.rendering_data.unstructured_lidar_tiling
    assert tiling is not None
    expected_times = torch.tensor([first_offset_us / 500, 0.6, 1.0], device="cuda")
    torch.testing.assert_close(tiling.relative_timestamps, expected_times)
    torch.testing.assert_close(bundle.rendering_data.rays[0, :, 0, 0], 5.0 * expected_times)
    assert tiling.meta["frame_duration_us"] == 500

    restamped = build_unstructured_tiling(template, 7_000, torch.device("cuda"))
    assert restamped.relative_timestamps is tiling.relative_timestamps
    assert (tiling.meta["frame_start_us"], tiling.meta["frame_end_us"]) == (1_000, 1_500)
    assert (restamped.meta["frame_start_us"], restamped.meta["frame_end_us"]) == (7_000, 7_500)


def test_tiling_cache_is_scoped_to_template_and_device():
    template = _template(n=6)
    other = _template(n=6)

    def fake_build_tiling(*args, **kwargs):
        return _FakeTiling(directions=object(), meta={})

    with mock.patch("nurec.utils.unstructured_lidar.build_tiling", side_effect=fake_build_tiling) as build:
        first = build_unstructured_tiling(template, 0, torch.device("cpu"))
        same = build_unstructured_tiling(template, 1, torch.device("cpu"))
        other_device = build_unstructured_tiling(template, 0, torch.device("cuda:0"))
        other_template = build_unstructured_tiling(other, 0, torch.device("cpu"))

    assert build.call_count == 3
    assert first.directions is same.directions
    assert first.directions is not other_device.directions
    assert first.directions is not other_template.directions


def test_tiling_cache_releases_resources_when_template_is_discarded():
    template = _template(n=6)
    template_ref = weakref.ref(template)

    class Resource:
        pass

    def fake_build_tiling(*args, **kwargs):
        return _FakeTiling(directions=Resource(), meta={})

    with mock.patch("nurec.utils.unstructured_lidar.build_tiling", fake_build_tiling):
        tiling = build_unstructured_tiling(template, 0, torch.device("cpu"))
    resource_ref = weakref.ref(tiling.directions)
    del tiling
    gc.collect()
    assert resource_ref() is not None

    del template
    gc.collect()
    assert template_ref() is None
    assert resource_ref() is None


def test_concurrent_requests_share_one_tiling_build():
    template = _template(n=6)
    barrier = threading.Barrier(4)

    def fake_build_tiling(*args, **kwargs):
        return _FakeTiling(directions=object(), meta={})

    def render(frame_start_us):
        barrier.wait(timeout=10)
        return build_unstructured_tiling(template, frame_start_us, torch.device("cpu"))

    with mock.patch("nurec.utils.unstructured_lidar.build_tiling", side_effect=fake_build_tiling) as build:
        with ThreadPoolExecutor(max_workers=4) as executor:
            tilings = list(executor.map(render, range(4)))

    assert build.call_count == 1
    assert all(tiling.directions is tilings[0].directions for tiling in tilings)
    assert [tiling.meta["frame_start_us"] for tiling in tilings] == list(range(4))


@pytest.mark.parametrize("end_angle", [np.pi / 2.0, np.pi])
def test_build_model_free_matches_a_slerp_pose_oracle(end_angle):
    template = LidarFiringPatternTemplate(
        directions_sensor=np.array([[1.0, 0.0, 0.0]] * 3, dtype=np.float32),
        timestamp_offsets_us=np.array([0, 250, 1_000], dtype=np.int64),
        packet_ray_offsets=np.array([0, 1, 3], dtype=np.int64),
    )
    half_angle = end_angle / 2.0
    poses = PoseRange(
        start_pose_tquat_sensor_world=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        end_pose_tquat_sensor_world=torch.tensor(
            [4.0, 0.0, 0.0, 0.0, 0.0, np.sin(half_angle), np.cos(half_angle)],
            dtype=torch.float32,
        ),
        start_timestamp_us=10_000,
        end_timestamp_us=11_000,
    )

    with mock.patch(
        "nurec.render.render.build_unstructured_tiling",
        return_value=mock.sentinel.tiling,
    ):
        bundle = LidarRayBundle.build_model_free(
            template,
            1_000,
            poses,
            FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            unique_sensor_idx=7,
        )

    rays = bundle.rendering_data.rays[0, :, 0]
    expected_origins = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        device="cuda",
    )
    expected_directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [np.cos(end_angle / 4.0), np.sin(end_angle / 4.0), 0.0],
            [np.cos(end_angle), np.sin(end_angle), 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    torch.testing.assert_close(rays[:, :3], expected_origins)
    torch.testing.assert_close(rays[:, 3:], expected_directions, atol=1e-5, rtol=1e-5)
    assert bundle.frame_meta.unique_sensor_idx == 7


def test_a_frame_request_is_cropped_to_the_template_firing_span():
    sweep_us, frame_us = 59_720, 100_000
    template = LidarFiringPatternTemplate(
        directions_sensor=np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float32),
        timestamp_offsets_us=np.array([0, 20_000, 40_000, sweep_us], dtype=np.int64),
        packet_ray_offsets=np.array([0, 2, 4], dtype=np.int64),
    )
    start = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    def end_pose(duration_us: int) -> torch.Tensor:
        half_angle = np.deg2rad(1.5) * duration_us / frame_us
        return torch.tensor(
            [
                1.5 * duration_us / frame_us,
                0.0,
                0.0,
                0.0,
                0.0,
                np.sin(half_angle),
                np.cos(half_angle),
            ],
            dtype=torch.float32,
        )

    def build(duration_us: int) -> LidarRayBundle:
        with mock.patch(
            "nurec.render.render.build_unstructured_tiling",
            return_value=mock.sentinel.tiling,
        ):
            return LidarRayBundle.build_model_free(
                firing_pattern_template=template,
                frame_period_us=frame_us,
                lidar_to_world=PoseRange(start, end_pose(duration_us), 0, duration_us),
                world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            )

    whole, sweep = build(frame_us), build(sweep_us)

    torch.testing.assert_close(
        whole.rendering_data.rays,
        sweep.rendering_data.rays,
        rtol=0.0,
        atol=2e-5,
    )
