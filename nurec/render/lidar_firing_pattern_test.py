# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from unittest import mock

import numpy as np
import pytest
import torch

from nurec.render.lidar_firing_pattern import LidarFiringPattern, build_unstructured_tiling
from nurec.render.render import LidarRayBundle, PoseRange
from nurec.utils.types import FrameConversion


def _pattern(n: int = 8, span_us: int = 70_000, frame_duration_us: int | None = None) -> LidarFiringPattern:
    """A small fan of rays sweeping azimuth, firing evenly across the span."""
    azimuths = np.linspace(-0.5, 0.5, n)
    directions = np.stack([np.cos(azimuths), np.sin(azimuths), np.zeros(n)], axis=-1).astype(np.float32)
    offsets = np.linspace(0, span_us, n).astype(np.int64)
    return LidarFiringPattern(
        directions_sensor=directions,
        timestamp_offsets_us=offsets,
        frame_duration_us=frame_duration_us if frame_duration_us is not None else max(span_us, 1),
    )


def _write_npz(tmp_path, name="pattern.npz", **arrays):
    path = tmp_path / name
    np.savez(path, **arrays)
    return path


class TestValidation:
    def test_rejects_directions_that_are_not_three_dimensional(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            LidarFiringPattern(
                directions_sensor=np.zeros((4, 2), dtype=np.float32),
                timestamp_offsets_us=np.zeros(4, dtype=np.int64),
                frame_duration_us=1,
            )

    def test_rejects_mismatched_timestamp_count(self):
        # A pattern whose times do not line up with its rays would silently mis-time the
        # whole sweep, so it must not be constructible.
        with pytest.raises(ValueError, match="matching directions_sensor"):
            LidarFiringPattern(
                directions_sensor=np.zeros((4, 3), dtype=np.float32),
                timestamp_offsets_us=np.zeros(3, dtype=np.int64),
                frame_duration_us=1,
            )

    def test_rejects_empty_pattern(self):
        with pytest.raises(ValueError, match="empty"):
            LidarFiringPattern(
                directions_sensor=np.zeros((0, 3), dtype=np.float32),
                timestamp_offsets_us=np.zeros(0, dtype=np.int64),
                frame_duration_us=1,
            )

    def test_rejects_negative_offsets(self):
        with pytest.raises(ValueError, match="non-negative"):
            LidarFiringPattern(
                directions_sensor=np.ones((2, 3), dtype=np.float32),
                timestamp_offsets_us=np.array([-1, 0], dtype=np.int64),
                frame_duration_us=1,
            )

    def test_rejects_a_sweep_that_overruns_its_frame(self):
        # A sweep far past its frame would extrapolate the caller's pose arc, not interpolate it.
        with pytest.raises(ValueError, match="past the .* frame it sweeps within"):
            LidarFiringPattern(
                directions_sensor=np.ones((2, 3), dtype=np.float32) / np.sqrt(3.0),
                timestamp_offsets_us=np.array([0, 200_000], dtype=np.int64),
                frame_duration_us=100_000,
            )

    def test_accepts_a_sweep_that_overruns_its_frame_by_jitter(self):
        # A spinning unit sweeps its whole frame and can pass it slightly (P128: 0.27%).
        LidarFiringPattern(
            directions_sensor=np.ones((2, 3), dtype=np.float32) / np.sqrt(3.0),
            timestamp_offsets_us=np.array([0, 100_258], dtype=np.int64),
            frame_duration_us=99_988,
        )

    @pytest.mark.parametrize("bad_direction", [[0.0, 0.0, 0.0], [np.nan, 0.0, 1.0], [np.inf, 0.0, 1.0]])
    def test_rejects_invalid_direction_vectors(self, bad_direction):
        with pytest.raises(ValueError, match="finite|zero-length"):
            LidarFiringPattern(
                directions_sensor=np.asarray([[1.0, 0.0, 0.0], bad_direction], dtype=np.float32),
                timestamp_offsets_us=np.array([0, 1], dtype=np.int64),
                frame_duration_us=1,
            )


class TestLoad:
    def test_refuses_a_pattern_without_a_frame_duration(self, tmp_path):
        # Serving replays the pattern over a frame, so the duration cannot be left to a guess.
        pattern = _pattern()
        path = _write_npz(
            tmp_path,
            directions_sensor=pattern.directions_sensor,
            timestamp_offsets_us=pattern.timestamp_offsets_us,
        )

        with pytest.raises(ValueError, match="frame_duration_us"):
            LidarFiringPattern.load(path)

    def test_round_trip(self, tmp_path):
        # The frame is longer than the sweep: an idling unit covers 70 of every 100 ms.
        pattern = _pattern(frame_duration_us=100_000)
        path = _write_npz(
            tmp_path,
            directions_sensor=pattern.directions_sensor,
            timestamp_offsets_us=pattern.timestamp_offsets_us,
            frame_duration_us=np.asarray(pattern.frame_duration_us, dtype=np.int64),
        )

        loaded = LidarFiringPattern.load(path)

        np.testing.assert_allclose(loaded.directions_sensor, pattern.directions_sensor, atol=1e-6)
        np.testing.assert_array_equal(loaded.timestamp_offsets_us, pattern.timestamp_offsets_us)
        assert loaded.frame_duration_us == 100_000
        assert loaded.span_us == 70_000

    def test_names_the_missing_fields(self, tmp_path):
        path = _write_npz(tmp_path, directions_sensor=np.zeros((2, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="timestamp_offsets_us"):
            LidarFiringPattern.load(path)

    def test_normalises_non_unit_directions(self, tmp_path):
        # A pattern stored as un-normalised directions would otherwise scale every ray, so
        # loading repairs it rather than rendering a distorted sweep.
        path = _write_npz(
            tmp_path,
            directions_sensor=np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([0, 10], dtype=np.int64),
            frame_duration_us=np.asarray(10, dtype=np.int64),
        )

        loaded = LidarFiringPattern.load(path)

        np.testing.assert_allclose(np.linalg.norm(loaded.directions_sensor, axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("bad_direction", [[0.0, 0.0, 0.0], [np.nan, 0.0, 1.0], [np.inf, 0.0, 1.0]])
    def test_rejects_invalid_directions_before_normalising(self, tmp_path, bad_direction):
        path = _write_npz(
            tmp_path,
            directions_sensor=np.asarray([[1.0, 0.0, 0.0], bad_direction], dtype=np.float32),
            timestamp_offsets_us=np.array([0, 1], dtype=np.int64),
            frame_duration_us=1,
        )

        with pytest.raises(ValueError, match="finite|zero-length"):
            LidarFiringPattern.load(path)

    def test_absolute_timestamps_are_refused(self, tmp_path):
        # Rebasing an epoch onto its own minimum would turn it into a plausible 50us sweep.
        path = _write_npz(
            tmp_path,
            directions_sensor=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([1_700_000_000, 1_700_000_050], dtype=np.int64),
            frame_duration_us=np.asarray(70_000, dtype=np.int64),
        )

        with pytest.raises(ValueError, match="look like absolute timestamps"):
            LidarFiringPattern.load(path)

    def test_a_lead_in_before_the_first_firing_is_kept(self, tmp_path):
        # Offsets are measured from the frame start, so a unit that idles before firing has a
        # non-zero minimum. That gap is real; removing it would place every ray early by it.
        path = _write_npz(
            tmp_path,
            directions_sensor=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([500, 60_500], dtype=np.int64),
            frame_duration_us=np.asarray(70_000, dtype=np.int64),
        )

        loaded = LidarFiringPattern.load(path)

        np.testing.assert_array_equal(loaded.timestamp_offsets_us, [500, 60_500])
        # And it survives into the request's window rather than collapsing to its start.
        np.testing.assert_array_equal(loaded.ray_timestamps_us(1_000), [1_500, 61_500])


class TestRayTimestamps:
    def test_carries_the_measured_offsets_from_the_frame_start(self):
        # The offsets are the sensor's own firing schedule, so they are offset, not stretched.
        pattern = _pattern(n=5, span_us=70_000)

        stamps = pattern.ray_timestamps_us(1_000)

        assert stamps[0] == 1_000
        assert stamps[-1] == 1_000 + 70_000
        np.testing.assert_array_equal(stamps - 1_000, pattern.timestamp_offsets_us)

    def test_the_requested_window_does_not_change_the_schedule(self):
        # A window of a different length would be a different sensor, not this one replayed
        # faster, so the spacing between firings is invariant.
        pattern = _pattern(n=3, span_us=70_000)

        np.testing.assert_array_equal(np.diff(pattern.ray_timestamps_us(0)), np.diff(pattern.ray_timestamps_us(5_000)))

    def test_zero_span_pattern_fires_at_the_frame_start(self):
        pattern = LidarFiringPattern(
            directions_sensor=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.zeros(2, dtype=np.int64),
            frame_duration_us=1,
        )

        np.testing.assert_array_equal(pattern.ray_timestamps_us(500), [500, 500])


class TestFovBounds:
    def test_brackets_every_ray(self):
        pattern = _pattern(n=16)
        az_lo, az_hi, el_lo, el_hi, periodic = pattern.fov()
        assert periodic is False

        d = pattern.directions_sensor.astype(np.float64)
        az = np.arctan2(d[:, 1], d[:, 0])
        el = np.arcsin(np.clip(d[:, 2], -1.0, 1.0))

        assert az_lo < az.min() and az_hi > az.max()
        assert el_lo < el.min() and el_hi > el.max()

    def test_follows_the_pattern_rather_than_a_fixed_field_of_view(self):
        # Units on one rig can sweep different widths, so bounds are derived per pattern.
        narrow = LidarFiringPattern(
            directions_sensor=np.array([[1.0, 0.0, 0.0], [0.966, 0.259, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([0, 1], dtype=np.int64),
            frame_duration_us=1,
        )
        wide = LidarFiringPattern(
            directions_sensor=np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([0, 1], dtype=np.int64),
            frame_duration_us=1,
        )

        narrow_span = narrow.fov().az_hi - narrow.fov().az_lo
        wide_span = wide.fov().az_hi - wide.fov().az_lo

        assert wide_span > narrow_span

    def test_a_full_circle_pattern_is_periodic(self):
        # A spinning unit's measured azimuth spans the whole circle; treating that as a
        # limited FOV puts a hard edge at the +/-pi seam.
        angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
        full_circle = LidarFiringPattern(
            directions_sensor=np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1).astype(
                np.float32
            ),
            timestamp_offsets_us=np.arange(64, dtype=np.int64),
            frame_duration_us=63,
        )

        fov = full_circle.fov()

        assert fov.periodic_az is True
        assert fov.az_lo == pytest.approx(-np.pi) and fov.az_hi == pytest.approx(np.pi)

    def test_limited_rear_arc_crossing_the_seam_is_refused(self):
        # Min/max would call this tiny rear arc a nearly full-circle sensor. The tiler cannot
        # represent a limited wrapping interval, so fail instead of silently changing its FOV.
        edge = LidarFiringPattern(
            directions_sensor=np.array([[-1.0, -1e-6, 0.0], [-1.0, 1e-6, 0.0]], dtype=np.float32),
            timestamp_offsets_us=np.array([0, 1], dtype=np.int64),
            frame_duration_us=1,
        )

        with pytest.raises(ValueError, match=r"crosses the \+/-180 deg seam"):
            edge.fov()


@dataclasses.dataclass
class _FakeTiling:
    """Stands in for UnstructuredLidarTilingData: the cache only needs `meta` and `.to`."""

    directions: object
    meta: dict

    def to(self, _device):
        return self


class TestTilingCache:
    """The tiling is a per-unit invariant, so it is built once and restamped per request."""

    def _patched(self):
        built = []

        def fake_build_tiling(directions, timestamps, fov, **frame_bounds):
            built.append((directions, timestamps, fov, frame_bounds))
            return _FakeTiling(directions=object(), meta={"frame_start_us": 0.0, "frame_end_us": 0.0})

        return built, mock.patch("nurec.utils.unstructured_lidar.build_tiling", fake_build_tiling)

    def test_every_request_shares_one_entry_per_device(self):
        # Nothing the request carries reaches the tiling, so requests cannot multiply entries.
        pattern = _pattern(n=6, span_us=70_000)
        built, patched = self._patched()

        with patched:
            build_unstructured_tiling(pattern, 0, "cpu")
            build_unstructured_tiling(pattern, 5_000, "cpu")
            build_unstructured_tiling(pattern, 9_999, "cpu")

        assert len(built) == 1
        assert len(pattern._tiling_cache) == 1
        # Bounds are gsplat's to infer from the offsets, which run 0..sweep.
        assert built[0][3] == {}

    def test_the_window_is_restamped_and_the_tensors_are_shared(self):
        pattern = _pattern(n=6, span_us=70_000)
        _, patched = self._patched()

        with patched:
            first = build_unstructured_tiling(pattern, 1_000, "cpu")
            second = build_unstructured_tiling(pattern, 7_000, "cpu")

        assert first.meta["frame_start_us"] == 1_000 and first.meta["frame_end_us"] == 71_000
        assert second.meta["frame_start_us"] == 7_000 and second.meta["frame_end_us"] == 77_000
        # Restamping must not mutate the cached copy, and must not rebuild the tensors.
        assert first.directions is second.directions
        assert first.meta is not second.meta


class TestWindowCheck:
    def test_accepts_either_the_sweep_or_the_frame(self):
        # The record's own brackets, and the rate the unit repeats at.
        pattern = _pattern(n=6, span_us=59_720, frame_duration_us=100_000)

        pattern.check_window(59_720)
        pattern.check_window(59_800)
        pattern.check_window(100_000)

    def test_refuses_a_window_that_is_neither(self):
        pattern = _pattern(n=6, span_us=59_720, frame_duration_us=100_000)

        with pytest.raises(ValueError, match="neither this unit's 59.7 ms sweep nor its 100.0 ms frame"):
            pattern.check_window(80_000)


@pytest.mark.parametrize("end_angle", [np.pi / 2.0, np.pi])
def test_build_model_free_matches_a_slerp_pose_oracle(end_angle):
    """Each ray uses the rigid pose at its own firing time, including non-small rotations."""
    pattern = LidarFiringPattern(
        directions_sensor=np.array([[1.0, 0.0, 0.0]] * 3, dtype=np.float32),
        timestamp_offsets_us=np.array([0, 250, 1_000], dtype=np.int64),
        frame_duration_us=1_000,
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

    with mock.patch("nurec.render.render.build_unstructured_tiling", return_value=mock.sentinel.tiling):
        bundle = LidarRayBundle.build_model_free(
            pattern,
            poses,
            FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            unique_sensor_idx=7,
        )

    rays = bundle.rendering_data.rays[0, :, 0]
    expected_origins = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], device="cuda")
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
    torch.testing.assert_close(
        bundle.rendering_data.rays_timestamps_us[0, :, 0, 0],
        torch.tensor([10_000, 10_250, 11_000], device="cuda"),
    )
    assert bundle.rendering_data.rays.shape == (1, 3, 1, 6)
    assert bundle.rendering_data.rays.device.type == "cuda"
    assert bundle.rendering_data.poses_tquat_startend.shape == (1, 2, 7)
    assert bundle.rendering_data.timestamps_startend_us.shape == (1, 2)
    assert bundle.rendering_data.timestamps_startend_us_cpu.device.type == "cpu"
    assert bundle.rendering_data.sensor_model_parameters == [None]
    assert bundle.rendering_data.unstructured_lidar_tiling is mock.sentinel.tiling
    assert bundle.frame_meta.unique_sensor_idx == 7
    assert bundle.frame_meta.unique_frame_idx == -1
    assert bundle.model_elements.shape == (3, 2)


def test_cropping_leaves_each_ray_where_the_uncropped_arc_would_put_it():
    """Cropping to the sweep still lands each ray at the pose of its own firing instant."""
    pattern = LidarFiringPattern(
        directions_sensor=np.array([[1.0, 0.0, 0.0]] * 2, dtype=np.float32),
        timestamp_offsets_us=np.array([100, 900], dtype=np.int64),
        frame_duration_us=1_000,
    )
    poses = PoseRange(
        start_pose_tquat_sensor_world=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        end_pose_tquat_sensor_world=torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        start_timestamp_us=20_000,
        end_timestamp_us=21_020,
    )
    built = []

    def fake_build_tiling(directions, timestamps, fov, **frame_bounds):
        built.append((timestamps, frame_bounds))
        return _FakeTiling(directions=directions, meta={"frame_start_us": 0.0, "frame_end_us": 0.0})

    with mock.patch("nurec.utils.unstructured_lidar.build_tiling", fake_build_tiling):
        bundle = LidarRayBundle.build_model_free(
            pattern,
            poses,
            FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        )

    expected_x = torch.tensor([10.0 * 100.0 / 1_020.0, 10.0 * 900.0 / 1_020.0], device="cuda")
    torch.testing.assert_close(bundle.rendering_data.rays[0, :, 0, 0], expected_x)
    torch.testing.assert_close(built[0][0], torch.tensor([100, 900]))
    # No bounds: the offsets already run 0..sweep, which is the window the rays sit in.
    assert built[0][1] == {}


def test_a_whole_frame_request_is_cropped_to_the_sweep_and_matches_a_sweep_request():
    """The frame and the sweep are different windows, and both must render the same rays."""
    sweep_us, frame_us = 59_720, 100_000
    pattern = LidarFiringPattern(
        directions_sensor=np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float32),
        timestamp_offsets_us=np.array([0, 20_000, 40_000, sweep_us], dtype=np.int64),
        frame_duration_us=frame_us,
    )
    half = np.deg2rad(1.5)
    start = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    def _end(duration_us: int) -> torch.Tensor:
        # The pose the sensor reaches at frame_start + duration, on one constant-rate arc.
        turned = half * duration_us / frame_us
        return torch.tensor(
            [1.5 * duration_us / frame_us, 0.0, 0.0, 0.0, 0.0, np.sin(turned), np.cos(turned)],
            dtype=torch.float32,
        )

    def _build(duration_us: int) -> LidarRayBundle:
        with mock.patch("nurec.render.render.build_unstructured_tiling", return_value=mock.sentinel.tiling):
            return LidarRayBundle.build_model_free(
                firing_pattern=pattern,
                lidar_to_world=PoseRange(start, _end(duration_us), 0, duration_us),
                world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            )

    whole, sweep = _build(frame_us), _build(sweep_us)

    torch.testing.assert_close(whole.rendering_data.rays, sweep.rendering_data.rays, rtol=0.0, atol=2e-5)
    np.testing.assert_array_equal(
        whole.rendering_data.rays_timestamps_us.cpu().numpy(),
        sweep.rendering_data.rays_timestamps_us.cpu().numpy(),
    )
