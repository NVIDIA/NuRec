# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json
import logging
import os
import stat
import subprocess
import sys

from pathlib import Path
from unittest import mock

import grpc
import numpy as np
import pytest
import torch

from click.testing import CliRunner

from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasReturn,
    AvailableEgoMasksReturn,
    CameraRenderOutput,
    CameraRenderProduct,
    CameraRenderRequest,
    CameraRenderReturn,
    HarmonizerSessionId,
    LidarFiringPatternReturn,
    LidarFloatingFilterSource,
    LidarOutputMode,
    LidarPacketLayout,
    LidarRenderRequest,
    LidarRenderReturn,
    LidarReturnType,
    ServerConfig,
)
from nurec.render.actors import ActorTracks
from nurec.utils.io.export.render_grpc import (
    _RESOURCE_EXHAUSTED_BASE_BACKOFF_S,
    _RESOURCE_EXHAUSTED_MAX_ATTEMPTS,
    _RESOURCE_EXHAUSTED_MAX_BACKOFF_S,
    _AdmissionLimiter,
    _atomic_save_npz,
    _build_dynamic_actor_poses,
    _build_lidar_render_filter,
    _cache_response_firing_patterns,
    _call_with_admission_retry,
    _claim_per_ray_output,
    _execute_renders,
    _extract_distance_colormap_vmax_m,
    _force_sequential_for_temporal_session,
    _maybe_create_temporal_harmonizer_session,
    _output_dir_for_repeat,
    _pin_requests_to_response_patterns,
    _render_grpc,
    _repeat_request_sequence,
    _save_firing_pattern_catalog,
    _write_distance_colormap_metadata,
    compose_sensor_to_world,
    firing_pattern_arrays,
    generate_lidar_request,
    per_ray_arrays,
    render_grpc,
    save_lidar_response,
    select_ego_mask_id,
)


def _se3(translation: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = translation
    return pose


def test_compose_sensor_to_world_applies_mount_extrinsics():
    # A roof lidar 1.94 m above the rig origin must end up 1.94 m above the rig in world
    # space too. Sending the bare rig pose (the nvbug 6107074 regression) would put the
    # sensor on the ground and collapse the point cloud.
    poses_rig_to_world = np.stack([_se3((10.0, 0.0, 0.0)), _se3((11.0, 0.0, 0.0))])
    pose_sensor_to_rig = _se3((1.10, 0.0, 1.94))

    composed = compose_sensor_to_world(poses_rig_to_world, np.eye(4), pose_sensor_to_rig)

    assert composed.shape == (2, 4, 4)
    np.testing.assert_allclose(composed[0][:3, 3], [11.10, 0.0, 1.94])
    np.testing.assert_allclose(composed[1][:3, 3], [12.10, 0.0, 1.94])


def test_compose_sensor_to_world_applies_rig_offset_between_rig_and_sensor():
    # The rig offset acts in the rig frame, i.e. between the rig pose and the extrinsics,
    # so it must shift the sensor for lidar exactly as it does for cameras.
    poses_rig_to_world = np.stack([_se3((0.0, 0.0, 0.0))] * 2)
    composed = compose_sensor_to_world(poses_rig_to_world, _se3((0.0, 2.0, 0.0)), _se3((1.0, 0.0, 1.94)))

    np.testing.assert_allclose(composed[0][:3, 3], [1.0, 2.0, 1.94])


def test_compose_sensor_to_world_rejects_malformed_poses():
    with pytest.raises(AssertionError):
        compose_sensor_to_world(_se3((0.0, 0.0, 0.0)), np.eye(4), np.eye(4))


def test_generate_lidar_request_leaves_output_and_layers_unset():
    request = generate_lidar_request(
        scene_id="scene",
        lidar_spec=None,
        poses=np.stack([np.eye(4), np.eye(4)]),
        frame_start_timestamp_us=1,
        frame_end_timestamp_us=2,
        dynamic_poses=[],
        render_filter=None,
    )

    assert not request.HasField("output_mode")
    assert not request.return_types


def test_generate_lidar_request_preserves_requested_return_layer_order():
    request = generate_lidar_request(
        scene_id="scene",
        lidar_spec=None,
        poses=np.stack([np.eye(4), np.eye(4)]),
        frame_start_timestamp_us=1,
        frame_end_timestamp_us=2,
        dynamic_poses=[],
        render_filter=None,
        return_types=[LidarReturnType.LIDAR_RETURN_FIRST, LidarReturnType.LIDAR_RETURN_STRONGEST],
    )

    assert list(request.return_types) == [
        LidarReturnType.LIDAR_RETURN_FIRST,
        LidarReturnType.LIDAR_RETURN_STRONGEST,
    ]


def test_output_dir_for_repeat_preserves_default_layout():
    assert _output_dir_for_repeat("render", 0, 1) == "render"


def test_output_dir_for_repeat_uses_numbered_subdirectories():
    assert _output_dir_for_repeat("render", 0, 30) == os.path.join("render", "repeat-000")
    assert _output_dir_for_repeat("render", 29, 30) == os.path.join("render", "repeat-029")


def test_repeat_request_sequence_reuses_same_request_objects():
    first_request = object()
    second_request = object()
    observed_requests = []

    async def render(request):
        observed_requests.append(request)
        return request

    async def collect_repetitions():
        return [
            (repeat_index, responses)
            async for repeat_index, responses in _repeat_request_sequence(
                render,
                [first_request, second_request],
                sequential=True,
                repeat_count=3,
            )
        ]

    repetitions = asyncio.run(collect_repetitions())

    assert [repeat_index for repeat_index, _ in repetitions] == [0, 1, 2]
    assert observed_requests == [first_request, second_request] * 3
    assert all(responses == [first_request, second_request] for _, responses in repetitions)


def _aio_rpc_error(code: grpc.StatusCode, details: str) -> grpc.aio.AioRpcError:
    """Build an AioRpcError with a given status (mirrors what a real gRPC channel raises)."""

    return grpc.aio.AioRpcError(
        code=code,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details=details,
    )


def _resource_exhausted() -> grpc.aio.AioRpcError:
    """The exact admission-shed error a `--multiproc-max-in-flight` server raises."""

    return _aio_rpc_error(grpc.StatusCode.RESOURCE_EXHAUSTED, "render admission limit reached (max_in_flight=1)")


class _AwaitingSheddingService:
    """Network-faithful fake unary render RPC for a `--multiproc-max-in-flight` server.

    Unlike a synchronous stub, it ``await``s (a network hop) BEFORE the admission decision, so
    concurrent callers actually overlap at the server -- reproducing the live shape where a fan-out
    burst all arrives at once and everything past ``max_in_flight`` is shed as RESOURCE_EXHAUSTED
    (unboundedly, at admission-timeout 0). A synchronous stub that raises before any await masks this
    (the excess callers never truly overlap). Tracks peak *arrival* concurrency (calls at the server
    at once) so a test can prove the client serialized down to one in flight.
    """

    def __init__(self, max_in_flight: int = 1, render_s: float = 0.01) -> None:
        self.max_in_flight = max_in_flight
        self.render_s = render_s
        self._admitted = 0
        self._arrivals = 0
        self.peak_arrivals = 0
        self.shed_count = 0
        self.completed: list[int] = []

    async def __call__(self, request: int) -> str:
        await asyncio.sleep(0)  # network hop: concurrent callers pile up here before admission
        self._arrivals += 1
        self.peak_arrivals = max(self.peak_arrivals, self._arrivals)
        try:
            if self._admitted >= self.max_in_flight:
                self.shed_count += 1
                raise _resource_exhausted()
            self._admitted += 1
            try:
                await asyncio.sleep(self.render_s)  # render
                self.completed.append(request)
                return f"rendered-{request}"
            finally:
                self._admitted -= 1
        finally:
            self._arrivals -= 1


def test_call_with_admission_retry_no_shed_single_call():
    calls = []

    async def _method(request):
        calls.append(request)
        return f"ok-{request}"

    result = asyncio.run(_call_with_admission_retry(_method, 7))

    assert result == "ok-7"
    assert calls == [7]  # a clean call is issued exactly once (no retry overhead)


def test_call_with_admission_retry_retries_resource_exhausted_then_succeeds():
    attempts = []

    async def _method(request):
        attempts.append(request)
        if len(attempts) < 3:
            raise _resource_exhausted()
        return "ok"

    result = asyncio.run(_call_with_admission_retry(_method, 1, base_backoff_s=0.0, max_backoff_s=0.0))

    assert result == "ok"
    assert len(attempts) == 3  # retried twice, then succeeded


def test_call_with_admission_retry_propagates_other_status_without_retry():
    attempts = []

    async def _method(request):
        attempts.append(request)
        raise _aio_rpc_error(grpc.StatusCode.UNAVAILABLE, "server down")

    with pytest.raises(grpc.aio.AioRpcError) as excinfo:
        asyncio.run(_call_with_admission_retry(_method, 1, base_backoff_s=0.0))

    assert excinfo.value.code() is grpc.StatusCode.UNAVAILABLE
    assert len(attempts) == 1  # non-admission statuses are not retried


def test_call_with_admission_retry_gives_up_after_max_attempts():
    attempts = []

    async def _method(request):
        attempts.append(request)
        raise _resource_exhausted()

    with pytest.raises(grpc.aio.AioRpcError) as excinfo:
        asyncio.run(_call_with_admission_retry(_method, 1, max_attempts=4, base_backoff_s=0.0))

    assert excinfo.value.code() is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert len(attempts) == 4  # bounded total attempts, then re-raise


def test_admission_limiter_rejects_nonpositive_explicit_limit():
    with pytest.raises(ValueError):
        _AdmissionLimiter(0)
    with pytest.raises(ValueError):
        _AdmissionLimiter(-1)


def test_admission_limiter_engages_adaptive_serializer_once_on_first_shed():
    async def _run():
        limiter = _AdmissionLimiter()
        assert limiter.active is None  # unbounded default until a shed is seen
        limiter.note_shed()
        engaged = limiter.active
        assert engaged is not None  # first shed installs the serializing semaphore
        limiter.note_shed()
        assert limiter.active is engaged  # idempotent: same semaphore instance
        return True

    assert asyncio.run(_run())


def test_admission_limiter_explicit_bound_disables_adaptive_override():
    async def _run():
        limiter = _AdmissionLimiter(2)
        bound = limiter.active
        assert bound is not None  # explicit bound is active from the start
        limiter.note_shed()
        assert limiter.active is bound  # a user bound suppresses the adaptive override
        return True

    assert asyncio.run(_run())


def test_admission_lottery_exhausts_without_the_adaptive_clamp():
    # Baseline reproducing the live failure (acceptance criterion #1): with NO shared clamp, many
    # contenders race for the single admission slot and, on a small retry budget, some caller loses
    # every attempt and exhausts (raises RESOURCE_EXHAUSTED). This is the shape the clamp must fix; it
    # is the failure the previous (bounded-shed, synchronous) fake could not surface.
    service = _AwaitingSheddingService(max_in_flight=1, render_s=0.01)
    requests = list(range(10))

    async def _run():
        return await asyncio.gather(
            *[
                _call_with_admission_retry(
                    service, r, limiter=None, max_attempts=5, base_backoff_s=0.001, max_backoff_s=0.01
                )
                for r in requests
            ],
            return_exceptions=True,
        )

    results = asyncio.run(_run())
    exhausted = [r for r in results if isinstance(r, grpc.aio.AioRpcError)]
    assert exhausted, "expected the un-clamped lottery to exhaust at least one caller"
    assert all(e.code() is grpc.StatusCode.RESOURCE_EXHAUSTED for e in exhausted)


def test_adaptive_clamp_prevents_exhaustion_under_heavy_contention():
    # Same heavy contention and the SAME small retry budget as the baseline above, but with a shared
    # adaptive limiter: the first shed installs a serializer so retries drain one at a time and every
    # caller completes instead of exhausting.
    service = _AwaitingSheddingService(max_in_flight=1, render_s=0.01)
    limiter = _AdmissionLimiter()
    requests = list(range(10))

    async def _run():
        return await asyncio.gather(
            *[
                _call_with_admission_retry(
                    service, r, limiter=limiter, max_attempts=5, base_backoff_s=0.001, max_backoff_s=0.01
                )
                for r in requests
            ]
        )

    results = asyncio.run(_run())
    assert sorted(results) == [f"rendered-{r}" for r in requests]  # all complete, none exhausted
    assert service.shed_count > 0  # heavy shedding actually occurred
    assert limiter.active is not None  # the adaptive serializer engaged


def test_execute_renders_completes_all_frames_under_heavy_admission_shedding():
    # End-to-end fan-out (the stock client shape): a max-in-flight-1 server that sheds every excess
    # concurrent render, unboundedly. With the default (unbounded) limiter the adaptive clamp must
    # engage on the first shed so every frame completes with no uncaught crash.
    service = _AwaitingSheddingService(max_in_flight=1, render_s=0.01)
    requests = list(range(8))
    limiter = _AdmissionLimiter()  # unbounded default

    results = asyncio.run(_execute_renders(service, requests, sequential_exec=False, limiter=limiter))

    assert results == [f"rendered-{i}" for i in requests]  # every frame rendered, order preserved
    assert sorted(service.completed) == requests
    assert service.shed_count > 0  # heavy shedding was actually exercised
    assert limiter.active is not None  # the adaptive serializer engaged


def test_execute_renders_preserves_order_and_parallelism_without_shedding():
    # No-shed control: against a non-admission server the fan-out is unchanged -- fully concurrent
    # (Trap 5: default must not reduce effective parallelism), never retried, and the adaptive clamp
    # never engages.
    service = _AwaitingSheddingService(max_in_flight=100)  # roomy: nothing sheds
    requests = list(range(5))
    limiter = _AdmissionLimiter()  # unbounded default

    results = asyncio.run(_execute_renders(service, requests, sequential_exec=False, limiter=limiter))

    assert results == [f"rendered-{i}" for i in requests]
    assert service.shed_count == 0  # nothing shed
    assert service.peak_arrivals >= 4  # the 4 post-warmup renders overlapped (parallelism intact)
    assert limiter.active is None  # adaptive clamp never engaged -> truly unbounded default


def test_execute_renders_explicit_max_concurrent_renders_serializes_from_the_start():
    # An explicit --max-concurrent-renders 1 bound serializes the fan-out from the start (only one
    # call reaches the server at a time), so even a max-in-flight-1 server that sheds unboundedly
    # never actually sheds.
    service = _AwaitingSheddingService(max_in_flight=1, render_s=0.01)
    requests = list(range(5))
    limiter = _AdmissionLimiter(1)  # explicit --max-concurrent-renders 1

    results = asyncio.run(_execute_renders(service, requests, sequential_exec=False, limiter=limiter))

    assert results == [f"rendered-{i}" for i in requests]
    assert service.peak_arrivals == 1  # explicit bound serialized: one call at the server at a time
    assert service.shed_count == 0  # so a max-in-flight-1 server never shed


def _available_ego_masks(*entries: tuple[str, str]) -> AvailableEgoMasksReturn:
    available = AvailableEgoMasksReturn()
    for rig_id, camera_id in entries:
        metadata = available.ego_mask_metadata.add()
        metadata.ego_mask_id.rig_config_id = rig_id
        metadata.ego_mask_id.camera_logical_id = camera_id
    return available


def _actor_track_and_interpolator(track_id: str, t_min: int, t_max: int):
    """A mock (cuboid_track, interpolator) pair for _build_dynamic_actor_poses tests.

    The interpolator returns identity poses so pose math stays trivial; the tests assert the
    inclusion decision and the (clamped) timestamps the interpolator is called with.
    """
    cuboid_track = mock.MagicMock()
    cuboid_track.tracks_id = [track_id]
    cuboid_track.tracks_timestamps_us = torch.tensor([t_min, t_max], dtype=torch.int64)
    interpolator = mock.MagicMock()
    interpolator.interpolate_to_timestamps.return_value = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    return cuboid_track, interpolator


@mock.patch("nurec.utils.io.export.render_grpc.lt")
def test_build_dynamic_actor_poses_partial_overlap_clamps_timestamps(mock_lt):
    # Frame window [0, 200] partially overlaps track [100, 500]: the actor is INCLUDED (overlap gate,
    # not full-containment) and its interpolation timestamps are clamped to [100, 200] so the
    # interpolator never extrapolates before track_min.
    cuboid_track, interpolator = _actor_track_and_interpolator("actor_a", 100, 500)
    poses = _build_dynamic_actor_poses(
        [cuboid_track],
        [interpolator],
        set(),
        0,
        200,
        0,
        demo_actor_transform=False,
        delta_matrices=np.zeros((1, 4, 4), dtype=np.float32),
    )
    assert [p["track_id"] for p in poses] == ["actor_a"]
    interpolator.interpolate_to_timestamps.assert_called_once_with([100, 200])


@mock.patch("nurec.utils.io.export.render_grpc.lt")
def test_build_dynamic_actor_poses_full_containment_uses_exact_timestamps(mock_lt):
    # Frame fully inside the track span: timestamps are unclamped (exact frame boundaries).
    cuboid_track, interpolator = _actor_track_and_interpolator("actor_a", 0, 1000)
    poses = _build_dynamic_actor_poses(
        [cuboid_track],
        [interpolator],
        set(),
        100,
        200,
        0,
        demo_actor_transform=False,
        delta_matrices=np.zeros((1, 4, 4), dtype=np.float32),
    )
    assert [p["track_id"] for p in poses] == ["actor_a"]
    interpolator.interpolate_to_timestamps.assert_called_once_with([100, 200])


@mock.patch("nurec.utils.io.export.render_grpc.lt")
def test_build_dynamic_actor_poses_no_overlap_excluded(mock_lt):
    # Frame entirely after the track span: actor is dropped, interpolator is never called.
    cuboid_track, interpolator = _actor_track_and_interpolator("actor_a", 0, 100)
    poses = _build_dynamic_actor_poses(
        [cuboid_track],
        [interpolator],
        set(),
        200,
        300,
        0,
        demo_actor_transform=False,
        delta_matrices=np.zeros((1, 4, 4), dtype=np.float32),
    )
    assert poses == []
    interpolator.interpolate_to_timestamps.assert_not_called()


@mock.patch("nurec.utils.io.export.render_grpc.lt")
def test_build_dynamic_actor_poses_removed_track_excluded(mock_lt):
    # Track listed in `remove` (edit-assets.json) is skipped regardless of overlap.
    cuboid_track, interpolator = _actor_track_and_interpolator("actor_a", 0, 1000)
    poses = _build_dynamic_actor_poses(
        [cuboid_track],
        [interpolator],
        {"actor_a"},
        100,
        200,
        0,
        demo_actor_transform=False,
        delta_matrices=np.zeros((1, 4, 4), dtype=np.float32),
    )
    assert poses == []


def test_actor_tracks_from_dict_cpu_stays_on_cpu_and_composes_delta():
    cpu = torch.device("cpu")
    poses = np.stack([_se3((0.0, 0.0, 0.0)), _se3((1.0, 0.0, 0.0))]).astype(np.float32)
    timestamps = np.array([0, 10], dtype=np.int64)
    serialized = CuboidTracks.Factory.from_numpy(
        ["a", "b"],
        [poses, poses],
        [timestamps, timestamps],
        ["car", "car"],
        [TrackFlags.CONTROLLABLE, TrackFlags(0)],
        [np.array([4.0, 2.0, 1.5], dtype=np.float32), np.ones(3, dtype=np.float32)],
        device=cpu,
    ).to_dict()
    delta = _se3((0.0, 2.0, 0.0)).astype(np.float32)
    delta[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

    actor_tracks = ActorTracks.from_dict_cpu(serialized)
    (track,) = actor_tracks._cuboid_tracks_list
    (dynamic_pose,) = _build_dynamic_actor_poses(
        actor_tracks._cuboid_tracks_list,
        actor_tracks._interpolator_list,
        set(),
        0,
        10,
        0,
        demo_actor_transform=True,
        delta_matrices=delta[None],
    )

    assert serialized["tracks_data"]["tracks_id"] == ["a", "b"]
    assert track.tracks_id == ["a"] and dynamic_pose["track_id"] == "a"
    assert track.tracks_poses.data.device == cpu
    torch.testing.assert_close(track.cuboids_dims, torch.tensor([[4.0, 2.0, 1.5]]))
    for key, keyframe in (("frame_start_actor_to_world", poses[0]), ("frame_end_actor_to_world", poses[1])):
        actual = dynamic_pose[key]
        assert actual.data.device == cpu
        torch.testing.assert_close(actual.matrix().reshape(4, 4), torch.from_numpy(keyframe @ delta), atol=1e-5, rtol=0)


def test_select_ego_mask_id_returns_none_without_requested_rig():
    available = _available_ego_masks(("rig_a", "front"))

    assert select_ego_mask_id(available, None, "front") is None


def test_select_ego_mask_id_matches_requested_rig_and_camera():
    available = _available_ego_masks(
        ("rig_a", "front"),
        ("rig_a", "left"),
        ("rig_b", "front"),
    )

    ego_mask_id = select_ego_mask_id(available, "rig_a", "left")

    assert ego_mask_id is not None
    assert ego_mask_id.rig_config_id == "rig_a"
    assert ego_mask_id.camera_logical_id == "left"


def test_select_ego_mask_id_rejects_missing_rig():
    available = _available_ego_masks(("rig_a", "front"))

    with pytest.raises(KeyError, match="rig_name='rig_missing' not found"):
        select_ego_mask_id(available, "rig_missing", "front")


def test_select_ego_mask_id_rejects_missing_camera_for_requested_rig():
    available = _available_ego_masks(("rig_a", "front"), ("rig_b", "left"))

    with pytest.raises(KeyError, match="No ego mask for rig_name='rig_a', camera_id='left'"):
        select_ego_mask_id(available, "rig_a", "left")


def test_render_grpc_deprecated_temporal_session_option_has_no_effect(caplog):
    runner = CliRunner()
    with (
        runner.isolated_filesystem(),
        mock.patch("nurec.utils.io.export.render_grpc._render_grpc", new_callable=mock.AsyncMock) as render,
        caplog.at_level(logging.WARNING, logger="nurec.utils.io.export.render_grpc"),
    ):
        result = runner.invoke(
            render_grpc,
            ["--artifact-path", "dummy", "--output-dir", "out", "--temporal-harmonizer-session"],
        )

    assert result.exit_code == 0, result.output
    assert render.await_args is not None
    assert render.await_args.args[-1] is False
    assert any("deprecated and has no effect" in record.message for record in caplog.records)


def test_render_grpc_rejects_apply_harmonizer_with_lidar():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            render_grpc,
            ["--artifact-path", "dummy", "--output-dir", "out", "--apply-harmonizer", "--lidar"],
        )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "not valid with --lidar" in str(result.exception)


def test_render_grpc_apply_harmonizer_requires_color_output():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            render_grpc,
            [
                "--artifact-path",
                "dummy",
                "--output-dir",
                "out",
                "--apply-harmonizer",
                "--output-type",
                "distance-colorized",
            ],
        )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "requires the color output type" in str(result.exception)


def test_render_grpc_requests_harmonizer_when_requested():
    runner = CliRunner()
    with (
        runner.isolated_filesystem(),
        mock.patch("nurec.utils.io.export.render_grpc._render_grpc", new_callable=mock.AsyncMock) as render,
    ):
        result = runner.invoke(
            render_grpc,
            ["--artifact-path", "dummy", "--output-dir", "out", "--apply-harmonizer"],
        )

    assert result.exit_code == 0, result.output
    assert render.await_args is not None
    assert render.await_args.args[-1] is True


@pytest.mark.parametrize(
    ("server_config", "expected_session_id"),
    [
        ({"enable_harmonizer_temporal": "true"}, "session-1"),
        ({"enable_harmonizer_temporal": "TRUE"}, "session-1"),
        ({"enable_harmonizer_temporal": "false"}, None),
        ({}, None),
    ],
)
def test_maybe_create_temporal_harmonizer_session_attaches_one_session_to_every_request(
    server_config, expected_session_id
):
    client_service = mock.MagicMock()
    client_service.get_server_config = mock.AsyncMock(return_value=ServerConfig(server_config=server_config))
    client_service.create_harmonizer_session = mock.AsyncMock(return_value=HarmonizerSessionId(session_id="session-1"))
    requests = [CameraRenderRequest(), CameraRenderRequest()]

    session_id = asyncio.run(_maybe_create_temporal_harmonizer_session(client_service, requests, repeat_count=1))

    assert session_id == expected_session_id
    client_service.get_server_config.assert_awaited_once()
    if expected_session_id is None:
        client_service.create_harmonizer_session.assert_not_awaited()
        assert [request.HasField("harmonizer_session_id") for request in requests] == [False, False]
    else:
        client_service.create_harmonizer_session.assert_awaited_once()
        assert [request.harmonizer_session_id for request in requests] == ["session-1", "session-1"]


def test_maybe_create_temporal_harmonizer_session_rejects_repeated_trajectory():
    client_service = mock.MagicMock()
    client_service.get_server_config = mock.AsyncMock(
        return_value=ServerConfig(server_config={"enable_harmonizer_temporal": "true"})
    )
    client_service.create_harmonizer_session = mock.AsyncMock()

    with pytest.raises(ValueError, match="repeat_count greater than 1"):
        asyncio.run(_maybe_create_temporal_harmonizer_session(client_service, [CameraRenderRequest()], repeat_count=2))

    client_service.create_harmonizer_session.assert_not_awaited()


@pytest.mark.parametrize(
    ("harmonizer_session_id", "sequential", "expected"),
    [("session-1", False, True), ("session-1", True, True), (None, False, False)],
)
def test_force_sequential_for_temporal_session(harmonizer_session_id, sequential, expected):
    assert _force_sequential_for_temporal_session(harmonizer_session_id, sequential) is expected


def _camera_render_return(distance_colormap_vmax_m: float | None) -> CameraRenderReturn:
    output = CameraRenderOutput(render_product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED)
    if distance_colormap_vmax_m is not None:
        output.distance_colormap_vmax_m = distance_colormap_vmax_m
    return CameraRenderReturn(camera_render_outputs=[output])


def test_extract_distance_colormap_vmax_m_explicit_request_value():
    """When the CLI passes --colorization-max-distance, the server echoes that value back."""
    response = _camera_render_return(37.0)

    assert _extract_distance_colormap_vmax_m(response) == pytest.approx(37.0)


def test_extract_distance_colormap_vmax_m_server_default():
    """When the CLI omits --colorization-max-distance, the server still resolves and returns a value."""
    response = _camera_render_return(12.5)

    assert _extract_distance_colormap_vmax_m(response) == pytest.approx(12.5)


def test_extract_distance_colormap_vmax_m_missing_field_returns_none():
    output = CameraRenderOutput(render_product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR)
    response = CameraRenderReturn(camera_render_outputs=[output])

    assert _extract_distance_colormap_vmax_m(response) is None


def test_write_distance_colormap_metadata_persists_resolved_value(tmp_path):
    responses = [_camera_render_return(37.0)]

    _write_distance_colormap_metadata(
        responses,
        [CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED],
        str(tmp_path),
        "front",
    )

    metadata_path = os.path.join(str(tmp_path), "distance_colorized", "front", "distance_colormap_vmax_m.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert metadata == pytest.approx({"distance_colormap_vmax_m": 37.0})


def test_write_distance_colormap_metadata_skips_when_product_not_requested(tmp_path):
    responses = [_camera_render_return(37.0)]

    _write_distance_colormap_metadata(
        responses,
        [CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
        str(tmp_path),
        "front",
    )

    assert not os.path.exists(os.path.join(str(tmp_path), "distance_colorized"))


def _filter(**overrides):
    """The filter a bare `--lidar` run builds, with only the named flags supplied."""
    kwargs = dict(
        raydrop_threshold=0.5,
        opacity_threshold=0.0,
        enable_distance_filter=None,
        distance_filter_threshold=None,
        neighbor_source=None,
        elevation_resolution_deg=None,
        azimuth_max_gap_deg=None,
    )
    kwargs.update(overrides)
    return _build_lidar_render_filter(**kwargs)


def test_build_lidar_render_filter_leaves_omitted_flags_unset():
    """Omitting a flag must not write the field: an unset field is what defers to the artifact
    config, so writing a default here would silently override it server-side."""
    render_filter = _filter()

    assert not render_filter.HasField("enable_distance_filter")
    assert not render_filter.HasField("distance_filter_threshold")
    assert not render_filter.HasField("neighbor_source")
    assert not render_filter.HasField("elevation_resolution_deg")
    assert not render_filter.HasField("azimuth_max_gap_deg")


def test_build_lidar_render_filter_angle_source_carries_the_angle_knobs():
    render_filter = _filter(
        enable_distance_filter=True,
        neighbor_source="angle",
        elevation_resolution_deg=0.055,
        azimuth_max_gap_deg=1.0,
    )

    assert render_filter.enable_distance_filter
    assert render_filter.neighbor_source == LidarFloatingFilterSource.LIDAR_FLOATING_FILTER_SOURCE_ANGLE
    assert render_filter.elevation_resolution_deg == pytest.approx(0.055)
    assert render_filter.azimuth_max_gap_deg == pytest.approx(1.0)


def test_build_lidar_render_filter_rangeview_source_maps_to_the_rangeview_enum():
    render_filter = _filter(neighbor_source="rangeview")

    assert render_filter.neighbor_source == LidarFloatingFilterSource.LIDAR_FLOATING_FILTER_SOURCE_RANGEVIEW


def test_render_grpc_forwards_the_angle_filter_flags():
    """The CLI flags must reach _render_grpc: the angle source is only reachable this way, since
    the server otherwise falls back to the artifact config's neighbor_source."""
    with mock.patch(
        "nurec.utils.io.export.render_grpc._render_grpc",
        side_effect=RuntimeError("stop after argument forwarding"),
    ) as render:
        CliRunner().invoke(
            render_grpc,
            [
                "--artifact-path",
                "unused",
                "--output-dir",
                "unused",
                "--lidar",
                "--lidar-distance-filter",
                "--lidar-distance-filter-neighbor-source",
                "angle",
                "--lidar-distance-filter-elevation-resolution-deg",
                "0.055",
                "--lidar-distance-filter-azimuth-max-gap-deg",
                "1.0",
            ],
        )

    assert render.called
    # Bind to the signature rather than searching the tuple: membership alone would still pass
    # if a value were forwarded into the wrong parameter.
    bound = inspect.signature(_render_grpc).bind(*render.call_args.args, **render.call_args.kwargs)
    assert bound.arguments["lidar_enable_distance_filter"] is True
    assert bound.arguments["lidar_distance_filter_neighbor_source"] == "angle"
    assert bound.arguments["lidar_distance_filter_elevation_resolution_deg"] == pytest.approx(0.055)
    assert bound.arguments["lidar_distance_filter_azimuth_max_gap_deg"] == pytest.approx(1.0)


def _per_ray_return(distances, valid, intensities=None, pattern=None, return_types=None) -> LidarRenderReturn:
    """A layer-major per-ray response with one packed LSB-first mask per layer."""
    distances = np.asarray(distances, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if distances.ndim == 1:
        distances = distances[None, :]
    if valid.ndim == 1:
        valid = valid[None, :]
    assert distances.shape == valid.shape
    num_layers, num_rays = distances.shape
    if return_types is None:
        assert num_layers == 1
        return_types = [LidarReturnType.LIDAR_RETURN_STRONGEST]

    result = LidarRenderReturn(
        num_rays=num_rays,
        num_return_layers=num_layers,
        return_types=return_types,
        ray_distances_m_buffer=distances.tobytes(),
        ray_valid_masks_packed=np.packbits(valid, axis=1, bitorder="little").tobytes(),
    )
    if intensities is not None:
        intensities = np.asarray(intensities, dtype=np.float32)
        if intensities.ndim == 1:
            intensities = intensities[None, :]
        assert intensities.shape == distances.shape
        result.ray_intensities_buffer = intensities.tobytes()
    if pattern is not None:
        result.firing_pattern.CopyFrom(pattern)
        result.firing_pattern_id = pattern.firing_pattern_id
    return result


def _firing_pattern(n_rays=6, n_columns=0, n_rows=0, with_packets=False) -> LidarFiringPatternReturn:
    result = LidarFiringPatternReturn(
        num_rays=n_rays,
        directions_sensor=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (n_rays, 1)).tobytes(),
        timestamp_offsets_us=np.arange(n_rays, dtype=np.int64).tobytes(),
        num_columns=n_columns,
        num_rows=n_rows,
        firing_pattern_id="sha256:test-pattern",
        frame_period_us=100_000,
    )
    if with_packets:
        result.packet_layout.CopyFrom(
            LidarPacketLayout(
                num_packets=2,
                ray_offsets_u32_le=np.array([0, n_rays // 2, n_rays], dtype="<u4").tobytes(),
            )
        )
    return result


def test_per_ray_arrays_unpacks_the_mask_aligned_with_the_other_arrays():
    # The mask arrives packed to keep a frame under gRPC's default receive limit; a consumer
    # needs it one entry per ray, and LSB-first is the order the proto documents.
    response = _per_ray_return([1.0, 2.0, 3.0], [True, False, True], [0.5, 0.0, 0.25])

    arrays = per_ray_arrays(response)

    np.testing.assert_array_equal(arrays["return_types"], [LidarReturnType.LIDAR_RETURN_STRONGEST])
    np.testing.assert_allclose(arrays["distance_m"], [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(arrays["valid"], [[True, False, True]])
    np.testing.assert_allclose(arrays["intensity"], [[0.5, 0.0, 0.25]])


def test_per_ray_arrays_trims_the_mask_to_the_declared_ray_count():
    # 3 rays pack into one byte, so the padding bits must not become phantom rays.
    arrays = per_ray_arrays(_per_ray_return([1.0, 2.0, 3.0], [True, True, True]))

    assert arrays["valid"].shape == (1, 3)


def test_per_ray_arrays_omits_intensity_when_the_model_predicts_none():
    assert "intensity" not in per_ray_arrays(_per_ray_return([1.0], [True]))


def test_per_ray_arrays_keeps_each_layer_padding_out_of_the_next_mask():
    response = _per_ray_return(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[True, False, True], [False, True, False]],
        return_types=[LidarReturnType.LIDAR_RETURN_FIRST, LidarReturnType.LIDAR_RETURN_STRONGEST],
    )

    arrays = per_ray_arrays(response)

    np.testing.assert_allclose(arrays["distance_m"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(arrays["valid"], [[True, False, True], [False, True, False]])


def test_per_ray_arrays_rejects_a_response_whose_arrays_disagree_on_length():
    # A truncated buffer would otherwise silently mis-address every ray after the cut.
    response = _per_ray_return([1.0, 2.0, 3.0], [True, False, True])
    response.num_rays = 4

    with pytest.raises(ValueError, match="1 layers x 4 rays"):
        per_ray_arrays(response)


@pytest.mark.parametrize("return_type", [LidarReturnType.LIDAR_RETURN_UNSPECIFIED, 99])
def test_per_ray_arrays_rejects_unspecified_and_unknown_return_types(return_type):
    """Reject invalid layer semantics before exporting plausible-looking ray data."""
    response = _per_ray_return([1.0], [True], return_types=[return_type])

    with pytest.raises(ValueError, match="unspecified or unsupported return type"):
        per_ray_arrays(response)


def test_firing_pattern_arrays_keeps_the_grid_when_the_sensor_has_one():
    arrays = firing_pattern_arrays(_firing_pattern(n_rays=6, n_columns=3, n_rows=2))

    assert arrays["directions_sensor"].shape == (6, 3)
    assert (arrays["n_columns"], arrays["n_rows"]) == (3, 2)


def test_firing_pattern_arrays_rejects_direction_count_that_disagrees_with_num_rays():
    pattern = _firing_pattern(n_rays=6)
    pattern.directions_sensor = pattern.directions_sensor[:-4]

    with pytest.raises(ValueError, match="declares 6 rays but carries 17 direction values"):
        firing_pattern_arrays(pattern)


def test_save_firing_pattern_catalog_keeps_every_pattern_id(tmp_path):
    first = _firing_pattern(n_rays=3)
    first.firing_pattern_id = "sha256:first"
    second = _firing_pattern(n_rays=3)
    second.firing_pattern_id = "sha256:second"

    _save_firing_pattern_catalog(str(tmp_path), {first.firing_pattern_id: first, second.firing_pattern_id: second})

    with open(tmp_path / "firing_patterns.json") as catalog_file:
        catalog = json.load(catalog_file)
    assert set(catalog) == {"sha256:first", "sha256:second"}
    for pattern_path in catalog.values():
        assert (tmp_path / pattern_path).is_file()


def test_save_firing_pattern_catalog_keeps_one_layout_and_skips_unchanged_patterns(tmp_path):
    singleton = _firing_pattern(n_rays=3)
    singleton.firing_pattern_id = "sha256:singleton"
    first = _firing_pattern(n_rays=3)
    first.firing_pattern_id = "sha256:first"
    second = _firing_pattern(n_rays=3)
    second.firing_pattern_id = "sha256:second"

    patterns = {singleton.firing_pattern_id: singleton}
    catalog = _save_firing_pattern_catalog(str(tmp_path), patterns)
    singleton_path = tmp_path / catalog[singleton.firing_pattern_id]
    assert singleton_path.is_file()
    assert not (tmp_path / "firing_pattern.npz").exists()
    original = singleton_path.stat().st_mtime_ns
    patterns.update({first.firing_pattern_id: first, second.firing_pattern_id: second})
    catalog = _save_firing_pattern_catalog(str(tmp_path), patterns, catalog)
    assert singleton_path.stat().st_mtime_ns == original
    with mock.patch("nurec.utils.io.export.render_grpc._atomic_output") as output:
        assert _save_firing_pattern_catalog(str(tmp_path), patterns, catalog) == catalog
    output.assert_not_called()
    with open(tmp_path / "firing_patterns.json") as catalog_file:
        assert json.load(catalog_file) == catalog
    assert set(catalog) == set(patterns)


def test_pattern_export_refuses_existing_run_and_preserves_legacy_files(tmp_path):
    legacy = tmp_path / "firing_pattern.npz"
    legacy.write_bytes(b"another run")
    with pytest.raises(ValueError, match="fresh output directory"):
        _claim_per_ray_output(str(tmp_path))
    assert legacy.read_bytes() == b"another run"
    run = tmp_path / "fresh"
    _claim_per_ray_output(str(run))
    with pytest.raises(ValueError, match="fresh output directory"):
        _claim_per_ray_output(str(run))
    pattern = _firing_pattern()
    _save_firing_pattern_catalog(str(run), {pattern.firing_pattern_id: pattern})
    before = (run / "firing_patterns.json").read_bytes()
    with pytest.raises(ValueError, match="another run"):
        _save_firing_pattern_catalog(str(run), {pattern.firing_pattern_id: pattern})
    assert (run / "firing_patterns.json").read_bytes() == before


@pytest.mark.parametrize("umask", [0o022, 0o002, 0o077])
def test_atomic_exports_respect_umask(tmp_path, umask):
    # The umask is process-wide; change it only in an isolated child, not the test runner.
    script = """
import os
import sys
from pathlib import Path
import numpy as np
from nurec.grpc.protos.sensorsim_pb2 import LidarFiringPatternReturn
from nurec.utils.io.export.render_grpc import _atomic_save_npz, _save_firing_pattern_catalog

root = Path(sys.argv[1])
mask = int(sys.argv[2])
os.umask(mask)
pattern = LidarFiringPatternReturn(
    num_rays=2,
    directions_sensor=np.array([[1, 0, 0], [1, 0, 0]], dtype="<f4").tobytes(),
    timestamp_offsets_us=np.array([0, 1], dtype="<i8").tobytes(),
    firing_pattern_id="pattern",
)
_atomic_save_npz(str(root / "frame.npz"), {"data": np.ones(2)})
_save_firing_pattern_catalog(str(root), {"pattern": pattern})
assert os.umask(mask) == mask
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(umask)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for relative_path in ("frame.npz", "firing_patterns.json", "firing_patterns/pattern_000.npz"):
        assert stat.S_IMODE((tmp_path / relative_path).stat().st_mode) == 0o666 & ~umask
    assert not list(tmp_path.rglob(".lidar-*"))


def test_atomic_npz_collision_does_not_overwrite_or_delete_another_file(tmp_path):
    collision = tmp_path / ".lidar-collision"
    collision.write_bytes(b"another writer")
    destination = tmp_path / "frame.npz"
    with mock.patch("nurec.utils.io.export.render_grpc.uuid4", return_value=mock.Mock(hex="collision")):
        with pytest.raises(FileExistsError):
            _atomic_save_npz(str(destination), {"data": np.ones(3)})
    assert collision.read_bytes() == b"another writer"
    assert not destination.exists()


def test_atomic_npz_failure_preserves_previous_file(tmp_path):
    path = tmp_path / "frame.npz"
    path.write_bytes(b"original")

    def fail(output, **arrays):
        output.write(b"partial")
        raise OSError("simulated disk error")

    with mock.patch("nurec.utils.io.export.render_grpc.np.savez_compressed", side_effect=fail):
        with pytest.raises(OSError, match="simulated disk error"):
            _atomic_save_npz(str(path), {"data": np.ones(3)})
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_catalog_publication_keeps_previous_catalog(tmp_path):
    first = _firing_pattern()
    catalog = _save_firing_pattern_catalog(str(tmp_path), {first.firing_pattern_id: first})
    old_bytes = (tmp_path / "firing_patterns.json").read_bytes()
    second = _firing_pattern()
    second.firing_pattern_id = "second"
    replace = os.replace

    def fail_catalog(source, destination):
        if str(destination).endswith("firing_patterns.json"):
            raise OSError("simulated rename error")
        replace(source, destination)

    with mock.patch("nurec.utils.io.export.render_grpc.os.replace", side_effect=fail_catalog):
        with pytest.raises(OSError, match="simulated rename error"):
            _save_firing_pattern_catalog(str(tmp_path), {second.firing_pattern_id: second}, catalog)
    assert (tmp_path / "firing_patterns.json").read_bytes() == old_bytes
    assert not list(tmp_path.glob(".lidar-*"))


class _PatternService:
    def __init__(self, patterns):
        self.patterns = patterns
        self.requests = []

    async def get_lidar_firing_pattern(self, request):
        self.requests.append(request)
        return self.patterns[request.firing_pattern_id]


def _response_with_pattern_id(pattern_id: str, n_rays: int = 3) -> LidarRenderReturn:
    response = _per_ray_return(np.ones(n_rays), np.ones(n_rays, dtype=bool))
    response.firing_pattern_id = pattern_id
    return response


def test_response_pattern_cache_fetches_once_per_distinct_id_without_mutating_requests():
    first = _firing_pattern(n_rays=3)
    first.firing_pattern_id = "sha256:A"
    second = _firing_pattern(n_rays=3)
    second.firing_pattern_id = "sha256:B"
    service = _PatternService({first.firing_pattern_id: first, second.firing_pattern_id: second})
    requests = [LidarRenderRequest() for _ in range(3)]
    responses = [
        _response_with_pattern_id("sha256:A"),
        _response_with_pattern_id("sha256:B"),
        _response_with_pattern_id("sha256:A"),
    ]
    cache = {}

    asyncio.run(
        _cache_response_firing_patterns(
            service,
            scene_id="scene",
            lidar_id="lidar",
            requests=requests,
            responses=responses,
            patterns_by_id=cache,
        )
    )
    asyncio.run(
        _cache_response_firing_patterns(
            service,
            scene_id="scene",
            lidar_id="lidar",
            requests=requests,
            responses=responses,
            patterns_by_id=cache,
        )
    )

    assert [request.firing_pattern_id for request in service.requests] == ["sha256:A", "sha256:B"]
    assert all(not request.firing_pattern_id for request in requests)
    _pin_requests_to_response_patterns(requests, responses)
    assert [request.firing_pattern_id for request in requests] == ["sha256:A", "sha256:B", "sha256:A"]
    assert set(cache) == {"sha256:A", "sha256:B"}


def test_response_pattern_cache_reuses_matching_embedded_geometry():
    pattern = _firing_pattern(n_rays=3)
    pattern.firing_pattern_id = "sha256:embedded"
    response = _per_ray_return(np.ones(3), np.ones(3, dtype=bool), pattern=pattern)
    service = _PatternService({})
    cache = {}

    asyncio.run(
        _cache_response_firing_patterns(
            service,
            scene_id="scene",
            lidar_id="lidar",
            requests=[LidarRenderRequest()],
            responses=[response],
            patterns_by_id=cache,
        )
    )

    assert not service.requests
    assert cache["sha256:embedded"] == pattern


@pytest.mark.parametrize("embedded_index", [0, 1])
def test_response_pattern_cache_reuses_embedded_geometry_across_batch(embedded_index):
    embedded = _firing_pattern(n_rays=3)
    embedded.firing_pattern_id = "sha256:A"
    missing = _firing_pattern(n_rays=3)
    missing.firing_pattern_id = "sha256:B"
    service = _PatternService({missing.firing_pattern_id: missing})
    requests = [LidarRenderRequest() for _ in range(3)]
    responses = [
        _response_with_pattern_id("sha256:A"),
        _response_with_pattern_id("sha256:A"),
        _response_with_pattern_id("sha256:B"),
    ]
    responses[embedded_index].firing_pattern.CopyFrom(embedded)
    cache = {}

    asyncio.run(
        _cache_response_firing_patterns(
            service,
            scene_id="scene",
            lidar_id="lidar",
            requests=requests,
            responses=responses,
            patterns_by_id=cache,
        )
    )

    assert [request.firing_pattern_id for request in service.requests] == ["sha256:B"]
    assert cache == {"sha256:A": embedded, "sha256:B": missing}
    assert all(not request.firing_pattern_id for request in requests)


def test_response_pattern_cache_rejects_a_response_that_changes_a_pinned_id():
    request = LidarRenderRequest(firing_pattern_id="sha256:A")

    with pytest.raises(ValueError, match="request pinned"):
        asyncio.run(
            _cache_response_firing_patterns(
                _PatternService({}),
                scene_id="scene",
                lidar_id="lidar",
                requests=[request],
                responses=[_response_with_pattern_id("sha256:B")],
                patterns_by_id={},
            )
        )


def test_firing_pattern_arrays_writes_no_grid_when_the_sensor_has_none():
    # Zero on the wire means "no addressing"; writing zeros would read as a real grid.
    arrays = firing_pattern_arrays(_firing_pattern(n_rays=6))

    assert "n_columns" not in arrays and "n_rows" not in arrays


def test_firing_pattern_arrays_preserves_identity_period_and_optional_packets():
    arrays = firing_pattern_arrays(_firing_pattern(n_rays=6, with_packets=True))

    assert str(arrays["firing_pattern_id"]) == "sha256:test-pattern"
    assert int(arrays["frame_period_us"]) == 100_000
    np.testing.assert_array_equal(arrays["packet_ray_offsets"], [0, 3, 6])


def test_firing_pattern_arrays_expands_exact_packet_timestamps_for_ray_consumers():
    pattern = _firing_pattern(n_rays=6, with_packets=True)
    pattern.timestamp_offsets_us = b""
    pattern.packet_layout.timestamp_offsets_us = np.array([10, 20], dtype=np.int64).tobytes()

    arrays = firing_pattern_arrays(pattern)

    np.testing.assert_array_equal(arrays["packet_timestamp_offsets_us"], [10, 20])
    np.testing.assert_array_equal(arrays["timestamp_offsets_us"], [10, 10, 10, 20, 20, 20])


def test_firing_pattern_arrays_rejects_two_timestamp_representations():
    pattern = _firing_pattern(n_rays=6, with_packets=True)
    pattern.packet_layout.timestamp_offsets_us = np.array([10, 20], dtype=np.int64).tobytes()

    with pytest.raises(ValueError, match="either per-ray or per-packet timestamps"):
        firing_pattern_arrays(pattern)


def test_per_ray_arrays_preserves_the_selected_pattern_id_without_geometry():
    response = _per_ray_return([1.0], [True])
    response.firing_pattern_id = "sha256:test-pattern"

    assert str(per_ray_arrays(response)["firing_pattern_id"]) == "sha256:test-pattern"


def test_save_lidar_response_writes_every_ray_to_npz(tmp_path):
    path = tmp_path / "frame.npz"

    asyncio.run(save_lidar_response(str(path), _per_ray_return([1.0, 9.0, 3.0], [True, False, True]), format="per-ray"))

    with np.load(path) as data:
        np.testing.assert_allclose(data["distance_m"], [[1.0, 9.0, 3.0]])
        np.testing.assert_array_equal(data["valid"], [[True, False, True]])
        np.testing.assert_array_equal(data["return_types"], [LidarReturnType.LIDAR_RETURN_STRONGEST])


def test_save_lidar_response_embeds_the_pattern_when_the_server_sent_one(tmp_path):
    path = tmp_path / "frame.npz"
    response = _per_ray_return([1.0] * 6, [True] * 6, pattern=_firing_pattern(6, 3, 2))

    asyncio.run(save_lidar_response(str(path), response, format="per-ray"))

    with np.load(path) as data:
        assert data["directions_sensor"].shape == (6, 3)
        assert int(data["n_columns"]) == 3


def test_save_lidar_response_explains_an_empty_per_ray_reply(tmp_path):
    # The likely cause is a request that never asked for per-ray output, so say so.
    with pytest.raises(ValueError, match="LIDAR_OUTPUT_PER_RAY"):
        asyncio.run(save_lidar_response(str(tmp_path / "f.npz"), LidarRenderReturn(), format="per-ray"))


@pytest.mark.parametrize("mask", [b"\x01", b"\x01\x00\x00"])
def test_firing_pattern_arrays_rejects_wrong_scan_mask_size(mask):
    pattern = _firing_pattern(9)
    pattern.packet_layout.CopyFrom(
        LidarPacketLayout(
            num_packets=9,
            ray_offsets_u32_le=np.arange(10, dtype="<u4").tobytes(),
            starts_new_scan_mask_packed=mask,
        )
    )
    with pytest.raises(ValueError, match="exactly 2 bytes"):
        firing_pattern_arrays(pattern)


def test_firing_pattern_arrays_decodes_literal_scan_mask():
    pattern = _firing_pattern(9)
    pattern.packet_layout.CopyFrom(
        LidarPacketLayout(
            num_packets=9,
            ray_offsets_u32_le=np.arange(10, dtype="<u4").tobytes(),
            starts_new_scan_mask_packed=bytes([0b00000101, 0b00000001]),
        )
    )
    np.testing.assert_array_equal(np.flatnonzero(firing_pattern_arrays(pattern)["packet_starts_new_scan"]), [0, 2, 8])


@pytest.mark.parametrize("columns,rows", [(2, 2), (3, 0), (0, 2)])
def test_firing_pattern_arrays_rejects_inconsistent_grid(columns, rows):
    with pytest.raises(ValueError, match="grid dimensions"):
        firing_pattern_arrays(_firing_pattern(6, columns, rows))


def test_per_ray_arrays_decodes_literal_independently_padded_layers():
    response = _per_ray_return(
        [[1, 2, 3], [4, 5, 6]],
        [[True] * 3] * 2,
        return_types=[LidarReturnType.LIDAR_RETURN_FIRST, LidarReturnType.LIDAR_RETURN_STRONGEST],
    )
    response.ray_valid_masks_packed = bytes([0b101, 0b010])
    arrays = per_ray_arrays(response)
    np.testing.assert_array_equal(arrays["valid"], [[True, False, True], [False, True, False]])
    np.testing.assert_array_equal(arrays["return_type_names"], ["LIDAR_RETURN_FIRST", "LIDAR_RETURN_STRONGEST"])


@pytest.mark.parametrize(
    "status",
    [
        grpc.StatusCode.NOT_FOUND,
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.FAILED_PRECONDITION,
        grpc.StatusCode.CANCELLED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
    ],
)
def test_response_cache_preserves_transport_failures_and_explains_missing_geometry(status):
    error = grpc.aio.AioRpcError(status, (), (), details="lookup failed")
    service = mock.Mock(get_lidar_firing_pattern=mock.AsyncMock(side_effect=error))
    expected = (
        ValueError if status in (grpc.StatusCode.NOT_FOUND, grpc.StatusCode.INVALID_ARGUMENT) else grpc.aio.AioRpcError
    )
    cache = {}
    with (
        mock.patch("nurec.utils.io.export.render_grpc.asyncio.sleep", new_callable=mock.AsyncMock) as sleep,
        pytest.raises(expected) as caught,
    ):
        asyncio.run(
            _cache_response_firing_patterns(
                service,
                scene_id="scene",
                lidar_id="lidar",
                requests=[LidarRenderRequest()],
                responses=[_response_with_pattern_id("pattern")],
                patterns_by_id=cache,
            )
        )
    service.get_lidar_firing_pattern.assert_awaited_once()
    sleep.assert_not_awaited()
    assert not cache
    if expected is grpc.aio.AioRpcError:
        assert caught.value is error


@pytest.mark.parametrize("succeeds", [True, False])
def test_response_cache_uses_bounded_capacity_retries(succeeds):
    pattern = _firing_pattern(n_rays=3)
    error = _aio_rpc_error(grpc.StatusCode.RESOURCE_EXHAUSTED, "no spare backends")
    lookup = mock.AsyncMock(side_effect=[error, error, pattern] if succeeds else error)
    service = mock.Mock(get_lidar_firing_pattern=lookup)
    cache = {}
    response = _response_with_pattern_id(pattern.firing_pattern_id)
    with mock.patch("nurec.utils.io.export.render_grpc.asyncio.sleep", new_callable=mock.AsyncMock) as sleep:
        task = _cache_response_firing_patterns(
            service,
            scene_id="scene",
            lidar_id="lidar",
            requests=[LidarRenderRequest()],
            responses=[response],
            patterns_by_id=cache,
        )
        if succeeds:
            asyncio.run(task)
            assert cache == {pattern.firing_pattern_id: pattern}
        else:
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                asyncio.run(task)
            assert caught.value is error
            assert not cache
    attempts = 3 if succeeds else _RESOURCE_EXHAUSTED_MAX_ATTEMPTS
    assert lookup.await_count == attempts
    assert sleep.await_args_list == [
        mock.call(min(_RESOURCE_EXHAUSTED_BASE_BACKOFF_S * 2**index, _RESOURCE_EXHAUSTED_MAX_BACKOFF_S))
        for index in range(attempts - 1)
    ]
    for call in lookup.await_args_list:
        request = call.args[0]
        assert (request.scene_id, request.lidar_id, request.firing_pattern_id) == (
            "scene",
            "lidar",
            pattern.firing_pattern_id,
        )


@pytest.mark.parametrize("mismatch", ["id", "count"])
def test_embedded_geometry_must_match_frame(tmp_path, mismatch):
    response = _per_ray_return([1] * 6, [True] * 6, pattern=_firing_pattern(6))
    if mismatch == "id":
        response.firing_pattern.firing_pattern_id = "different"
    else:
        response.firing_pattern.num_rays = 5
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(save_lidar_response(str(tmp_path / "f.npz"), response, format="per-ray"))
    assert not (tmp_path / "f.npz").exists()


@pytest.mark.parametrize("lidar_format", ["bin", "ply", "per-ray"])
def test_render_grpc_lidar_orchestration_requests_modes_and_exports_replay_metadata(tmp_path, lidar_format):
    prefix = "nurec.utils.io.export.render_grpc"
    artifact = mock.Mock(
        scene_id="scene", rig_trajectories={}, sequence_tracks={"sequence": {"tracks_data": {"tracks_id": []}}}
    )
    trajectory = mock.Mock(lidars_frame_timestamps_us={"lidar@scene": torch.tensor([[1000, 1100], [1100, 1100]])})
    rigs = mock.Mock(
        rig_trajectories=[trajectory], lidar_calibrations={"lidar@scene": mock.Mock(T_sensor_rig=torch.eye(4))}
    )
    base_x = 12345678.123456
    poses = np.stack([_se3((base_x, 0, 0)), _se3((base_x + 1, 0, 0))])
    patterns = {"A": _firing_pattern(3), "B": _firing_pattern(3)}
    for key, pattern in patterns.items():
        pattern.firing_pattern_id = key
    service = _PatternService(patterns)
    service.get_available_cameras = mock.AsyncMock(return_value=AvailableCamerasReturn())
    observed = []

    async def render(request):
        observed.append(LidarRenderRequest.FromString(request.SerializeToString()))
        if lidar_format == "per-ray":
            return _response_with_pattern_id("A" if request.frame_start_us == 1000 else "B")
        return LidarRenderReturn(num_points=1, point_xyzs_buffer=np.ones(3, dtype="<f4").tobytes())

    service.render_lidar = render
    with (
        mock.patch(f"{prefix}.Artifact.discover_from_glob", return_value=[artifact]),
        mock.patch(f"{prefix}.RigTrajectories.from_dict", return_value=rigs),
        mock.patch(
            f"{prefix}.ActorTracks.from_dict_cpu", return_value=mock.Mock(_cuboid_tracks_list=[], _interpolator_list=[])
        ),
        mock.patch(f"{prefix}.ncore_transformations.PoseInterpolator") as interpolator,
        mock.patch(f"{prefix}.grpc.aio.insecure_channel"),
        mock.patch(f"{prefix}.SensorsimServiceStub", return_value=service),
    ):
        interpolator.return_value.interpolate_to_timestamps.return_value = poses
        result = CliRunner().invoke(
            render_grpc,
            [
                "--artifact-path",
                "unused",
                "--output-dir",
                str(tmp_path),
                "--lidar",
                "--lidar-id",
                "lidar",
                "--lidar-format",
                lidar_format,
                "--repeat-count",
                "2",
                "--rig-translation-offset",
                "0",
                "2",
                "0",
            ],
        )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert len(observed) == 4
    mode = (
        LidarOutputMode.LIDAR_OUTPUT_PER_RAY if lidar_format == "per-ray" else LidarOutputMode.LIDAR_OUTPUT_POINT_CLOUD
    )
    assert all(
        request.HasField("output_mode") and request.output_mode == mode and not request.return_types
        for request in observed
    )
    assert [request.frame_end_us for request in observed] == [1100, 1101, 1100, 1101]
    suffix = "npz" if lidar_format == "per-ray" else lidar_format
    assert (tmp_path / "repeat-000" / f"000000.{suffix}").is_file()
    assert (tmp_path / "repeat-001" / f"000001.{suffix}").is_file()
    if lidar_format != "per-ray":
        assert not service.requests
        return
    assert [request.firing_pattern_id for request in observed] == ["", "", "A", "B"]
    assert [request.firing_pattern_id for request in service.requests] == ["A", "B"]
    catalog = json.loads((tmp_path / "firing_patterns.json").read_text())
    assert set(catalog) == {"A", "B"}
    assert all((tmp_path / path).is_file() for path in catalog.values())
    assert not (tmp_path / "firing_pattern.npz").exists()
    assert not (tmp_path / "repeat-001" / "firing_patterns.json").exists()
    with np.load(tmp_path / "repeat-001" / "000001.npz") as frame:
        assert frame["frame_start_us"] == 1100
        assert frame["frame_end_us"] == 1101
        assert frame["T_lidar_to_world_start"].dtype == np.float64
        # Store what was actually sent, including the protobuf pose's float32 rounding.
        assert frame["T_lidar_to_world_start"][0, 3] == observed[3].sensor_pose.start_pose.vec.x
        assert frame["T_lidar_to_world_start"][1, 3] == 2
        assert frame["T_lidar_to_world_end"][0, 3] == observed[3].sensor_pose.end_pose.vec.x
        assert str(frame["scene_id"]) == "scene"
        assert str(frame["lidar_id"]) == "lidar"
        assert str(frame["firing_pattern_id"]) == "B"


def test_per_ray_cli_refuses_nonempty_output_before_overwriting_settings(tmp_path):
    marker = tmp_path / "render_grpc_cli_args.json"
    marker.write_text("previous settings")
    result = CliRunner().invoke(
        render_grpc,
        ["--artifact-path", "unused", "--output-dir", str(tmp_path), "--lidar", "--lidar-format", "per-ray"],
    )
    assert result.exit_code == 2
    assert "Usage:" in result.output
    assert "Error: Per-ray export requires a fresh, empty --output-dir" in result.output
    assert "Traceback" not in result.output
    assert marker.read_text() == "previous settings"
    assert list(tmp_path.iterdir()) == [marker]


@pytest.mark.parametrize("repeat_count", [1, 2])
def test_per_ray_cli_losing_directory_claim_preserves_winner_settings(tmp_path, repeat_count):
    """A concurrent winner keeps its settings even after the loser's empty-directory check."""
    settings = [tmp_path / "render_grpc_cli_args.json"]
    if repeat_count > 1:
        settings.extend(tmp_path / f"repeat-{index:03d}" / "render_grpc_cli_args.json" for index in range(repeat_count))
    original_iterdir = Path.iterdir

    def stale_empty_check(directory):
        """Let another export claim and populate the directory after this run saw it empty."""
        assert directory == tmp_path
        assert not list(original_iterdir(directory))
        _claim_per_ray_output(str(directory))
        for path in settings:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("winning run settings")
        return iter(())

    with (
        mock.patch.object(Path, "iterdir", autospec=True, side_effect=stale_empty_check),
        mock.patch("nurec.utils.io.export.render_grpc.Artifact.discover_from_glob") as discover,
    ):
        result = CliRunner().invoke(
            render_grpc,
            [
                "--artifact-path",
                "unused",
                "--output-dir",
                str(tmp_path),
                "--lidar",
                "--lidar-format",
                "per-ray",
                "--repeat-count",
                str(repeat_count),
            ],
        )

    assert all(path.read_text() == "winning run settings" for path in settings)
    assert result.exit_code == 2
    assert "Error: Per-ray export requires a fresh output directory" in result.output
    assert "Traceback" not in result.output
    discover.assert_not_called()


def test_per_ray_cli_hands_off_its_claim_without_disabling_direct_runner_protection(tmp_path):
    """Only the CLI invocation that acquired the reservation skips the runner's claim."""
    with mock.patch("nurec.utils.io.export.render_grpc._render_grpc", new_callable=mock.AsyncMock) as render:
        result = CliRunner().invoke(
            render_grpc,
            ["--artifact-path", "unused", "--output-dir", str(tmp_path), "--lidar", "--lidar-format", "per-ray"],
        )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "firing_patterns").is_dir()
    assert render.await_args is not None
    kwargs = dict(render.await_args.kwargs)
    assert kwargs.pop("per_ray_output_claimed") is True
    with pytest.raises(ValueError, match="fresh output directory"):
        asyncio.run(_render_grpc(*render.await_args.args, **kwargs))
