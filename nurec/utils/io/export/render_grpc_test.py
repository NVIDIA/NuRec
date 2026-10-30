# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json
import logging
import os

from unittest import mock

import grpc
import numpy as np
import pytest
import torch

from click.testing import CliRunner

from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.grpc.protos.sensorsim_pb2 import (
    AvailableEgoMasksReturn,
    CameraRenderOutput,
    CameraRenderProduct,
    CameraRenderRequest,
    CameraRenderReturn,
    HarmonizerSessionId,
    LidarFloatingFilterSource,
    ServerConfig,
)
from nurec.render.actors import ActorTracks
from nurec.utils.io.export.render_grpc import (
    _AdmissionLimiter,
    _build_dynamic_actor_poses,
    _build_lidar_render_filter,
    _call_with_admission_retry,
    _execute_renders,
    _extract_distance_colormap_vmax_m,
    _force_sequential_for_temporal_session,
    _maybe_create_temporal_harmonizer_session,
    _output_dir_for_repeat,
    _render_grpc,
    _repeat_request_sequence,
    _write_distance_colormap_metadata,
    compose_sensor_to_world,
    render_grpc,
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
