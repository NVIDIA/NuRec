# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

from bisect import bisect_left
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
import torch

import nurec.models.gaussians.vehicle_light as vehicle_light_module

from nurec.libs.gaussian_collective_payload_ops.cross_stream_test_utils import CrossStreamAllocationTestHarness
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.models.gaussians.initializations import NoPointsFoundException
from nurec.models.gaussians.vehicle_light import (
    VehicleLightRigidGaussianModel,
    VehicleLightTracksInitialization,
    _aggregate_track_observations,
    _resolve_vehicle_light_mask_inputs,
    _TrackFrameSlotCache,
    nearest_track_frame_slots,
    propagate_temporal_colors,
)


@pytest.fixture
def device() -> Iterator[str]:
    """Exercise production on CUDA; the scalar oracle explicitly reads CPU values."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with torch.device("cuda"):
        yield "cuda"


@pytest.mark.parametrize("cpu_input", range(3))
@pytest.mark.parametrize("empty", [False, True])
def test_nearest_track_frame_slots_rejects_cpu_before_cuda_dispatch(cpu_input: int, empty: bool, device: str) -> None:
    timestamps, packinfo, track_ids = _lookup_inputs()
    inputs = [track_ids, timestamps, packinfo]
    if empty:
        inputs[0] = inputs[0][:0]
    inputs[cpu_input] = inputs[cpu_input].cpu()
    with mock.patch.object(torch.cuda, "current_stream") as current_stream:
        with pytest.raises(RuntimeError, match="must be CUDA tensors"):
            nearest_track_frame_slots(2, *inputs)
        current_stream.assert_not_called()


@pytest.mark.parametrize("empty", [False, True])
def test_nearest_track_frame_slots_rejects_all_cpu_inputs(empty: bool) -> None:
    timestamps, packinfo, track_ids = _lookup_inputs("cpu")
    with mock.patch.object(torch.cuda, "current_stream") as current_stream:
        with pytest.raises(RuntimeError, match="must be CUDA tensors"):
            nearest_track_frame_slots(2, track_ids[:0] if empty else track_ids, timestamps, packinfo)
        current_stream.assert_not_called()


def test_nearest_track_frame_slots_uses_track_relative_indices(device: str) -> None:
    timestamps = torch.tensor([10, 20, 40, 100, 130], dtype=torch.int64)
    packinfo = torch.tensor([[0, 3], [3, 2]], dtype=torch.int64)
    track_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    slots = nearest_track_frame_slots(28, track_ids, timestamps, packinfo)

    torch.testing.assert_close(slots, torch.tensor([1, 1, 0, 0]))


def test_nearest_track_frame_slots_clamps_and_keeps_earlier_midpoint_ties(device: str) -> None:
    timestamps = torch.tensor([10, 20, 40], dtype=torch.int64)
    packinfo = torch.tensor([[0, 3]], dtype=torch.int64)
    track_ids = torch.tensor([0], dtype=torch.int32)

    for timestamp_us, expected_slot in ((-1, 0), (10, 0), (15, 0), (16, 1), (20, 1), (30, 1), (31, 2), (100, 2)):
        slots = nearest_track_frame_slots(timestamp_us, track_ids, timestamps, packinfo)
        torch.testing.assert_close(slots, torch.tensor([expected_slot]))


def _assert_frame_slots(
    timestamp_us: int,
    track_ids: list[int],
    timestamps: list[int],
    packinfo: list[list[int]],
    expected: list[int],
) -> None:
    """Assert frame slots for compact Python-list inputs."""

    slots = nearest_track_frame_slots(
        timestamp_us,
        torch.tensor(track_ids, dtype=torch.int32),
        torch.tensor(timestamps),
        torch.tensor(packinfo),
    )
    torch.testing.assert_close(slots, torch.tensor(expected))


def test_nearest_track_frame_slots_handles_multiple_tracks_without_host_iteration(device: str) -> None:
    _assert_frame_slots(4, [2, 1, 0, 1, 0, 2], [1, 3, 5, 2, 4, 6], [[0, 3], [3, 2], [5, 1]], [0, 1, 1, 1, 1, 0])


def test_nearest_track_frame_slots_vectorization_respects_reordered_packinfo(device: str) -> None:
    _assert_frame_slots(5, [0, 1, 2], [0, 2, 4, 6, 8, 10], [[4, 2], [0, 2], [2, 2]], [0, 1, 0])


def _lookup_inputs(device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact two-track timestamps, packinfo, and Gaussian IDs."""

    return (
        torch.tensor([1, 3, 6, 8], dtype=torch.int64, device=device),
        torch.tensor([[0, 2], [2, 2]], dtype=torch.int64, device=device),
        torch.tensor([0, 1, 0], dtype=torch.int32, device=device),
    )


def _reference_track_frame_slots(
    timestamp_us: int,
    track_ids: torch.Tensor,
    timestamps: torch.Tensor,
    packinfo: torch.Tensor,
) -> torch.Tensor:
    """Scalar reference with the original searchsorted boundary/duplicate semantics."""

    expected: list[int] = []
    for track_id in track_ids.cpu().tolist():
        start, count = packinfo[track_id].cpu().tolist()
        timeline = timestamps[start : start + count].cpu().tolist()
        if not timeline:
            expected.append(0)
            continue
        right = min(bisect_left(timeline, timestamp_us), count - 1)
        left = max(right - 1, 0)
        expected.append(left if abs(timestamp_us - timeline[left]) <= abs(timeline[right] - timestamp_us) else right)
    return torch.tensor(expected, dtype=torch.long, device=track_ids.device)


@pytest.mark.parametrize("offset", [0, 2**53 + 1])
def test_cached_track_frame_slots_match_reference_for_ragged_and_duplicate_timestamps(offset: int, device: str) -> None:
    timestamps = torch.tensor([offset + value for value in [1, 2, 4, 4, 6, 8, 8, 10]], dtype=torch.int64, device=device)
    packinfo = torch.tensor([[0, 0], [0, 1], [5, 3], [1, 4]], dtype=torch.int64, device=device)
    track_ids = torch.tensor([3, 0, 2, 1, 2, 3], dtype=torch.int32, device=device)
    cache = _TrackFrameSlotCache()

    for relative_time in (-1, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11):
        timestamp_us = offset + relative_time
        expected = _reference_track_frame_slots(timestamp_us, track_ids, timestamps, packinfo)
        torch.testing.assert_close(nearest_track_frame_slots(timestamp_us, track_ids, timestamps, packinfo), expected)
        for _ in range(2):
            actual = nearest_track_frame_slots(timestamp_us, track_ids, timestamps, packinfo, cache=cache)
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cached_track_frame_slots_preserve_strided_id_shapes(dtype: torch.dtype, device: str) -> None:
    timestamps, packinfo, _ = _lookup_inputs(device)
    ids = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=dtype, device=device).T
    cache = _TrackFrameSlotCache()
    expected = torch.tensor([[1, 0], [0, 1], [1, 0]], device=device)
    for _ in range(2):
        actual = nearest_track_frame_slots(3, ids, timestamps, packinfo, cache=cache)
        torch.testing.assert_close(actual, expected)


def test_cached_track_frame_slots_follow_reordered_added_and_inplace_gaussian_ids(device: str) -> None:
    timestamps = torch.tensor([1, 3, 6, 8, 4], dtype=torch.int64)
    packinfo = torch.tensor([[0, 2], [2, 2], [4, 1]], dtype=torch.int64)
    cache = _TrackFrameSlotCache()
    track_ids = torch.tensor([0, 1], dtype=torch.int32)
    nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)

    for ids in ([1, 0], [2, 0, 2, 1, 0], [1], []):
        track_ids = torch.tensor(ids, dtype=torch.int32)
        actual = nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)
        torch.testing.assert_close(actual, _reference_track_frame_slots(3, track_ids, timestamps, packinfo))

    track_ids = torch.tensor([0, 1], dtype=torch.int32)
    nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)
    track_ids.copy_(torch.tensor([1, 0], dtype=torch.int32))
    torch.testing.assert_close(
        nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache), torch.tensor([0, 1])
    )


@pytest.mark.parametrize("source", ["timestamps", "packinfo"])
@pytest.mark.parametrize("replace", [False, True])
def test_cached_track_frame_slots_invalidate_changed_track_metadata(source: str, replace: bool, device: str) -> None:
    timestamps, packinfo, track_ids = _lookup_inputs()
    cache = _TrackFrameSlotCache()
    original = nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)
    torch.testing.assert_close(original, torch.tensor([1, 0, 1]))

    if source == "timestamps":
        if replace:
            timestamps = torch.tensor([1, 5, 6, 8], dtype=torch.int64)
        else:
            timestamps[1] = 5
    else:
        if replace:
            packinfo = torch.tensor([[2, 2], [0, 2]], dtype=torch.int64)
        else:
            packinfo.copy_(torch.tensor([[2, 2], [0, 2]], dtype=torch.int64))

    expected = _reference_track_frame_slots(3, track_ids, timestamps, packinfo)
    assert not torch.equal(original, expected)
    torch.testing.assert_close(nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache), expected)


def test_track_frame_slot_cache_serializes_concurrent_metadata_rebuilds(device: str) -> None:
    cache = _TrackFrameSlotCache()
    barrier = Barrier(2)
    packinfo = torch.tensor([[0, 2], [2, 2]], dtype=torch.int64)
    track_ids = torch.tensor([0, 1, 0], dtype=torch.int32)

    def lookup(timestamps: torch.Tensor, expected: torch.Tensor) -> None:
        """Repeatedly rebuild shared cache metadata from one serving thread."""

        for _ in range(8):
            barrier.wait()
            actual = nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)
            torch.testing.assert_close(actual, expected)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(lookup, torch.tensor([1, 3, 6, 8]), torch.tensor([1, 0, 1])),
            executor.submit(lookup, torch.tensor([1, 5, 6, 8]), torch.tensor([0, 0, 0])),
        )
        for future in futures:
            future.result()


@pytest.mark.parametrize("max_entries", [0, -1])
def test_track_frame_slot_cache_rejects_nonpositive_capacity(max_entries: int) -> None:
    with pytest.raises(ValueError, match="max_entries must be positive"):
        _TrackFrameSlotCache(max_entries=max_entries)


def test_track_frame_slot_cache_bounds_rows_and_evicts_least_recently_used_timestamp(device: str) -> None:
    timestamps = torch.tensor([1, 2, 3], dtype=torch.int64)
    packinfo = torch.tensor([[0, 3]], dtype=torch.int64)
    track_ids = torch.tensor([0], dtype=torch.int32)
    cache = _TrackFrameSlotCache(max_entries=2)
    for timestamp_us in (1, 2, 1, 3):
        actual = nearest_track_frame_slots(timestamp_us, track_ids, timestamps, packinfo, cache=cache)
        torch.testing.assert_close(actual, _reference_track_frame_slots(timestamp_us, track_ids, timestamps, packinfo))
        assert len(cache._rows) <= 2
    assert list(cache._rows) == [1, 3]

    torch.testing.assert_close(
        nearest_track_frame_slots(2, track_ids, timestamps, packinfo, cache=cache), torch.tensor([1])
    )
    assert list(cache._rows) == [3, 2]


def test_track_frame_slot_cache_default_capacity_covers_stress_working_set(device: str) -> None:
    timestamps = torch.tensor([0], dtype=torch.int64)
    packinfo = torch.tensor([[0, 1]], dtype=torch.int64)
    track_ids = torch.tensor([0], dtype=torch.int32)
    cache = _TrackFrameSlotCache()

    for timestamp_us in range(2726):
        slots = nearest_track_frame_slots(timestamp_us, track_ids, timestamps, packinfo, cache=cache)

    torch.testing.assert_close(slots, torch.tensor([0]))
    assert cache._max_entries == 4096
    assert len(cache._rows) == 2726
    assert next(iter(cache._rows)) == 0


def test_track_frame_slot_cache_pickle_preserves_capacity_and_drops_runtime_state(device: str) -> None:
    timestamps, packinfo, track_ids = _lookup_inputs(device)
    cache = _TrackFrameSlotCache(max_entries=2)
    nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)

    restored = pickle.loads(pickle.dumps(cache))

    assert restored._max_entries == 2
    assert not restored._rows
    torch.testing.assert_close(
        nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=restored),
        torch.tensor([1, 0, 1], device=device),
    )


def test_cached_track_frame_slots_support_inference_then_temporal_color_training(device: str) -> None:
    timestamps, packinfo, track_ids = _lookup_inputs(device)
    cache = _TrackFrameSlotCache()
    with torch.inference_mode():
        nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)

    colors = torch.arange(18, dtype=torch.float32, device=device).reshape(3, 2, 3).requires_grad_()
    reference_colors = colors.detach().clone().requires_grad_()
    slots = nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)
    reference_slots = _reference_track_frame_slots(3, track_ids, timestamps, packinfo)
    rows = torch.arange(len(track_ids), device=device)
    selected = colors[rows, slots]
    reference_selected = reference_colors[rows, reference_slots]
    selected.square().sum().backward()
    reference_selected.square().sum().backward()

    torch.testing.assert_close(selected, reference_selected)
    torch.testing.assert_close(colors.grad, reference_colors.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("count", [0, 2])
def test_fused_track_frame_slots_cover_empty_timelines_and_block_tail(count: int, device: str) -> None:
    # Cover a partial second block; counts retain the packinfo column stride.
    tracks = vehicle_light_module.packed_ops.TRACK_FRAME_SLOTS_THREADS_PER_BLOCK + 1
    timestamps = torch.arange(tracks * count, dtype=torch.int64, device="cuda")
    packinfo = torch.stack(
        (torch.arange(tracks, device="cuda") * count, torch.full((tracks,), count, device="cuda")), 1
    )
    ids = torch.arange(tracks, device="cuda")
    cache = _TrackFrameSlotCache()
    kernel = vehicle_light_module.packed_ops
    with mock.patch.object(kernel, "nearest_track_frame_slots", wraps=kernel.nearest_track_frame_slots) as launch:
        actual = cache.get(tracks, ids, timestamps, packinfo)
        launch.assert_called_once()
    torch.testing.assert_close(actual, _reference_track_frame_slots(tracks, ids, timestamps, packinfo))


@pytest.mark.parametrize("source", ["timestamps", "packinfo"])
def test_cached_track_frame_slots_handle_mutated_inference_metadata(source: str, device: str) -> None:
    timestamps, packinfo, track_ids = _lookup_inputs()
    with torch.inference_mode():
        if source == "timestamps":
            timestamps = timestamps.clone()
        else:
            packinfo = packinfo.clone()
    cache = _TrackFrameSlotCache()
    torch.testing.assert_close(
        nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache), torch.tensor([1, 0, 1])
    )

    with torch.inference_mode():
        if source == "timestamps":
            timestamps[1] = 5
        else:
            packinfo.copy_(torch.tensor([[2, 2], [0, 2]], dtype=torch.int64))

    torch.testing.assert_close(
        nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache),
        _reference_track_frame_slots(3, track_ids, timestamps, packinfo),
    )


@pytest.mark.parametrize("state", ["cold", "warm"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to detect GPU synchronization")
def test_track_frame_slot_cache_does_not_synchronize_cuda(state: str) -> None:
    timestamps, packinfo, track_ids = _lookup_inputs("cuda")
    cache = _TrackFrameSlotCache()
    if state == "warm":
        nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)
        torch.cuda.synchronize()

    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    torch.testing.assert_close(actual, torch.tensor([1, 0, 1], device="cuda"))


@pytest.mark.parametrize("state", ["warm_row", "cold_metadata"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise stream ordering")
def test_track_frame_slot_cache_orders_and_records_cross_stream_uses(state: str) -> None:
    timestamps, packinfo, track_ids = _lookup_inputs("cuda")
    cache = _TrackFrameSlotCache(max_entries=1)
    lifetime = CrossStreamAllocationTestHarness.create()
    with torch.cuda.stream(lifetime.producer):
        nearest_track_frame_slots(3, track_ids, timestamps, packinfo, cache=cache)
        lifetime.producer_ready.record()

    consumer_timestamp_us = 3 if state == "warm_row" else 8
    expected = _reference_track_frame_slots(consumer_timestamp_us, track_ids, timestamps, packinfo)
    pointers: tuple[int, ...]
    if state == "warm_row":
        pointers = (cache._rows[3].data_ptr(),)
    else:
        assert cache._track_timestamps is not None
        assert cache._track_packinfo is not None
        pointers = (cache._track_timestamps.data_ptr(), cache._track_packinfo.data_ptr())

    actual: list[torch.Tensor] = []

    def consume(consumer_timestamps: torch.Tensor, consumer_packinfo: torch.Tensor) -> None:
        """Enqueue one cache lookup while observing its dependency wait."""

        stream_type = type(lifetime.consumer)
        original_wait_event = stream_type.wait_event
        with mock.patch.object(stream_type, "wait_event", autospec=True, side_effect=original_wait_event) as wait_event:
            actual.append(
                nearest_track_frame_slots(
                    consumer_timestamp_us,
                    track_ids,
                    consumer_timestamps,
                    consumer_packinfo,
                    cache=cache,
                )
            )
        assert wait_event.call_count == 1
        assert wait_event.call_args.args[0].cuda_stream == lifetime.consumer.cuda_stream

    consumer_done = lifetime.enqueue_gated(consume, timestamps, packinfo)

    with torch.cuda.stream(lifetime.producer):
        if state == "warm_row":
            nearest_track_frame_slots(8, track_ids, timestamps, packinfo, cache=cache)
        else:
            replacement_timestamps = timestamps.clone()
            replacement_packinfo = packinfo.clone()
            nearest_track_frame_slots(
                4,
                track_ids,
                replacement_timestamps,
                replacement_packinfo,
                cache=cache,
            )
    del timestamps, packinfo
    lifetime.assert_allocations_pending(*pointers)
    consumer_done.synchronize()
    torch.testing.assert_close(actual[0], expected)


def test_only_vehicle_light_model_opts_into_animated_sh_export() -> None:
    base_model = RigidGaussianModel.__new__(RigidGaussianModel)
    vehicle_light_model = VehicleLightRigidGaussianModel.__new__(VehicleLightRigidGaussianModel)

    assert not base_model.has_temporal_appearance_for_export()
    assert vehicle_light_model.has_temporal_appearance_for_export()


def test_missing_sam3_inputs_skip_the_optional_vehicle_light_layer() -> None:
    try:
        _resolve_vehicle_light_mask_inputs(cast(Any, {"mask_dir": None, "mask_manifest_path": None}))
    except NoPointsFoundException as error:
        assert "not configured" in str(error)
    else:
        raise AssertionError("Missing SAM3 inputs must use the standard optional-layer skip exception")


def test_partial_sam3_inputs_fail_loudly() -> None:
    partial_configs = (
        {"mask_dir": "/masks", "mask_manifest_path": None},
        {"mask_dir": None, "mask_manifest_path": "/masks/manifest.json"},
    )
    for config in partial_configs:
        try:
            _resolve_vehicle_light_mask_inputs(cast(Any, config))
        except ValueError as error:
            assert "requires both" in str(error)
        else:
            raise AssertionError("A partial SAM3 configuration must fail instead of silently skipping")


def test_complete_sam3_inputs_enable_vehicle_light_initialization() -> None:
    assert _resolve_vehicle_light_mask_inputs(
        cast(
            Any,
            {
                "mask_dir": "/masks",
                "mask_manifest_path": "/masks/manifest.json",
            },
        )
    ) == ("/masks", "/masks/manifest.json")


def test_resolved_aux_artifact_supplies_both_sam3_inputs() -> None:
    config = cast(Any, {"mask_dir": None, "mask_manifest_path": None, "sam3_aux_dir": "/aux/sam3"})

    assert _resolve_vehicle_light_mask_inputs(config) == ("/aux/sam3/masks", "/aux/sam3/manifest.json")


def test_explicit_sam3_inputs_take_precedence_over_aux() -> None:
    config = cast(Any, {"mask_dir": "/masks", "mask_manifest_path": "/manifest.json", "sam3_aux_dir": "/aux/sam3"})

    assert _resolve_vehicle_light_mask_inputs(config) == ("/masks", "/manifest.json")
    config["mask_manifest_path"] = None
    with pytest.raises(ValueError, match="requires both"):
        _resolve_vehicle_light_mask_inputs(config)


def test_color_propagation_preserves_observed_black_and_fills_time() -> None:
    colors = torch.zeros((2, 3, 3), dtype=torch.float32)
    observed = torch.zeros((2, 3), dtype=torch.bool)
    colors[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    colors[1, 2] = torch.tensor([1.0, 0.5, 0.25])
    observed[0, 0] = True
    observed[1, 2] = True
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])

    filled = propagate_temporal_colors(colors, observed, positions, spatial_k=1, spatial_iterations=1)

    torch.testing.assert_close(filled[0, 0], torch.zeros(3))
    torch.testing.assert_close(filled[1, 2], torch.tensor([1.0, 0.5, 0.25]))
    assert torch.isfinite(filled).all()


def test_voxel_aggregation_applies_final_scale_floor() -> None:
    _, aggregated_scales, _, _ = _aggregate_track_observations(
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        colors_uint8=torch.tensor([[255, 0, 0], [0, 255, 0]], dtype=torch.uint8),
        scales=torch.tensor([0.001, 0.001]),
        frame_slots=torch.tensor([0, 0]),
        n_frames=1,
        voxel_size_m=0.02,
        max_points=10,
        spatial_k=1,
        spatial_iterations=1,
        min_scale_voxel_ratio=0.2,
        scale_multiplier=0.4,
    )

    torch.testing.assert_close(aggregated_scales * 0.4, torch.full_like(aggregated_scales, 0.004))


def test_all_zero_masks_raise_no_points_found_for_layer_skip() -> None:
    class _CuboidTracksStub:
        tracks_id = ["vehicle-1"]
        tracks_packinfo = torch.tensor([[0, 1]], dtype=torch.int64)

    class _NCOREDataSourceStub:
        def get_masked_camera_track_point_clouds(self, **kwargs):
            return ()

    initializer = VehicleLightTracksInitialization.__new__(VehicleLightTracksInitialization)
    torch.nn.Module.__init__(initializer)
    initializer.config = cast(
        Any,
        {
            "mask_dir": "/tmp/all-zero-vehicle-light-masks",
            "mask_manifest_path": "/tmp/all-zero-vehicle-light-masks/manifest.json",
            "camera_ids": ["camera_front_wide_120fov"],
        },
    )
    initializer.register_buffer("_device_indicator", torch.tensor(0), persistent=False)
    summary = SimpleNamespace(datasource=_NCOREDataSourceStub())

    with (
        mock.patch.object(vehicle_light_module, "CuboidTracks", _CuboidTracksStub),
        mock.patch.object(vehicle_light_module, "NCOREDataSource", _NCOREDataSourceStub),
        mock.patch.object(vehicle_light_module, "get_masked_camera_track_point_clouds", return_value=()),
    ):
        try:
            initializer.initialize_from_datasource(cast(Any, summary), cuboid_tracks=_CuboidTracksStub())
        except NoPointsFoundException as error:
            assert "No vehicle-light points" in str(error)
        else:
            raise AssertionError("All-zero masks must use the standard empty-layer skip exception")


def test_vehicle_light_optimizer_applies_lr_x5_with_distributed_scaling() -> None:
    for world_size, expected_lr in ((1, 0.0125), (4, 0.025)):
        model = VehicleLightRigidGaussianModel.__new__(VehicleLightRigidGaussianModel)
        torch.nn.Module.__init__(model)
        model.config = cast(Any, SimpleNamespace(features_albedo_lr=0.0125))
        model.trainer_config = cast(
            Any,
            SimpleNamespace(world_size=world_size, batch_size_scaling_factor=1.0, relative_lr=True),
        )
        albedo = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [albedo],
                    "name": "model.vehicle_lights.features_albedo",
                    "lr": 0.0025 * world_size**0.5,
                }
            ]
        )
        configured = cast(Any, [{"optimizer": optimizer}])

        with mock.patch.object(RigidGaussianModel, "configure_optimizers", return_value=configured):
            result = model.configure_optimizers()

        assert result[0]["optimizer"].param_groups[0]["lr"] == expected_lr


def test_vehicle_light_initializer_type_guard_runs_before_base_mutation() -> None:
    model = VehicleLightRigidGaussianModel.__new__(VehicleLightRigidGaussianModel)
    torch.nn.Module.__init__(model)

    with mock.patch.object(RigidGaussianModel, "initialize_gaussians_from_datasource") as base_initialize:
        try:
            model.initialize_gaussians_from_datasource(cast(Any, object()), cast(Any, object()))
        except TypeError as error:
            assert "VehicleLightTracksInitialization" in str(error)
        else:
            raise AssertionError("Wrong initializer type must fail before base initialization")

    base_initialize.assert_not_called()


def test_vehicle_light_appearance_uses_original_timeline_when_motion_is_frozen(device: str) -> None:
    model = VehicleLightRigidGaussianModel.__new__(VehicleLightRigidGaussianModel)
    torch.nn.Module.__init__(model)
    cast(Any, model).cuboid_tracks = SimpleNamespace(
        tracks_timestamps_us=torch.tensor([10, 20, 30, 100, 130], dtype=torch.int64),
        tracks_packinfo=torch.tensor([[0, 3], [3, 2]], dtype=torch.int64),
    )
    cast(Any, model).prepare_motion = mock.MagicMock(
        side_effect=AssertionError("motion timeline must not index temporal appearance")
    )
    cast(Any, model).get_frame_timestamp = mock.MagicMock(return_value=20)
    layer_data = SimpleNamespace(
        features_albedo=torch.tensor(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ]
        ),
        tracks_ids=torch.tensor([0, 1], dtype=torch.int32),
        embed_data=object(),
    )
    context = SimpleNamespace(
        rendering_data=object(),
        tracks_edit=SimpleNamespace(
            cuboid_tracks=SimpleNamespace(
                tracks_timestamps_us=torch.tensor([-999_999_990, 1_000_000_030], dtype=torch.int64)
            )
        ),
    )

    with mock.patch.object(RigidGaussianModel, "get_layer_data", return_value=layer_data):
        result = model.get_layer_data(cast(Any, context), {"sentinel": torch.tensor(1)})

    torch.testing.assert_close(
        result.features_albedo,
        torch.tensor([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    )
    cast(mock.MagicMock, model.prepare_motion).assert_not_called()
    assert result.embed_data is None


def test_bound_vehicle_light_keeps_original_frame_columns_when_owner_timeline_is_coalesced(device: str) -> None:
    owner = RigidGaussianModel.__new__(RigidGaussianModel)
    light = VehicleLightRigidGaussianModel.__new__(VehicleLightRigidGaussianModel)
    for model in (owner, light):
        torch.nn.Module.__init__(model)
        model._motion_source_ref = None
        model.is_static = False
        model.nearest_neighbor_track_for_lidar = False
    owner.config = cast(Any, SimpleNamespace(motion_source_layer=None))
    light.config = cast(Any, SimpleNamespace(motion_source_layer="body"))
    owner.cuboid_tracks = cast(
        Any,
        SimpleNamespace(
            tracks_id=["a", "b", "c"],
            tracks_timestamps_us=torch.tensor([10, 40, 15, 45, 100, 130]),
            tracks_packinfo=torch.tensor([[0, 2], [2, 2], [4, 2]]),
        ),
    )
    original_frames = SimpleNamespace(
        tracks_id=["a", "b", "c"],
        tracks_timestamps_us=torch.tensor([10, 20, 40, 15, 25, 35, 45, 100, 130]),
        tracks_packinfo=torch.tensor([[0, 3], [3, 4], [7, 2]]),
    )
    light.cuboid_tracks = cast(Any, original_frames)
    light.bind_motion_source(owner)
    assert light.cuboid_tracks is original_frames
    assert light.get_motion_source() is owner
    cast(Any, light).get_frame_timestamp = mock.Mock(return_value=20)
    colors = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3)
    layer_data = SimpleNamespace(
        features_albedo=colors,
        tracks_ids=torch.tensor([1, 0, 2, 1], dtype=torch.int32),
        embed_data=object(),
    )
    # The owner may later append motion metadata without extending the light's initialized color frames.
    owner.cuboid_tracks = cast(Any, SimpleNamespace(tracks_id=["a", "b", "c", "inserted"]))
    with mock.patch.object(RigidGaussianModel, "get_layer_data", return_value=layer_data):
        result = light.get_layer_data(cast(Any, SimpleNamespace(rendering_data=object())), {})
    torch.testing.assert_close(result.features_albedo, colors[torch.arange(4), torch.tensor([0, 1, 0, 0])])
    assert light.cuboid_tracks is original_frames
    assert result.embed_data is None
