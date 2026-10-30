# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import math
import subprocess
import sys
import tempfile
import time
import unittest
import weakref

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional, cast
from unittest.mock import Mock, patch

import torch

import nurec.models.gaussians.fused_parameter_gather as fused_gather

from nurec.utils.misc import all_gather_tensor_list
from nurec.utils.selective_adam_visibility import prepare_async_visibility_union


_NODES = ("background", "road", "dynamic_rigids", "dynamic_deformables")
_LIDARFREE_NODES = (*_NODES, "static_rigids", "traffic_light", "vehicle_lights", "static_vehicle_lights")


def _layer_count_packet_values(
    counts: tuple[int, ...],
    schema_hash: tuple[int, ...],
    *,
    integer_ranges_cached: bool = False,
    unused_inline_values: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Mirror the explicit Python wire contract without using CUDA capacity."""
    inline_counts = counts[: fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY]
    unused_count = fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY - len(inline_counts)
    if len(unused_inline_values) > unused_count:
        raise ValueError("too many unused inline values")
    return (
        fused_gather._LAYER_COUNT_PACKET_PROTOCOL_VERSION,
        len(counts),
        *schema_hash,
        int(integer_ranges_cached),
        *inline_counts,
        *unused_inline_values,
        *(0 for _ in range(unused_count - len(unused_inline_values))),
    )


def _reference_gather(
    parameters_by_node: dict[str, dict[str, torch.Tensor]],
    process_group: Any,
) -> dict[str, dict[str, torch.Tensor]]:
    """Gather each layer independently through the established autograd path."""
    world_size = torch.distributed.get_world_size(process_group)
    gathered_by_node: dict[str, dict[str, torch.Tensor]] = {}
    for node_id, parameters in parameters_by_node.items():
        keys = tuple(parameters)
        tensors = tuple(parameters.values())
        local_count = tensors[0].shape[0]
        shapes = tuple(tensor.shape for tensor in tensors)
        dtypes = tuple(tensor.dtype for tensor in tensors)
        flattened = [tensor.reshape(local_count, math.prod(tensor.shape[1:])) for tensor in tensors]
        gathered = all_gather_tensor_list(world_size, flattened, group=process_group)
        gathered_count = gathered[0].shape[0]
        gathered_by_node[node_id] = {
            key: tensor.reshape(gathered_count, *shape[1:]).to(dtype)
            for key, tensor, shape, dtype in zip(keys, gathered, shapes, dtypes, strict=True)
        }
    return gathered_by_node


def _make_nccl_oracle_parameters(rank: int, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    first_count = rank + 1
    second_count = 2 - rank

    def values(
        count: int,
        width: int,
        offset: float,
        *,
        requires_grad: bool,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        tensor = torch.arange(count * width, device=device, dtype=dtype).reshape(count, width)
        return (tensor + offset).requires_grad_(requires_grad)

    # The active fields contain nine values on rank 0 and twelve on rank 1,
    # forcing the fused ReduceScatter input to include rank padding.
    return {
        "first": {
            # Mixed FP16+FP32 fields exercise payload promotion, SoA dtype
            # unpacking, and the FP16 local-gradient cast after ReduceScatter.
            "used": values(first_count, 2, rank * 10.0, requires_grad=True, dtype=torch.float16),
            "unused": values(first_count, 1, rank * 20.0, requires_grad=True),
            "frozen": values(first_count, 1, rank * 30.0, requires_grad=False),
            "zero_width": values(first_count, 0, rank * 35.0, requires_grad=True),
        },
        "partially_empty": {
            "used": values(rank, 2, rank * 40.0, requires_grad=True),
        },
        # Skipping this globally empty payload must leave both ranks aligned
        # for the equal-count collective that immediately follows it.
        "globally_empty": {
            "used": values(0, 2, rank * 50.0, requires_grad=True),
        },
        "equal_count": {
            "used": values(1, 2, rank * 60.0, requires_grad=True),
        },
        "second": {
            "globally_unused": values(second_count, 2, rank * 70.0, requires_grad=True),
        },
    }


def _make_dynamic_nccl_oracle_parameters(
    rank: int,
    generation: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, tuple[int, ...]]]:
    """Build one schema whose row counts change between consecutive forwards."""
    counts_by_generation = (
        ((0, 1), (2, 0), (1, 2), (0, 1)),
        ((2, 0), (0, 1), (0, 1), (1, 0)),
    )
    node_ids = ("first", "partially_empty", "globally_empty", "equal_count")
    counts_by_node: dict[str, tuple[int, ...]] = dict(zip(node_ids, counts_by_generation[generation], strict=True))

    def values(
        node_id: str,
        width: int,
        offset: float,
        *,
        requires_grad: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        local_count = counts_by_node[node_id][rank]
        tensor = torch.arange(local_count * width, device=device, dtype=dtype).reshape(local_count, width)
        return (tensor + generation * 100.0 + rank * 10.0 + offset).requires_grad_(requires_grad)

    parameters_by_node = {
        "first": {
            "used": values("first", 2, 0.0, dtype=torch.float16),
            "unused": values("first", 1, 1.0),
            "frozen": values("first", 1, 2.0, requires_grad=False),
        },
        "partially_empty": {"used": values("partially_empty", 2, 3.0)},
        "globally_empty": {"used": values("globally_empty", 2, 4.0)},
        "equal_count": {"used": values("equal_count", 2, 5.0)},
    }
    return parameters_by_node, counts_by_node


def _clone_oracle_parameters(
    parameters_by_node: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        node_id: {
            key: tensor.detach().clone().requires_grad_(tensor.requires_grad) for key, tensor in parameters.items()
        }
        for node_id, parameters in parameters_by_node.items()
    }


def _fake_two_rank_payload_gather(output: torch.Tensor, payload: torch.Tensor, **_kwargs: Any) -> None:
    if payload.ndim == 1:
        assert tuple(output.shape) == (2 * payload.shape[0],)
        gathered_by_rank = output.view(2, payload.shape[0])
    else:
        assert tuple(output.shape) == (2 * payload.shape[0], payload.shape[1])
        gathered_by_rank = output.view(2, payload.shape[0], payload.shape[1])
    gathered_by_rank[0].copy_(payload)
    gathered_by_rank[1].zero_()


def _assert_nccl_oracle_matches_reference(
    fused_parameters: dict[str, dict[str, torch.Tensor]],
    expected_counts: dict[str, tuple[int, ...]],
    expected_metadata_gathers: int,
    process_group: Any,
) -> None:
    reference_parameters = _clone_oracle_parameters(fused_parameters)
    metadata_gathers = 0
    real_all_gather = torch.distributed.all_gather_into_tensor

    def tracked_all_gather(
        output: torch.Tensor,
        local: torch.Tensor,
        group=None,
        async_op: bool = False,
    ) -> Any:
        nonlocal metadata_gathers
        if local.dtype == fused_gather._COUNT_METADATA_DTYPE:
            metadata_gathers += 1
        return real_all_gather(output, local, group=group, async_op=async_op)

    with patch.object(
        torch.distributed,
        "all_gather_into_tensor",
        side_effect=tracked_all_gather,
    ):
        fused_result = fused_gather._fused_all_gather_gaussian_parameters(
            list(fused_parameters),
            fused_parameters,
            process_group=process_group,
        )

    assert metadata_gathers == expected_metadata_gathers
    fused_outputs = fused_result.parameters_by_node
    assert fused_result.counts_by_node == expected_counts
    reference_outputs = _reference_gather(reference_parameters, process_group)
    assert fused_outputs["first"]["used"].dtype == torch.float16
    assert fused_outputs["first"]["unused"].dtype == torch.float32
    for node_id, parameters in reference_outputs.items():
        for key, expected in parameters.items():
            assert fused_outputs[node_id][key].is_contiguous()
            torch.testing.assert_close(fused_outputs[node_id][key], expected, rtol=0, atol=0)
    first_float32_storage = fused_outputs["first"]["unused"].untyped_storage().data_ptr()
    assert fused_outputs["first"]["frozen"].untyped_storage().data_ptr() == first_float32_storage
    assert fused_outputs["first"]["used"].untyped_storage().data_ptr() != first_float32_storage

    def active_loss(parameters_by_node: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        loss = parameters_by_node["first"]["used"].square().sum()
        loss = loss + parameters_by_node["first"]["frozen"].sum()
        if "zero_width" in parameters_by_node["first"]:
            loss = loss + parameters_by_node["first"]["zero_width"].sum()
        for node_id in ("partially_empty", "globally_empty", "equal_count"):
            if node_id in parameters_by_node:
                loss = loss + parameters_by_node[node_id]["used"].square().sum()
        return loss

    fused_loss = active_loss(fused_outputs)
    reference_loss = active_loss(reference_outputs)
    fused_loss.backward()
    reference_loss.backward()

    for node_id, parameters in reference_parameters.items():
        for key, reference in parameters.items():
            fused = fused_parameters[node_id][key]
            if not reference.requires_grad:
                assert fused.grad is None and reference.grad is None
            elif reference.grad is None:
                assert fused.grad is None
            else:
                assert fused.grad is not None
                assert fused.grad.dtype == reference.grad.dtype
                torch.testing.assert_close(fused.grad, reference.grad, rtol=0, atol=0)


def _grouped_nccl_counts(generation: int, *, lidarfree: bool = False) -> dict[str, tuple[int, ...]]:
    counts_by_generation: tuple[dict[str, tuple[int, ...]], ...] = (
        {
            "background": (1, 2),
            "road": (2, 1),
            "dynamic_rigids": (0, 1),
            "dynamic_deformables": (1, 1),
        },
        {
            "background": (2, 0),
            "road": (0, 2),
            "dynamic_rigids": (1, 0),
            "dynamic_deformables": (2, 1),
        },
    )
    counts = counts_by_generation[generation]
    if lidarfree:
        counts.update(
            static_rigids=(0, 1) if generation == 0 else (2, 0),
            traffic_light=(0, 0) if generation == 0 else (1, 1),
            vehicle_lights=(2, 1),
            static_vehicle_lights=(1, 0),
        )
    return counts


def _make_grouped_nccl_parameters(
    rank: int,
    device: torch.device,
    generation: int = 0,
    *,
    lidarfree: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    counts_by_node = _grouped_nccl_counts(generation, lidarfree=lidarfree)
    counts = {node_id: rank_counts[rank] for node_id, rank_counts in counts_by_node.items()}

    def values(node: str, width: int, offset: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        count = counts[node]
        tensor = torch.arange(count * width, dtype=torch.float32, device=device).reshape(count, width)
        return (tensor + generation * 1000 + offset + rank * 100).to(dtype).requires_grad_()

    def fourier_values(node: str, offset: float) -> torch.Tensor:
        count = counts[node]
        tensor = torch.arange(count * 2 * 3, dtype=torch.float32, device=device).reshape(count, 2, 3)
        return (tensor + generation * 1000 + offset + rank * 100).requires_grad_()

    parameters = {
        "background": {"value": values("background", 2, 0)},
        "road": {
            "half": values("road", 2, 10, torch.float16),
            "float": values("road", 1, 20),
        },
        "dynamic_rigids": {
            "value": values("dynamic_rigids", 3, 30),
            "features_albedo": fourier_values("dynamic_rigids", 35),
        },
        "dynamic_deformables": {"value": values("dynamic_deformables", 2, 40)},
    }
    if lidarfree:
        for node in _LIDARFREE_NODES[len(_NODES) :]:
            count = counts[node]
            # Vehicle-light appearance is a per-track-frame [G,N,3] table,
            # not an RGB vector. Use a representative 97-frame temporal table,
            # including empty shards and rank-local frame selections below.
            frame_count = 97 if node in ("vehicle_lights", "static_vehicle_lights") else 1
            parameters[node] = {
                "positions": values(node, 3, 1),
                "features_albedo": values(node, frame_count * 3, 2)
                .detach()
                .reshape(count, frame_count, 3)
                .requires_grad_(),
                "gaussian_cuboid_ids": torch.arange(count, dtype=torch.int64, device=device),
            }
    return parameters


def _assert_grouped_nccl_oracle_matches_reference(
    rank: int,
    device: torch.device,
    process_group: Any,
    *,
    generation: int = 0,
    prefetch_payload: bool = False,
    source_node_ids: tuple[str, ...] = _NODES,
    node_ids: tuple[str, ...] = _NODES,
    lidarfree: bool = False,
) -> None:
    assert len(source_node_ids) == len(node_ids)
    canonical_parameters = _make_grouped_nccl_parameters(rank, device, generation, lidarfree=lidarfree)
    canonical_counts = _grouped_nccl_counts(generation, lidarfree=lidarfree)
    fused_parameters = {
        node_id: canonical_parameters[source_node_id]
        for source_node_id, node_id in zip(source_node_ids, node_ids, strict=True)
    }
    expected_counts = {
        node_id: canonical_counts[source_node_id]
        for source_node_id, node_id in zip(source_node_ids, node_ids, strict=True)
    }
    reference_parameters = _clone_oracle_parameters(fused_parameters)
    metadata_gathers = 0
    payload_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    payload_async_flags: list[bool] = []
    real_all_gather = torch.distributed.all_gather_into_tensor

    def tracked_all_gather(
        output: torch.Tensor,
        local: torch.Tensor,
        group=None,
        async_op: bool = False,
    ) -> Any:
        nonlocal metadata_gathers
        if local.dtype == fused_gather._COUNT_METADATA_DTYPE:
            metadata_gathers += 1
        else:
            payload_shapes.append((tuple(output.shape), tuple(local.shape)))
            payload_async_flags.append(async_op)
        return real_all_gather(output, local, group=group, async_op=async_op)

    with patch.object(
        torch.distributed,
        "all_gather_into_tensor",
        side_effect=tracked_all_gather,
    ):
        prepared = None
        pending = None
        plan = None
        if prefetch_payload:
            prepared = fused_gather._prepare_fused_gather_inputs(
                list(fused_parameters),
                fused_parameters,
                process_group=process_group,
            )
            plan = fused_gather._resolve_fused_gather_generation_plan(
                prepared,
                process_group=process_group,
            )
            pending = fused_gather._start_fused_payload_gather_from_generation_plan(
                prepared,
                plan,
                process_group=process_group,
            )
            # Model the independent FreePose work between payload launch and
            # the gather-to-collector resolver boundary.
            torch.arange(4096, dtype=torch.float32, device=device).cos_()
        fused_result = fused_gather._fused_all_gather_gaussian_parameters(
            list(fused_parameters),
            fused_parameters,
            process_group=process_group,
            prepared_inputs=prepared,
            generation_plan=plan,
            prefetched_payload_gather=pending,
        )

    assert metadata_gathers == (1 if len(node_ids) <= fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY else 2)
    assert len(payload_shapes) == 1
    assert len(payload_shapes[0][1]) == 1
    assert payload_async_flags == [prefetch_payload]
    assert fused_result.counts_by_node == expected_counts
    if "dynamic_rigids" in source_node_ids:
        dynamic_node_id = node_ids[source_node_ids.index("dynamic_rigids")]
        assert fused_result.parameters_by_node[dynamic_node_id]["features_albedo"].shape[1:] == (2, 3)

    reference_outputs = _reference_gather(reference_parameters, process_group)
    for node_id, parameters in reference_outputs.items():
        for key, expected in parameters.items():
            actual = fused_result.parameters_by_node[node_id][key]
            assert actual.dtype == expected.dtype
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def loss_for(parameters_by_node: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        loss = torch.zeros((), device=device)
        for node_id, parameters in parameters_by_node.items():
            for key, field in parameters.items():
                if node_id in ("vehicle_lights", "static_vehicle_lights") and key == "features_albedo":
                    # Different cameras/ranks select different frame slots
                    # after gathering, as VehicleLightRigidGaussianModel does.
                    field = field[:, rank * 48]
                loss = loss + field.square().sum()
        return loss

    fused_loss = loss_for(fused_result.parameters_by_node)
    reference_loss = loss_for(reference_outputs)
    fused_loss.backward()
    reference_loss.backward()
    for node_id, parameters in reference_parameters.items():
        for key, reference in parameters.items():
            fused = fused_parameters[node_id][key]
            if not reference.requires_grad:
                assert fused.grad is None and reference.grad is None
                continue
            assert fused.grad is not None and reference.grad is not None
            assert fused.grad.dtype == reference.grad.dtype
            torch.testing.assert_close(fused.grad, reference.grad, rtol=0, atol=0)


def _assert_single_node_training_nccl_matches_prior_per_node_path(
    rank: int,
    device: torch.device,
    process_group: Any,
) -> None:
    """Exercise the real modality-independent boundary used by camera-only training.

    Camera versus lidar selection happens after the full parameter gather and
    is therefore absent from this API.  The composite routing tests pin that
    boundary; here actual two-rank NCCL communication proves that the eligible
    one-layer prefetched route has exact forward and local-gradient parity with
    the former independent per-node autograd gather.
    """
    _assert_grouped_nccl_oracle_matches_reference(
        rank,
        device,
        process_group,
        prefetch_payload=True,
        source_node_ids=("dynamic_rigids",),
        node_ids=("only_layer",),
    )


def _assert_generation_plan_nccl_reuse_and_refresh(
    rank: int,
    device: torch.device,
    process_group: Any,
    *,
    lidarfree: bool = False,
) -> None:
    """Exercise collector-boundary warmup, stable reuse, and one refresh."""
    metadata_gathers = 0
    payload_gathers = 0
    validity_reductions = 0
    track_fused_collectives = True
    real_all_gather = torch.distributed.all_gather_into_tensor
    real_all_reduce = torch.distributed.all_reduce

    def tracked_all_gather(
        output: torch.Tensor,
        local: torch.Tensor,
        group=None,
        async_op: bool = False,
    ) -> Any:
        nonlocal metadata_gathers, payload_gathers
        if track_fused_collectives:
            if local.dtype == fused_gather._COUNT_METADATA_DTYPE:
                metadata_gathers += 1
            elif async_op:
                payload_gathers += 1
        return real_all_gather(output, local, group=group, async_op=async_op)

    def tracked_all_reduce(
        tensor: torch.Tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=None,
        async_op: bool = False,
    ) -> Any:
        nonlocal validity_reductions
        validity_reductions += 1
        return real_all_reduce(tensor, op=op, group=group, async_op=async_op)

    def add_exact_integer_ids(
        parameters: dict[str, dict[str, torch.Tensor]],
        generation: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        local_count = parameters["background"]["value"].shape[0]
        parameters["background"]["id"] = (
            torch.arange(local_count, dtype=torch.int64, device=device).reshape(local_count, 1)
            + generation * 1000
            + rank * 100
        )
        return parameters

    def collective_counts() -> tuple[int, int, int]:
        # Count metadata AllGather, payload AllGather, validity AllReduce.
        return metadata_gathers, payload_gathers, validity_reductions

    def collective_delta(
        before: tuple[int, int, int],
        after: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        return after[0] - before[0], after[1] - before[1], after[2] - before[2]

    def validate(
        parameters: dict[str, dict[str, torch.Tensor]],
        result: Any,
        generation: int,
    ) -> None:
        nonlocal track_fused_collectives
        reference_parameters = _clone_oracle_parameters(parameters)
        track_fused_collectives = False
        try:
            reference_outputs = _reference_gather(reference_parameters, process_group)
        finally:
            track_fused_collectives = True
        assert result.counts_by_node == _grouped_nccl_counts(generation, lidarfree=lidarfree)
        fused_loss = torch.zeros((), device=device)
        reference_loss = torch.zeros((), device=device)
        for node_id, expected_parameters in reference_outputs.items():
            for key, expected in expected_parameters.items():
                actual = result.parameters_by_node[node_id][key]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                fused_loss = fused_loss + actual.square().sum()
                reference_loss = reference_loss + expected.square().sum()
        fused_loss.backward()
        reference_loss.backward()
        for node_id, expected_parameters in reference_parameters.items():
            for key, expected in expected_parameters.items():
                actual = parameters[node_id][key]
                if not expected.requires_grad:
                    assert actual.grad is None and expected.grad is None
                else:
                    assert actual.grad is not None and expected.grad is not None
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    with (
        patch.object(torch.distributed, "all_gather_into_tensor", side_effect=tracked_all_gather),
        patch.object(torch.distributed, "all_reduce", side_effect=tracked_all_reduce),
    ):
        initial_parameters = add_exact_integer_ids(
            _make_grouped_nccl_parameters(rank, device, generation=0, lidarfree=lidarfree),
            generation=0,
        )
        generation_before = collective_counts()
        initial_prepared = fused_gather._prepare_fused_gather_inputs(
            list(initial_parameters),
            initial_parameters,
            process_group=process_group,
        )
        plan = fused_gather._resolve_fused_gather_generation_plan(
            initial_prepared,
            process_group=process_group,
        )
        initial_pending = fused_gather._start_fused_payload_gather_from_generation_plan(
            initial_prepared,
            plan,
            process_group=process_group,
        )
        initial_result = fused_gather._fused_all_gather_gaussian_parameters(
            list(initial_parameters),
            initial_parameters,
            process_group=process_group,
            prepared_inputs=initial_prepared,
            generation_plan=plan,
            prefetched_payload_gather=initial_pending,
        )
        validate(initial_parameters, initial_result, generation=0)
        generation_after = collective_counts()
        generation_delta = collective_delta(generation_before, generation_after)
        count_collectives = (
            1 if len(initial_parameters) <= fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY else 2
        )
        assert generation_delta == (count_collectives, 1, 1), generation_delta

        stable_parameters = add_exact_integer_ids(
            _make_grouped_nccl_parameters(rank, device, generation=0, lidarfree=lidarfree),
            generation=0,
        )
        # Floating payloads remain live, while the validated integer tensor is
        # structural generation state and therefore retains its identity.
        stable_parameters["background"]["id"] = initial_parameters["background"]["id"]
        if lidarfree:
            for node in _LIDARFREE_NODES[len(_NODES) :]:
                stable_parameters[node]["gaussian_cuboid_ids"] = initial_parameters[node]["gaussian_cuboid_ids"]
        steady_before = collective_counts()
        stable_prepared = fused_gather._prepare_fused_gather_inputs_from_generation_plan(
            list(stable_parameters),
            stable_parameters,
            plan,
            process_group=process_group,
        )
        stable_pending = fused_gather._start_fused_payload_gather_from_generation_plan(
            stable_prepared,
            plan,
            process_group=process_group,
        )
        stable_result = fused_gather._fused_all_gather_gaussian_parameters(
            list(stable_parameters),
            stable_parameters,
            process_group=process_group,
            prepared_inputs=stable_prepared,
            generation_plan=plan,
            prefetched_payload_gather=stable_pending,
        )
        validate(stable_parameters, stable_result, generation=0)
        steady_after = collective_counts()
        steady_delta = collective_delta(steady_before, steady_after)
        assert steady_delta == (0, 1, 0), steady_delta

        mutated_parameters = add_exact_integer_ids(
            _make_grouped_nccl_parameters(rank, device, generation=1, lidarfree=lidarfree),
            generation=1,
        )
        refresh_before = collective_counts()
        mutated_prepared = fused_gather._prepare_fused_gather_inputs(
            list(mutated_parameters),
            mutated_parameters,
            process_group=process_group,
        )
        refresh = fused_gather._start_layer_count_gather(
            mutated_prepared.world_size,
            mutated_prepared.local_counts,
            mutated_prepared.device,
            mutated_prepared.schema_hash,
            process_group=process_group,
        )
        refreshed_plan = fused_gather._make_fused_gather_generation_plan(
            mutated_prepared,
            fused_gather._finish_layer_count_gather(refresh),
            process_group=process_group,
        )
        refreshed_parameters = add_exact_integer_ids(
            _make_grouped_nccl_parameters(rank, device, generation=1, lidarfree=lidarfree),
            generation=1,
        )
        refreshed_parameters["background"]["id"] = mutated_parameters["background"]["id"]
        if lidarfree:
            for node in _LIDARFREE_NODES[len(_NODES) :]:
                refreshed_parameters[node]["gaussian_cuboid_ids"] = mutated_parameters[node]["gaussian_cuboid_ids"]
        refreshed_prepared = fused_gather._prepare_fused_gather_inputs_from_generation_plan(
            list(refreshed_parameters),
            refreshed_parameters,
            refreshed_plan,
            process_group=process_group,
        )
        refreshed_pending = fused_gather._start_fused_payload_gather_from_generation_plan(
            refreshed_prepared,
            refreshed_plan,
            process_group=process_group,
        )
        refreshed_result = fused_gather._fused_all_gather_gaussian_parameters(
            list(refreshed_parameters),
            refreshed_parameters,
            process_group=process_group,
            prepared_inputs=refreshed_prepared,
            generation_plan=refreshed_plan,
            prefetched_payload_gather=refreshed_pending,
        )
        validate(refreshed_parameters, refreshed_result, generation=1)
        refresh_after = collective_counts()
        refresh_delta = collective_delta(refresh_before, refresh_after)
        assert refresh_delta == (count_collectives, 1, 1), refresh_delta

    observed_collectives = (metadata_gathers, payload_gathers, validity_reductions)
    assert metadata_gathers == 2 * count_collectives, observed_collectives
    assert payload_gathers == 3, observed_collectives
    assert validity_reductions == 2, observed_collectives


def _assert_generation_plan_rank_asymmetric_consumption(
    rank: int,
    device: torch.device,
    process_group: Any,
) -> None:
    """Keep collectives aligned when local sensors use different fields."""
    parameters = _make_grouped_nccl_parameters(rank, device)
    prepared = fused_gather._prepare_fused_gather_inputs(
        list(parameters),
        parameters,
        process_group=process_group,
    )
    plan = fused_gather._resolve_fused_gather_generation_plan(
        prepared,
        process_group=process_group,
    )
    pending = fused_gather._start_fused_payload_gather_from_generation_plan(
        prepared,
        plan,
        process_group=process_group,
    )
    result = fused_gather._fused_all_gather_gaussian_parameters(
        list(parameters),
        parameters,
        process_group=process_group,
        prepared_inputs=prepared,
        generation_plan=plan,
        prefetched_payload_gather=pending,
    )

    # Model rank-local modality selection while preserving the established
    # whole-layer participation contract: rank 0 consumes only the first field
    # of every layer, while rank 1 consumes every field. Both ranks therefore
    # own the same fused autograd node and enter the same ReduceScatter.
    loss = torch.zeros((), device=device)
    for fields in result.parameters_by_node.values():
        selected_fields = tuple(fields.values())[:1] if rank == 0 else tuple(fields.values())
        for field in selected_fields:
            loss = loss + field.square().sum()
    loss.backward()

    for fields in parameters.values():
        for field_index, field in enumerate(fields.values()):
            # First fields receive both ranks' derivatives; sibling fields
            # receive only rank 1's derivative.
            scale = 4 if field_index == 0 else 2
            assert field.grad is not None
            torch.testing.assert_close(field.grad, field.detach() * scale, rtol=0, atol=0)

    torch.distributed.barrier()


def _assert_all_empty_generation_plan(
    rank: int,
    device: torch.device,
    process_group: Any,
) -> None:
    """Exercise the production cached route without issuing zero-byte collectives."""
    field = torch.empty((0, 2), dtype=torch.float32, device=device, requires_grad=True)
    parameters = {"empty": {"value": field}}
    prepared = fused_gather._prepare_fused_gather_inputs(
        ["empty"],
        parameters,
        process_group=process_group,
    )
    plan = fused_gather._resolve_fused_gather_generation_plan(
        prepared,
        process_group=process_group,
    )

    real_all_gather = torch.distributed.all_gather_into_tensor
    payload_collectives = 0

    def track_payload(output: torch.Tensor, local: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        nonlocal payload_collectives
        if local.dtype != fused_gather._COUNT_METADATA_DTYPE:
            payload_collectives += 1
        return real_all_gather(output, local, *args, **kwargs)

    with patch.object(torch.distributed, "all_gather_into_tensor", side_effect=track_payload):
        pending = fused_gather._start_fused_payload_gather_from_generation_plan(
            prepared,
            plan,
            process_group=process_group,
        )
        assert pending.work is None
        with (
            patch.object(fused_gather.gaussian_collective_payload_ops, "_pack_gradients_trusted") as pack_gradients,
            patch.object(torch.distributed, "reduce_scatter_tensor") as reduce_scatter,
        ):
            result = fused_gather._fused_all_gather_gaussian_parameters(
                ["empty"],
                parameters,
                process_group=process_group,
                prepared_inputs=prepared,
                generation_plan=plan,
                prefetched_payload_gather=pending,
            )
            result.parameters_by_node["empty"]["value"].sum().backward()

    assert payload_collectives == 0
    assert pending.state == "finished" and pending.work is None
    pack_gradients.assert_not_called()
    reduce_scatter.assert_not_called()
    assert field.grad is not None and tuple(field.grad.shape) == (0, 2)
    torch.distributed.barrier()


def _two_rank_nccl_oracle_worker(rank: int, world_size: int, init_file: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    assert torch.distributed.get_backend() == torch.distributed.Backend.NCCL
    default_group = torch.distributed.distributed_c10d._get_default_group()
    gaussian_group = torch.distributed.new_group(backend="nccl")
    assert gaussian_group is not default_group

    collective_routes: set[str] = set()
    unpatched_all_gather = torch.distributed.all_gather_into_tensor
    unpatched_all_reduce = torch.distributed.all_reduce
    unpatched_reduce_scatter = torch.distributed.reduce_scatter_tensor

    def checked_all_gather(
        output: torch.Tensor,
        local: torch.Tensor,
        group=None,
        async_op: bool = False,
    ) -> Any:
        assert group is gaussian_group, "fused Gaussian AllGather used the default process group"
        if local.dtype == fused_gather._COUNT_METADATA_DTYPE:
            route = "count_packet" if local.numel() == fused_gather._LAYER_COUNT_PACKET_WIDTH else "full_layer_counts"
        elif local.ndim == 0:
            # The independent reference path gathers one scalar row count on
            # the same group; it is not one of the fused protocol's six sites.
            return unpatched_all_gather(output, local, group=group, async_op=async_op)
        elif local.ndim == 1:
            route = "grouped_payload"
        elif local.ndim == 2:
            route = "padded_payload"
        else:
            raise AssertionError(f"unexpected fused Gaussian AllGather input shape {tuple(local.shape)}")
        collective_routes.add(route)
        return unpatched_all_gather(output, local, group=group, async_op=async_op)

    def checked_all_reduce(
        tensor: torch.Tensor,
        op: Any = torch.distributed.ReduceOp.SUM,
        group=None,
        async_op: bool = False,
    ) -> Any:
        assert group is gaussian_group, "fused Gaussian AllReduce used the default process group"
        collective_routes.add("integer_range_validation")
        return unpatched_all_reduce(tensor, op=op, group=group, async_op=async_op)

    def checked_reduce_scatter(
        output: torch.Tensor,
        local: torch.Tensor,
        op: Any = torch.distributed.ReduceOp.SUM,
        group=None,
        async_op: bool = False,
    ) -> Any:
        assert group is gaussian_group, "fused Gaussian ReduceScatter used the default process group"
        collective_routes.add("backward_reduce_scatter")
        return unpatched_reduce_scatter(output, local, op=op, group=group, async_op=async_op)

    collective_patches = (
        patch.object(torch.distributed, "all_gather_into_tensor", side_effect=checked_all_gather),
        patch.object(torch.distributed, "all_reduce", side_effect=checked_all_reduce),
        patch.object(torch.distributed, "reduce_scatter_tensor", side_effect=checked_reduce_scatter),
    )
    try:
        for collective_patch in collective_patches:
            collective_patch.start()
        expected_counts = {
            "first": (1, 2),
            "partially_empty": (0, 1),
            "globally_empty": (0, 0),
            "equal_count": (1, 1),
        }
        inline_node_ids = tuple(expected_counts)
        assert len(inline_node_ids) <= fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY
        for layer_count in (1, 2, 3, 4):
            all_parameters = _make_nccl_oracle_parameters(rank, device)
            selected_node_ids = inline_node_ids[:layer_count]
            inline_parameters = {node_id: all_parameters[node_id] for node_id in selected_node_ids}
            _assert_nccl_oracle_matches_reference(
                inline_parameters,
                {node_id: expected_counts[node_id] for node_id in selected_node_ids},
                expected_metadata_gathers=1,
                process_group=gaussian_group,
            )

        fallback_parameters = _make_nccl_oracle_parameters(rank, device)
        fallback_counts: dict[str, tuple[int, ...]] = {
            "first": (1, 2),
            "partially_empty": (0, 1),
            "globally_empty": (0, 0),
            "equal_count": (1, 1),
            "second": (2, 1),
        }
        for index in range(
            len(fallback_parameters), fused_gather.gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS + 1
        ):
            node_id = f"unused_{index}"
            fallback_parameters[node_id] = {"unused": torch.ones((1, 2), device=device, requires_grad=True)}
            fallback_counts[node_id] = (1, 1)
        assert len(fallback_parameters) > fused_gather.gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS
        _assert_nccl_oracle_matches_reference(
            fallback_parameters,
            fallback_counts,
            expected_metadata_gathers=2,
            process_group=gaussian_group,
        )

        for generation in range(2):
            dynamic_parameters, expected_generation_counts = _make_dynamic_nccl_oracle_parameters(
                rank,
                generation,
                device,
            )
            _assert_nccl_oracle_matches_reference(
                dynamic_parameters,
                expected_generation_counts,
                expected_metadata_gathers=1,
                process_group=gaussian_group,
            )

        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))
        dynamic_counts = ((rank + 1, rank, 0), (2 - rank, 1 - rank, rank + 1))
        expected_dynamic_counts = (((1, 2), (0, 1), (0, 0)), ((2, 1), (1, 0), (1, 2)))
        for local_counts, expected in zip(dynamic_counts, expected_dynamic_counts, strict=True):
            assert (
                fused_gather._all_gather_layer_counts(
                    world_size,
                    local_counts,
                    device,
                    schema_hash,
                    process_group=gaussian_group,
                )
                == expected
            )

        _assert_grouped_nccl_oracle_matches_reference(rank, device, gaussian_group)
        _assert_grouped_nccl_oracle_matches_reference(
            rank,
            device,
            gaussian_group,
            generation=0,
            prefetch_payload=True,
        )
        _assert_grouped_nccl_oracle_matches_reference(
            rank,
            device,
            gaussian_group,
            prefetch_payload=True,
            source_node_ids=("dynamic_rigids", "background", "dynamic_deformables", "road"),
            node_ids=("vehicles", "sky_shell", "cloth", "asphalt"),
        )
        _assert_grouped_nccl_oracle_matches_reference(
            rank,
            device,
            gaussian_group,
            prefetch_payload=True,
            source_node_ids=("road", "dynamic_rigids", "background"),
            node_ids=("surface", "actors", "environment"),
        )
        _assert_grouped_nccl_oracle_matches_reference(
            rank,
            device,
            gaussian_group,
            prefetch_payload=True,
            source_node_ids=("dynamic_deformables", "road"),
            node_ids=("deformable_assets", "ground"),
        )
        _assert_single_node_training_nccl_matches_prior_per_node_path(rank, device, gaussian_group)
        _assert_generation_plan_nccl_reuse_and_refresh(rank, device, gaussian_group)
        for layer_count in (5, 8):
            for prefetch_payload in (False, True):
                _assert_grouped_nccl_oracle_matches_reference(
                    rank,
                    device,
                    gaussian_group,
                    source_node_ids=_LIDARFREE_NODES[:layer_count],
                    node_ids=_LIDARFREE_NODES[:layer_count],
                    prefetch_payload=prefetch_payload,
                    lidarfree=True,
                )
        _assert_generation_plan_nccl_reuse_and_refresh(rank, device, gaussian_group, lidarfree=True)
        _assert_generation_plan_rank_asymmetric_consumption(rank, device, gaussian_group)
        _assert_all_empty_generation_plan(rank, device, gaussian_group)

        # First validate one stable integer topology, then prove its warm
        # forward skips both the value scan and its reduction. A unilateral
        # rank-1 mutation must make rank 0's otherwise-clean cache bit miss as
        # well, so both ranks revalidate and reject before payload AllGather.
        integer_parameters = {
            "integer_payload": {
                "value": torch.ones((1, 1), device=device, dtype=torch.float32, requires_grad=True),
                "id": torch.full((1, 1), rank, device=device, dtype=torch.int64),
            }
        }
        integer_state = fused_gather._PayloadIntegerRangeValidationState()
        payload_gathers = 0
        range_validations = 0
        real_all_gather = torch.distributed.all_gather_into_tensor
        real_validate_integer_ranges = fused_gather._validate_payload_integer_ranges

        def track_integer_payload_gather(
            output: torch.Tensor,
            local: torch.Tensor,
            group=None,
            async_op: bool = False,
        ) -> Any:
            nonlocal payload_gathers
            if local.dtype != fused_gather._COUNT_METADATA_DTYPE:
                payload_gathers += 1
            return real_all_gather(output, local, group=group, async_op=async_op)

        def track_integer_range_validation(*args: Any, **kwargs: Any) -> None:
            nonlocal range_validations
            range_validations += 1
            real_validate_integer_ranges(*args, **kwargs)

        with (
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=track_integer_payload_gather,
            ),
            patch.object(
                fused_gather,
                "_validate_payload_integer_ranges",
                side_effect=track_integer_range_validation,
            ),
        ):
            for _ in range(2):
                fused_gather._fused_all_gather_gaussian_parameters(
                    list(integer_parameters),
                    integer_parameters,
                    process_group=gaussian_group,
                    integer_range_validation_state=integer_state,
                )
            assert range_validations == 1

            if rank == 1:
                integer_parameters["integer_payload"]["id"].fill_((1 << 24) + 1)
            payload_gathers = 0
            try:
                fused_gather._fused_all_gather_gaussian_parameters(
                    list(integer_parameters),
                    integer_parameters,
                    process_group=gaussian_group,
                    integer_range_validation_state=integer_state,
                )
            except RuntimeError as error:
                if "outside its promoted floating dtype's contiguous exact-integer range" not in str(error):
                    raise
            else:
                raise AssertionError("rank-asymmetric integer overflow unexpectedly passed payload validation")
            assert payload_gathers == 0
            assert range_validations == 2

            # A failed generation is never published. Restoring rank 1's
            # value therefore forces another symmetric validation before the
            # recovered payload collective.
            if rank == 1:
                integer_parameters["integer_payload"]["id"].fill_(rank)
            fused_gather._fused_all_gather_gaussian_parameters(
                list(integer_parameters),
                integer_parameters,
                process_group=gaussian_group,
                integer_range_validation_state=integer_state,
            )
            assert range_validations == 3
        torch.distributed.barrier()

        # Martin's cross-layer boundary: every 33-field layer is individually
        # legal, while 4 * 33 = 132 trainable fields exceed the one backward
        # gradient-pack descriptor. Both ranks must reject after the fixed
        # metadata packet and before any payload pack/AllGather.
        over_capacity_parameters = {
            f"layer_{layer}": {
                f"value_{index}": torch.ones((1, 1), device=device, requires_grad=True) for index in range(33)
            }
            for layer in range(4)
        }
        metadata_gathers = 0
        payload_gathers = 0

        def track_capacity_gather(
            output: torch.Tensor,
            local: torch.Tensor,
            group=None,
            async_op: bool = False,
        ) -> Any:
            nonlocal metadata_gathers, payload_gathers
            if local.dtype == fused_gather._COUNT_METADATA_DTYPE:
                metadata_gathers += 1
            else:
                payload_gathers += 1
            return real_all_gather(output, local, group=group, async_op=async_op)

        with (
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=track_capacity_gather,
            ),
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_pack_grouped_payload_trusted",
                wraps=fused_gather.gaussian_collective_payload_ops._pack_grouped_payload_trusted,
            ) as grouped_pack,
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_pack_payload_trusted",
                wraps=fused_gather.gaussian_collective_payload_ops._pack_payload_trusted,
            ) as layer_pack,
        ):
            try:
                fused_gather._fused_all_gather_gaussian_parameters(
                    list(over_capacity_parameters),
                    over_capacity_parameters,
                    process_group=gaussian_group,
                )
            except RuntimeError as error:
                if "supports at most 128 trainable input tensors, got 132" not in str(error):
                    raise
            else:
                raise AssertionError("cross-layer gradient-pack overflow unexpectedly passed schema validation")
        assert metadata_gathers == 1
        assert payload_gathers == 0
        grouped_pack.assert_not_called()
        layer_pack.assert_not_called()
        torch.distributed.barrier()

        # Recover on the same process group at the exact 128-input boundary.
        # Each layer retains one additional frozen forward field, proving that
        # only trainable fields consume gradient descriptor slots.
        at_capacity_parameters = {
            f"layer_{layer}": {
                f"value_{index}": torch.ones(
                    (1, 1),
                    device=device,
                    requires_grad=index < 32,
                )
                for index in range(33)
            }
            for layer in range(4)
        }
        at_capacity_result = fused_gather._fused_all_gather_gaussian_parameters(
            list(at_capacity_parameters),
            at_capacity_parameters,
            process_group=gaussian_group,
        )
        at_capacity_loss = torch.zeros((), device=device)
        for parameters in at_capacity_result.parameters_by_node.values():
            for field in parameters.values():
                at_capacity_loss = at_capacity_loss + field.sum()
        at_capacity_loss.backward()
        trainable_count = 0
        for parameters in at_capacity_parameters.values():
            for field in parameters.values():
                if field.requires_grad:
                    trainable_count += 1
                    assert field.grad is not None
                    torch.testing.assert_close(
                        field.grad,
                        torch.full_like(field, world_size),
                        rtol=0,
                        atol=0,
                    )
                else:
                    assert field.grad is None
        assert trainable_count == fused_gather.gaussian_collective_payload_ops._MAX_GRADIENT_PACK_INPUTS
        torch.distributed.barrier()

        # An exception in FreePose must drain the already-entered payload on
        # every rank. A barrier and a second dynamic generation prove that the
        # NCCL sequence remains usable after the abandoned forward.
        abandoned_parameters = _make_grouped_nccl_parameters(rank, device, generation=1)
        abandoned_prepared = fused_gather._prepare_fused_gather_inputs(
            list(abandoned_parameters),
            abandoned_parameters,
            process_group=gaussian_group,
        )
        abandoned_plan = fused_gather._resolve_fused_gather_generation_plan(
            abandoned_prepared,
            process_group=gaussian_group,
        )
        abandoned_pending = fused_gather._start_fused_payload_gather_from_generation_plan(
            abandoned_prepared,
            abandoned_plan,
            process_group=gaussian_group,
        )
        fused_gather._discard_fused_payload_gather(abandoned_pending)
        assert abandoned_pending.state == "discarded"
        torch.distributed.barrier()
        _assert_grouped_nccl_oracle_matches_reference(
            rank,
            device,
            gaussian_group,
            generation=1,
            prefetch_payload=True,
        )
        torch.cuda.synchronize(device)

        inline_capacity = fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY
        asymmetric_arity_counts = (1,) * (inline_capacity if rank == 0 else inline_capacity + 1)
        try:
            asymmetric_arity_pending = fused_gather._start_layer_count_gather(
                world_size,
                asymmetric_arity_counts,
                device,
                schema_hash,
                process_group=gaussian_group,
            )
            fused_gather._finish_layer_count_gather(asymmetric_arity_pending)
        except RuntimeError as error:
            if "layer count differs across ranks" not in str(error):
                raise
        else:
            raise AssertionError("rank-asymmetric arity unexpectedly passed metadata validation")
        torch.distributed.barrier()
        asymmetric_parameters = {
            "first": {
                "value": torch.ones(1, 1, device=device, dtype=torch.float64, requires_grad=rank == 0),
            },
            "second": {
                "value": torch.ones(1, 1, device=device, requires_grad=True),
            },
        }
        try:
            fused_gather._fused_all_gather_gaussian_parameters(
                list(asymmetric_parameters),
                asymmetric_parameters,
                process_group=gaussian_group,
            )
        except RuntimeError as error:
            if "schema differs across ranks" not in str(error):
                raise
        else:
            raise AssertionError("rank-asymmetric requires_grad unexpectedly passed schema validation")

        torch.distributed.barrier()
        grouped_asymmetric_parameters = _make_grouped_nccl_parameters(rank, device)
        if rank == 0:
            grouped_asymmetric_parameters["road"]["half"].requires_grad_(False)
        try:
            grouped_asymmetric_prepared = fused_gather._prepare_fused_gather_inputs(
                list(grouped_asymmetric_parameters),
                grouped_asymmetric_parameters,
                process_group=gaussian_group,
            )
            fused_gather._resolve_fused_gather_generation_plan(
                grouped_asymmetric_prepared,
                process_group=gaussian_group,
            )
        except RuntimeError as error:
            if "schema differs across ranks" not in str(error):
                raise
        else:
            raise AssertionError("rank-asymmetric grouped payload unexpectedly passed schema validation")
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
        assert collective_routes == {
            "count_packet",
            "full_layer_counts",
            "padded_payload",
            "grouped_payload",
            "integer_range_validation",
            "backward_reduce_scatter",
        }
    finally:
        for collective_patch in reversed(collective_patches):
            collective_patch.stop()
        torch.distributed.destroy_process_group()


def _selective_adam_visibility_regression_worker(rank: int, world_size: int, init_file: str) -> None:
    """Reproduce the owner-local bug, then prove the asynchronous union fixes it."""
    # Keep this heavyweight apex/gsplat/config import out of the lightweight
    # NCCL oracle subprocess, whose startup has an independent 45 s deadline.
    from nurec.utils.optim import SelectiveAdam

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    assert torch.distributed.get_backend() == torch.distributed.Backend.NCCL
    default_group = torch.distributed.distributed_c10d._get_default_group()
    gaussian_group = torch.distributed.new_group(backend="nccl")
    assert gaussian_group is not default_group

    def run_step(*, use_global_union: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parameter = torch.nn.Parameter(torch.full((1, 1), 10.0 + rank, device=device))
        dummy = torch.nn.Parameter(torch.zeros((1, 1), device=device))
        optimizer = SelectiveAdam([parameter], lr=0.1, eps=1e-15, selective_enabled=True)
        gathered = fused_gather._fused_all_gather_gaussian_parameters(
            ["layer", "dummy"],
            {
                "layer": {"value": parameter},
                "dummy": {"value": dummy},
            },
            process_group=gaussian_group,
        ).parameters_by_node["layer"]["value"]

        # Rank 1 alone sees and differentiates rank 0's Gaussian. Both ranks
        # participate in the same fused backward, which ReduceScatters SUM to
        # the owner and gives rank 0 a nonzero local gradient.
        global_renderer_visibility = torch.tensor(
            [rank == 1, False],
            device=device,
            dtype=torch.bool,
        )
        if use_global_union:
            pending = prepare_async_visibility_union(
                [global_renderer_visibility],
                element_count=world_size,
                device=device,
                local_ranges=((rank, 1),),
                positive_only=False,
            )
            pending.launch()
        else:
            pending = None

        loss = gathered[0].sum() * (1.0 if rank == 1 else 0.0)
        loss.backward()
        assert parameter.grad is not None
        gradient = parameter.grad.detach().clone()

        # The local-only arm gives each owner only its own renderer's visibility.
        # The global-union arm resolves the cross-rank OR launched above.
        owner_visibility = pending.resolve() if pending is not None else global_renderer_visibility[rank : rank + 1]
        before = parameter.detach().clone()
        optimizer.set_visibility(owner_visibility)
        optimizer.step()
        torch.cuda.synchronize(device)
        return gradient, owner_visibility.detach().clone(), before, parameter.detach().clone()

    try:
        local_gradient, local_visibility, local_before, local_after = run_step(use_global_union=False)
        union_gradient, union_visibility, union_before, union_after = run_step(use_global_union=True)

        if rank == 0:
            torch.testing.assert_close(local_gradient, torch.ones_like(local_gradient), rtol=0, atol=0)
            assert not bool(local_visibility[0])
            torch.testing.assert_close(local_after, local_before, rtol=0, atol=0)

            torch.testing.assert_close(union_gradient, torch.ones_like(union_gradient), rtol=0, atol=0)
            assert bool(union_visibility[0])
            assert not torch.equal(union_after, union_before)
        else:
            torch.testing.assert_close(local_gradient, torch.zeros_like(local_gradient), rtol=0, atol=0)
            torch.testing.assert_close(union_gradient, torch.zeros_like(union_gradient), rtol=0, atol=0)
            assert not bool(local_visibility[0])
            assert not bool(union_visibility[0])
            torch.testing.assert_close(local_after, local_before, rtol=0, atol=0)
            torch.testing.assert_close(union_after, union_before, rtol=0, atol=0)

        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


class TestFusedParameterGather(unittest.TestCase):
    def test_grouped_payload_flatten_preserves_native_contiguous_fields(self) -> None:
        native = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        vector = torch.arange(3, dtype=torch.float32)
        volume = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        noncontiguous = torch.arange(12, dtype=torch.float32).reshape(4, 3).t()
        layout = fused_gather._make_layer_layout(
            "background",
            {
                "native": native,
                "vector": vector,
                "volume": volume,
                "noncontiguous": noncontiguous,
            },
        )

        (flattened,) = fused_gather._flatten_grouped_payload_fields(
            (layout,),
            ((native, vector, volume, noncontiguous),),
        )

        self.assertIs(flattened[0], native)
        self.assertIsNot(flattened[1], vector)
        self.assertIsNot(flattened[2], volume)
        self.assertIsNot(flattened[3], noncontiguous)
        self.assertEqual(tuple(tensor.shape for tensor in flattened), ((3, 4), (3, 1), (3, 6), (3, 4)))
        self.assertTrue(all(tensor.is_contiguous() for tensor in flattened))
        torch.testing.assert_close(flattened[1], vector.reshape(3, 1))
        torch.testing.assert_close(flattened[2], volume.reshape(3, 6))
        torch.testing.assert_close(flattened[3], noncontiguous.contiguous())

    def test_layouts_are_frozen_and_slotted(self) -> None:
        layer_layout = fused_gather._make_layer_layout(
            "road",
            {"value": torch.ones(1, 2, dtype=torch.float32)},
        )
        pack_plan = fused_gather._GradientPackPlan(
            copy_rows=torch.empty(0, dtype=torch.int64),
            output_numel=0,
            num_blocks=0,
        )
        unpack_layout = fused_gather._PayloadUnpackLayout(
            segments=torch.empty(0, dtype=torch.int64),
            output_numel=0,
            num_blocks=0,
        )
        grouped_layout = fused_gather._GroupedPayloadLayout(
            layer_offsets=(0,),
            layer_spans=(2,),
            rank_stride=2,
        )

        self.assertFalse(hasattr(layer_layout, "__dict__"))
        self.assertFalse(hasattr(pack_plan, "__dict__"))
        self.assertFalse(hasattr(unpack_layout, "__dict__"))
        self.assertFalse(hasattr(grouped_layout, "__dict__"))
        self.assertEqual(layer_layout.payload_dtype, torch.float32)
        with self.assertRaises(AttributeError):
            setattr(layer_layout, "node_id", "changed")

    def test_pending_payload_discard_drains_once_and_is_one_shot(self) -> None:
        work = Mock()
        pending = fused_gather._PendingFusedPayloadGather(
            local_grouped_payload=torch.empty(0),
            grouped_gathered_payload=torch.empty(0),
            work=work,
        )

        fused_gather._discard_fused_payload_gather(pending)
        fused_gather._discard_fused_payload_gather(pending)

        work.wait.assert_called_once_with()
        self.assertEqual(pending.state, "discarded")
        self.assertIsNone(pending.local_grouped_payload)
        self.assertIsNone(pending.grouped_gathered_payload)
        self.assertIsNone(pending.work)
        with self.assertRaisesRegex(RuntimeError, "already discarded"):
            fused_gather._finish_fused_payload_gather(pending)

    def test_pending_payload_drain_failure_still_closes_ownership(self) -> None:
        work = Mock()
        work.wait.side_effect = RuntimeError("injected drain failure")
        pending = fused_gather._PendingFusedPayloadGather(
            local_grouped_payload=torch.empty(0),
            grouped_gathered_payload=torch.empty(0),
            work=work,
        )

        with self.assertRaisesRegex(RuntimeError, "injected drain failure"):
            fused_gather._discard_fused_payload_gather(pending)

        self.assertEqual(pending.state, "discarded")
        fused_gather._discard_fused_payload_gather(pending)
        work.wait.assert_called_once_with()

    def test_payload_unpack_metadata_covers_uneven_and_empty_rank_fields(self) -> None:
        segments, output_numel, num_blocks = fused_gather._payload_unpack_segment_rows(
            counts=(2, 0, 1),
            widths=(2, 0, 1),
            max_count=2,
            field_major=False,
        )

        self.assertEqual(
            segments,
            (
                (0, 4, 0, 2, 1),
                (4, 2, 12, 2, 2),
                (6, 2, 2, 1, 3),
                (8, 1, 14, 1, 4),
            ),
        )
        self.assertEqual(output_numel, 9)
        self.assertEqual(num_blocks, 4)

    def test_payload_unpack_metadata_rejects_invalid_topology(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be nonnegative"):
            fused_gather._payload_unpack_segment_rows((2, -1), (3,), 2)
        with self.assertRaisesRegex(ValueError, "does not match"):
            fused_gather._payload_unpack_segment_rows((2, 1), (3,), 1)
        with self.assertRaisesRegex(ValueError, "widths must be nonnegative"):
            fused_gather._payload_unpack_segment_rows((2, 1), (3, -1), 2)

    def test_grouped_payload_layout_preserves_old_padded_spans_with_empty_layer(self) -> None:
        layouts = tuple(
            fused_gather._make_layer_layout(node_id, parameters)
            for node_id, parameters in (
                ("road", {"first": torch.ones(1, 2), "second": torch.ones(1, 1)}),
                ("dynamic_rigids", {"value": torch.empty(0, 5)}),
                ("dynamic_deformables", {"value": torch.ones(1, 4)}),
            )
        )
        counts_by_layer = ((2, 1), (0, 0), (3, 1))

        grouped = fused_gather._make_grouped_payload_layout(layouts, counts_by_layer)

        self.assertEqual(grouped.layer_offsets, (0, 6, 6))
        self.assertEqual(grouped.layer_spans, (6, 0, 12))
        self.assertEqual(grouped.rank_stride, 18)
        self.assertEqual(grouped.rank_stride, sum(grouped.layer_spans))

    def test_grouped_payload_layout_rejects_malformed_topology(self) -> None:
        layout = fused_gather._make_layer_layout("road", {"value": torch.ones(1, 2)})
        invalid_calls = (
            ((layout,), (), "layouts but 0 count vectors"),
            ((), (), "requires at least one layer"),
            ((layout,), ((),), "requires at least one rank"),
            ((layout,), ((1, -1),), "negative counts"),
            ((layout, layout), ((1, 2), (1,)), "has 1 ranks, expected 2"),
        )
        for layouts, counts, message in invalid_calls:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                fused_gather._make_grouped_payload_layout(layouts, counts)

    def test_grouped_payload_unpack_metadata_covers_72_ranks(self) -> None:
        counts_by_layer = tuple(tuple((rank * (layer + 1) + layer) % 4 for rank in range(72)) for layer in range(4))
        widths_by_layer = ((2, 0, 1), (4,), (1, 3, 2), (5, 1))
        layer_spans = tuple(
            max(counts) * sum(widths) for counts, widths in zip(counts_by_layer, widths_by_layer, strict=True)
        )
        grouped_layout = fused_gather._GroupedPayloadLayout(
            layer_offsets=tuple(sum(layer_spans[:layer]) for layer in range(4)),
            layer_spans=layer_spans,
            rank_stride=sum(layer_spans),
        )

        segments, output_numels, num_blocks = fused_gather._grouped_payload_unpack_segment_rows(
            counts_by_layer,
            widths_by_layer,
            grouped_layout,
        )

        self.assertEqual(
            output_numels,
            tuple(sum(counts) * sum(widths) for counts, widths in zip(counts_by_layer, widths_by_layer, strict=True)),
        )
        self.assertEqual(
            len(segments),
            sum(
                sum(count > 0 for count in counts) * sum(width > 0 for width in widths)
                for counts, widths in zip(counts_by_layer, widths_by_layer, strict=True)
            ),
        )
        self.assertEqual(segments[-1][-1], num_blocks)
        self.assertTrue(all(0 <= output_index < 4 for output_index, *_rest in segments))
        self.assertTrue(
            all(
                0 <= source <= 72 * grouped_layout.rank_stride - length
                for _output, _destination, length, source, _block_end in segments
            )
        )

    def test_grouped_payload_unpack_metadata_rejects_stale_group_geometry(self) -> None:
        counts_by_layer = ((2, 1), (1, 2))
        widths_by_layer = ((2, 1), (3,))
        valid = fused_gather._GroupedPayloadLayout(
            layer_offsets=(0, 6),
            layer_spans=(6, 6),
            rank_stride=12,
        )
        stale = replace(valid, layer_offsets=(0, 7))

        with self.assertRaisesRegex(ValueError, "starts at 7, expected 6"):
            fused_gather._grouped_payload_unpack_segment_rows(counts_by_layer, widths_by_layer, stale)

    def test_grouped_payload_gate_is_rank_independent_and_layout_driven(self) -> None:
        def make_layout(node_id: str, dtype: torch.dtype = torch.float32, fields: int = 1):
            return fused_gather._make_layer_layout(
                node_id,
                {f"value_{index}": torch.ones(1, 1, dtype=dtype) for index in range(fields)},
            )

        layouts = tuple(make_layout(node_id) for node_id in _NODES)
        cuda_zero = torch.device("cuda:0")
        devices = tuple(tuple(cuda_zero for _ in layout.keys) for layout in layouts)

        for layer_count in range(1, len(layouts) + 1):
            self.assertTrue(fused_gather._can_group_payloads(layouts[:layer_count], devices[:layer_count]))
        self.assertTrue(fused_gather._can_group_payloads((layouts[1], layouts[0], *layouts[2:]), devices))
        self.assertFalse(fused_gather._can_group_payloads((), ()))

        max_grouped_layers = fused_gather.gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS
        over_layer_capacity_layouts = tuple(make_layout(f"capacity_{index}") for index in range(max_grouped_layers + 1))
        over_layer_capacity_devices = tuple(
            tuple(cuda_zero for _ in layout.keys) for layout in over_layer_capacity_layouts
        )
        self.assertTrue(
            fused_gather._can_group_payloads(
                over_layer_capacity_layouts[:max_grouped_layers],
                over_layer_capacity_devices[:max_grouped_layers],
            )
        )
        self.assertFalse(fused_gather._can_group_payloads(over_layer_capacity_layouts, over_layer_capacity_devices))

        mixed_dtype_layouts = (
            layouts[0],
            layouts[1],
            make_layout("dynamic_rigids", torch.float64),
            layouts[3],
        )
        self.assertFalse(fused_gather._can_group_payloads(mixed_dtype_layouts, devices))

        wide_layer_layouts = (
            layouts[0],
            layouts[1],
            make_layout(
                "dynamic_rigids",
                fields=17,
            ),
            layouts[3],
        )
        wide_layer_devices = tuple(tuple(cuda_zero for _ in layout.keys) for layout in wide_layer_layouts)
        self.assertTrue(fused_gather._can_group_payloads(wide_layer_layouts, wide_layer_devices))

        native_over_capacity_layouts = (
            layouts[0],
            layouts[1],
            make_layout(
                "dynamic_rigids",
                fields=64,
            ),
            layouts[3],
        )
        over_capacity_devices = tuple(tuple(cuda_zero for _ in layout.keys) for layout in native_over_capacity_layouts)
        # Python owns only rank-independent layout compatibility. The grouped
        # native entry point owns and validates its private descriptor capacity.
        self.assertTrue(fused_gather._can_group_payloads(native_over_capacity_layouts, over_capacity_devices))

        mixed_devices = (
            devices[0],
            devices[1],
            (torch.device("cuda:1"),),
            devices[3],
        )
        self.assertFalse(fused_gather._can_group_payloads(layouts, mixed_devices))
        cpu_devices = tuple(tuple(torch.device("cpu") for _ in layout.keys) for layout in layouts)
        self.assertFalse(fused_gather._can_group_payloads(layouts, cpu_devices))

    def test_grouped_payload_route_owns_node_capacity_and_distinctness(self) -> None:
        capacity = fused_gather.gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS

        self.assertFalse(fused_gather._can_route_node_selection(()))
        self.assertTrue(fused_gather._can_route_node_selection(tuple(f"node_{index}" for index in range(capacity))))
        self.assertFalse(
            fused_gather._can_route_node_selection(tuple(f"node_{index}" for index in range(capacity + 1)))
        )
        self.assertFalse(fused_gather._can_route_node_selection(("duplicate", "duplicate")))

    def test_lidarfree_layer_roster_supports_grouped_payload_and_prefetch(self) -> None:
        parameters = _make_grouped_nccl_parameters(0, torch.device("cpu"), lidarfree=True)
        layouts = tuple(fused_gather._make_layer_layout(node, fields) for node, fields in parameters.items())
        devices = tuple(tuple(torch.device("cuda:0") for _ in layout.keys) for layout in layouts)
        self.assertEqual(tuple(parameters), _LIDARFREE_NODES)
        self.assertTrue(fused_gather._can_route_node_selection(_LIDARFREE_NODES))
        self.assertTrue(fused_gather._can_group_payloads(layouts, devices))
        fused_gather._validate_fused_schema(
            layouts,
            tuple(field.requires_grad for fields in parameters.values() for field in fields.values()),
        )

    def test_cuda_gradient_pack_plan_covers_fields_and_padding(self) -> None:
        copy_rows, output_numel, num_blocks = fused_gather._gradient_pack_plan_rows(
            counts_by_layer=((2, 1), (1, 2)),
            field_layers=(0, 0, 1),
            field_widths=(2, 1, 3),
            active_fields=(0, 2),
            sizes_by_rank=(7, 8),
        )

        self.assertEqual(
            copy_rows,
            (
                (0, 4, 0, 0, 1),
                (4, 3, 1, 0, 2),
                (7, 1, -1, 0, 3),
                (8, 2, 0, 4, 4),
                (10, 6, 1, 3, 5),
            ),
        )
        self.assertEqual(output_numel, 16)
        self.assertEqual(num_blocks, 5)

    def test_cuda_gradient_pack_plan_enforces_trusted_row_invariants(self) -> None:
        counts_by_layer = ((5000, 0), (1, 2))
        field_layers = (0, 1)
        field_widths = (1, 3)
        active_fields = (0, 1)
        copy_rows, output_numel, num_blocks = fused_gather._gradient_pack_plan_rows(
            counts_by_layer=counts_by_layer,
            field_layers=field_layers,
            field_widths=field_widths,
            active_fields=active_fields,
            sizes_by_rank=(5003, 6),
        )

        destination = 0
        expected_block_end = 0
        source_offsets = [0, 0]
        values_per_block = int(fused_gather.gaussian_collective_payload_ops._GRADIENT_PACK_VALUES_PER_BLOCK)
        for row_destination, length, input_index, source_offset, block_end in copy_rows:
            self.assertEqual(row_destination, destination)
            self.assertGreater(length, 0)
            if input_index == -1:
                self.assertEqual(source_offset, 0)
            else:
                self.assertIn(input_index, range(len(active_fields)))
                self.assertEqual(source_offset, source_offsets[input_index])
                source_offsets[input_index] += length
            destination += length
            expected_block_end += (length + values_per_block - 1) // values_per_block
            self.assertEqual(block_end, expected_block_end)

        self.assertTrue(all(length > 0 for _, length, _, _, _ in copy_rows))
        self.assertTrue(any(length > values_per_block for _, length, _, _, _ in copy_rows))
        self.assertTrue(any(input_index == -1 for _, _, input_index, _, _ in copy_rows))
        self.assertEqual(destination, output_numel)
        self.assertEqual(source_offsets, [5000, 9])
        self.assertEqual(expected_block_end, num_blocks)

    def test_cuda_gradient_pack_plan_rejects_invalid_generated_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows contain 7 values, expected 6"):
            fused_gather._gradient_pack_plan_rows(
                counts_by_layer=((2, 1), (1, 2)),
                field_layers=(0, 1),
                field_widths=(2, 3),
                active_fields=(0, 1),
                sizes_by_rank=(6, 8),
            )

        with self.assertRaisesRegex(ValueError, "negative packed length"):
            fused_gather._gradient_pack_plan_rows(
                counts_by_layer=((-1,),),
                field_layers=(0,),
                field_widths=(1,),
                active_fields=(0,),
                sizes_by_rank=(1,),
            )

    def test_cuda_gradient_pack_plan_cache_tracks_topology_and_stream(self) -> None:
        fused_gather._cached_gradient_pack_plan.cache_clear()
        args = (
            ((2, 1), (1, 2)),
            (0, 0, 1),
            (2, 1, 3),
            (0, 2),
            torch.device("cuda"),
        )
        try:
            with patch.object(fused_gather.torch, "tensor", return_value=torch.empty(0)) as upload:
                first = fused_gather._cached_gradient_pack_plan(*args, 7)
                same = fused_gather._cached_gradient_pack_plan(*args, 7)
                changed_counts = fused_gather._cached_gradient_pack_plan(
                    ((3, 1), (1, 2)),
                    *args[1:],
                    7,
                )
                changed_stream = fused_gather._cached_gradient_pack_plan(*args, 8)

            self.assertIs(first, same)
            self.assertIsNot(first, changed_counts)
            self.assertIsNot(first, changed_stream)
            self.assertEqual(upload.call_count, 3)
            cache_info = fused_gather._cached_gradient_pack_plan.cache_info()
            self.assertEqual(cache_info.hits, 1)
            self.assertEqual(cache_info.misses, 3)
        finally:
            fused_gather._cached_gradient_pack_plan.cache_clear()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_layer_payload_round_trip_preserves_layout(self) -> None:
        source = torch.arange(24, dtype=torch.float32, device="cuda").reshape(3, 8)
        parameters = {
            "position": source[:, ::2],
            "empty": torch.empty(3, 0, device="cuda"),
            "ids": torch.tensor([[1], [2], [3]], dtype=torch.int64, device="cuda"),
        }

        layout = fused_gather._make_layer_layout("road", parameters)
        payload = fused_gather._pack_layer_payload(layout, tuple(parameters.values()))
        unpacked = fused_gather._unpack_layer_payload(layout, payload, (3,), 3)

        self.assertEqual(tuple(unpacked), tuple(parameters))
        for key, expected in parameters.items():
            self.assertEqual(unpacked[key].dtype, expected.dtype)
            self.assertEqual(unpacked[key].shape, expected.shape)
            self.assertTrue(unpacked[key].is_contiguous())
            torch.testing.assert_close(unpacked[key], expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_layer_payload_promotes_mixed_float_dtypes(self) -> None:
        half = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16, device="cuda")
        single = torch.tensor([[5.0], [6.0]], dtype=torch.float32, device="cuda")
        parameters = {"half": half, "single": single}

        layout = fused_gather._make_layer_layout("mixed", parameters)
        payload = fused_gather._pack_layer_payload(layout, tuple(parameters.values()))
        unpacked = fused_gather._unpack_layer_payload(layout, payload, (2,), 2)

        self.assertEqual(layout.payload_dtype, torch.float32)
        self.assertEqual(payload.dtype, torch.float32)
        self.assertEqual(unpacked["half"].dtype, torch.float16)
        self.assertEqual(unpacked["single"].dtype, torch.float32)
        self.assertTrue(unpacked["half"].is_contiguous())
        self.assertTrue(unpacked["single"].is_contiguous())
        torch.testing.assert_close(unpacked["half"], half, rtol=0, atol=0)
        torch.testing.assert_close(unpacked["single"], single, rtol=0, atol=0)

    def test_communication_schema_hash_tracks_collective_topology(self) -> None:
        first = fused_gather._make_layer_layout(
            "first",
            {
                "position": torch.ones(1, 3, dtype=torch.float32),
                "opacity": torch.ones(1, 1, dtype=torch.float16),
            },
        )
        second = fused_gather._make_layer_layout(
            "second",
            {"features": torch.ones(1, 2, 4, dtype=torch.float32)},
        )
        baseline = fused_gather._communication_schema_hash((first, second), (True, False, True))
        same_topology_with_other_row_counts = fused_gather._make_layer_layout(
            "first",
            {
                "position": torch.ones(9, 3, dtype=torch.float32),
                "opacity": torch.ones(9, 1, dtype=torch.float16),
            },
        )

        self.assertEqual(baseline, fused_gather._communication_schema_hash((first, second), (True, False, True)))
        self.assertEqual(
            baseline,
            fused_gather._communication_schema_hash(
                (same_topology_with_other_row_counts, second),
                (True, False, True),
            ),
        )
        variants = (
            fused_gather._communication_schema_hash((second, first), (True, True, False)),
            fused_gather._communication_schema_hash((first, second), (False, False, True)),
            fused_gather._communication_schema_hash(
                (
                    fused_gather._make_layer_layout(
                        "first",
                        {
                            "opacity": torch.ones(1, 1, dtype=torch.float16),
                            "position": torch.ones(1, 3, dtype=torch.float32),
                        },
                    ),
                    second,
                ),
                (False, True, True),
            ),
            fused_gather._communication_schema_hash(
                (
                    fused_gather._make_layer_layout(
                        "first",
                        {
                            "position": torch.ones(1, 4, dtype=torch.float32),
                            "opacity": torch.ones(1, 1, dtype=torch.float16),
                        },
                    ),
                    second,
                ),
                (True, False, True),
            ),
            fused_gather._communication_schema_hash(
                (
                    fused_gather._make_layer_layout(
                        "first",
                        {
                            "position": torch.ones(1, 3, dtype=torch.float64),
                            "opacity": torch.ones(1, 1, dtype=torch.float16),
                        },
                    ),
                    second,
                ),
                (True, False, True),
            ),
        )
        for schema_hash in variants:
            self.assertNotEqual(schema_hash, baseline)

    def test_integer_range_state_tracks_weak_identity_version_schema_and_rank_generation(self) -> None:
        values = torch.ones(2, 1)
        ids = torch.tensor([[1], [2]], dtype=torch.int64)
        layout = fused_gather._make_layer_layout("node", {"value": values, "id": ids})
        schema_hash = fused_gather._communication_schema_hash((layout,), (False, False))
        state = fused_gather._PayloadIntegerRangeValidationState()

        state.observe((layout,), ((values, ids),), schema_hash, 0, 2)
        self.assertEqual(state.generation, 1)
        self.assertFalse(state.local_ranges_cached)
        state.mark_validated()
        state.observe((layout,), ((values, ids),), schema_hash, 0, 2)
        self.assertEqual(state.generation, 1)
        self.assertTrue(state.local_ranges_cached)

        # Floating optimizer updates are irrelevant to integer exactness.
        values.add_(1)
        state.observe((layout,), ((values, ids),), schema_hash, 0, 2)
        self.assertEqual(state.generation, 1)
        self.assertTrue(state.local_ranges_cached)

        ids.add_(1)
        state.observe((layout,), ((values, ids),), schema_hash, 0, 2)
        self.assertEqual(state.generation, 2)
        self.assertFalse(state.local_ranges_cached)
        state.mark_validated()

        replacement = ids.clone()
        replacement_ref = weakref.ref(replacement)
        replacement_layout = fused_gather._make_layer_layout("node", {"value": values, "id": replacement})
        state.observe((replacement_layout,), ((values, replacement),), schema_hash, 0, 2)
        self.assertEqual(state.generation, 3)
        self.assertFalse(state.local_ranges_cached)
        state.mark_validated()

        changed_schema_hash = tuple(word + 1 for word in schema_hash)
        state.observe((replacement_layout,), ((values, replacement),), changed_schema_hash, 0, 2)
        self.assertEqual(state.generation, 4)
        state.mark_validated()
        state.observe((replacement_layout,), ((values, replacement),), changed_schema_hash, 1, 2)
        self.assertEqual(state.generation, 5)

        del replacement
        gc.collect()
        self.assertIsNone(replacement_ref())

    def test_layer_count_packet_capacity_is_independent_of_cuda_grouped_capacity(self) -> None:
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))
        counts = tuple(range(1, fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY + 1))
        baseline = fused_gather._make_layer_count_packet(counts, torch.device("cpu"), schema_hash)
        self.assertEqual(len(counts), 8)
        self.assertEqual(baseline.numel(), 19)

        for cuda_capacity in (1, 8):
            with (
                self.subTest(cuda_capacity=cuda_capacity),
                patch.object(
                    fused_gather.gaussian_collective_payload_ops,
                    "_MAX_GROUPED_PAYLOAD_LAYERS",
                    cuda_capacity,
                ),
            ):
                packet = fused_gather._make_layer_count_packet(counts, torch.device("cpu"), schema_hash)
                torch.testing.assert_close(packet, baseline, rtol=0, atol=0)
                self.assertEqual(packet.numel(), fused_gather._LAYER_COUNT_PACKET_WIDTH)
                decoded = fused_gather._decode_layer_count_packet(torch.cat((packet, packet)), 2)
                self.assertEqual(decoded.inline_counts, tuple((count, count) for count in counts))

    def test_integer_range_state_does_not_fail_before_agreement_for_inference_or_unsupported_tensors(self) -> None:
        with torch.inference_mode():
            values = torch.ones(1, 1)
            ids = torch.ones(1, 1, dtype=torch.int64)
        layout = fused_gather._make_layer_layout("inference", {"value": values, "id": ids})
        schema_hash = fused_gather._communication_schema_hash((layout,), (False, False))
        state = fused_gather._PayloadIntegerRangeValidationState()
        state.observe((layout,), ((values, ids),), schema_hash, 0, 2)
        first_generation = state.generation
        state.mark_validated()
        state.observe((layout,), ((values, ids),), schema_hash, 0, 2)
        self.assertEqual(state.generation, first_generation + 1)
        self.assertFalse(state.local_ranges_cached)

        unsupported = torch.ones(1, 1, dtype=torch.complex64)
        unsupported_layout = fused_gather._make_layer_layout("unsupported", {"value": unsupported})
        unsupported_hash = fused_gather._communication_schema_hash((unsupported_layout,), (False,))
        state.observe((unsupported_layout,), ((unsupported,),), unsupported_hash, 0, 2)

    def test_layer_count_packet_cache_bit_requires_every_rank_and_rejects_protocol_drift(self) -> None:
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))
        cached = fused_gather._make_layer_count_packet(
            (2,),
            torch.device("cpu"),
            schema_hash,
            integer_ranges_cached=True,
        )
        dirty = fused_gather._make_layer_count_packet((3,), torch.device("cpu"), schema_hash)
        decoded = fused_gather._decode_layer_count_packet(torch.cat((cached, dirty)), 2)
        self.assertFalse(decoded.all_integer_ranges_cached)
        self.assertEqual(decoded.inline_counts, ((2, 3),))

        incompatible = dirty.clone()
        incompatible[fused_gather._LAYER_COUNT_PACKET_PROTOCOL_VERSION_INDEX] += 1
        with self.assertRaisesRegex(RuntimeError, "protocol version mismatch"):
            fused_gather._decode_layer_count_packet(torch.cat((cached, incompatible)), 2)

    def test_non_cuda_pack_is_rejected_before_payload_collectives(self) -> None:
        parameters = {
            "first": {"value": torch.ones(1, 1, requires_grad=True)},
            "second": {"value": torch.ones(1, 1, requires_grad=True)},
        }
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(fused_gather, "_all_gather_layer_counts") as gather_counts,
            self.assertRaisesRegex(RuntimeError, "requires CUDA tensors"),
        ):
            fused_gather._fused_all_gather_gaussian_parameters(
                ["first", "second"],
                parameters,
                process_group=cast(torch.distributed.ProcessGroup, object()),
            )

        gather_counts.assert_not_called()

    def test_schema_validation_enforces_exported_gradient_pack_capacity(self) -> None:
        capacity = fused_gather.gaussian_collective_payload_ops._MAX_GRADIENT_PACK_INPUTS
        self.assertEqual(capacity, 128)
        fields_per_layer = 33
        layouts = tuple(
            fused_gather._make_layer_layout(
                f"layer_{layer}",
                {f"value_{index}": torch.ones(1, 1) for index in range(fields_per_layer)},
            )
            for layer in range(4)
        )
        at_capacity_flags = tuple(index < 32 for _layer in range(4) for index in range(fields_per_layer))
        over_capacity_flags = (True,) * (4 * fields_per_layer)
        self.assertEqual(sum(at_capacity_flags), capacity)

        # Frozen fields remain legal forward payload inputs and do not consume
        # slots in the backward gradient-pack descriptor.
        fused_gather._validate_fused_schema(layouts, at_capacity_flags)
        with self.assertRaisesRegex(
            RuntimeError,
            rf"supports at most {capacity} trainable input tensors, got {len(over_capacity_flags)}",
        ):
            fused_gather._validate_fused_schema(layouts, over_capacity_flags)

    def test_cross_layer_gradient_capacity_rejects_after_metadata_before_payload(self) -> None:
        fields_per_layer = 33
        fields_by_layer = tuple(
            tuple(torch.ones(1, 1, requires_grad=True) for _ in range(fields_per_layer)) for _ in range(4)
        )
        layouts = tuple(
            fused_gather._make_layer_layout(
                f"layer_{layer}",
                {f"value_{index}": field for index, field in enumerate(fields)},
            )
            for layer, fields in enumerate(fields_by_layer)
        )
        fields = tuple(field for layer_fields in fields_by_layer for field in layer_fields)
        counts = ((1, 1),) * len(layouts)

        with (
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=counts) as gather_counts,
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_pack_grouped_payload_trusted",
            ) as grouped_pack,
            patch.object(fused_gather.gaussian_collective_payload_ops, "_pack_payload_trusted") as layer_pack,
            patch.object(torch.distributed, "all_gather_into_tensor") as payload_gather,
            self.assertRaisesRegex(RuntimeError, "supports at most 128 trainable input tensors, got 132"),
        ):
            fused_gather._FusedGaussianParameterGather.apply(
                layouts,
                fused_gather._PayloadIntegerRangeValidationState(),
                0,
                2,
                cast(torch.distributed.ProcessGroup, object()),
                None,
                *fields,
            )

        gather_counts.assert_called_once()
        grouped_pack.assert_not_called()
        layer_pack.assert_not_called()
        payload_gather.assert_not_called()

    def test_gradient_plan_builders_defensively_enforce_exported_input_capacity(self) -> None:
        capacity = fused_gather.gaussian_collective_payload_ops._MAX_GRADIENT_PACK_INPUTS
        active_fields = tuple(range(capacity + 1))
        counts_by_layer = ((1, 1),)
        field_layers = (0,) * len(active_fields)
        field_widths = (1,) * len(active_fields)
        sizes_by_rank = (len(active_fields), len(active_fields))

        with self.assertRaisesRegex(RuntimeError, rf"supports at most {capacity}.*got {capacity + 1}"):
            fused_gather._gradient_pack_plan_rows(
                counts_by_layer,
                field_layers,
                field_widths,
                active_fields,
                sizes_by_rank,
            )
        with self.assertRaisesRegex(RuntimeError, rf"supports at most {capacity}.*got {capacity + 1}"):
            fused_gather._cached_gradient_pack_plan(
                counts_by_layer,
                field_layers,
                field_widths,
                active_fields,
                torch.device("cpu"),
                0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_gradient_pack_capacity_accepts_128_and_rejects_129(self) -> None:
        gradients = [torch.empty(0, device="cuda") for _ in range(129)]
        output = torch.empty(0, device="cuda")
        copy_rows = torch.empty((0, 5), device="cuda", dtype=torch.int64)

        result = fused_gather.gaussian_collective_payload_ops._pack_gradients_trusted(
            output,
            gradients[:128],
            copy_rows,
            0,
        )
        self.assertEqual(result.data_ptr(), output.data_ptr())
        with self.assertRaisesRegex(RuntimeError, "pack_gradients supports at most 128 input tensors, got 129"):
            fused_gather.gaussian_collective_payload_ops._pack_gradients_trusted(
                output,
                gradients,
                copy_rows,
                0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_releases_grouped_local_payloads_after_gather_and_gathered_payloads_after_unpack(self) -> None:
        device = torch.device("cuda")
        payloads = tuple(torch.ones(index + 1, 2, device=device) for index in range(4))
        counts = tuple((payload.shape[0], payload.shape[0]) for payload in payloads)
        layouts = tuple(
            fused_gather._make_layer_layout(node_id, {"value": payload})
            for node_id, payload in zip(_NODES, payloads, strict=True)
        )
        local_payload_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        gathered_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        single_unpack_calls = 0
        grouped_unpack_calls = 0
        real_unpack = fused_gather.gaussian_collective_payload_ops._unpack_payload_trusted
        real_grouped_unpack = fused_gather.gaussian_collective_payload_ops._unpack_grouped_payload_trusted

        def fake_payload(output: torch.Tensor, padded: torch.Tensor, **_kwargs: Any) -> None:
            if local_payload_refs:
                self.assertIsNone(local_payload_refs[-1]())
            local_payload_refs.append(weakref.ref(padded))
            gathered_refs.append(weakref.ref(output))
            gathered_by_rank = output.reshape(2, -1)
            gathered_by_rank[0].copy_(padded.reshape(-1))
            gathered_by_rank[1].zero_()

        def tracked_unpack(*args):
            nonlocal single_unpack_calls
            self.assertTrue(all(reference() is None for reference in local_payload_refs))
            self.assertEqual(single_unpack_calls, 0)
            result = real_unpack(*args)
            single_unpack_calls += 1
            return result

        def tracked_grouped_unpack(*args):
            nonlocal grouped_unpack_calls
            self.assertTrue(all(reference() is None for reference in local_payload_refs))
            self.assertEqual(single_unpack_calls, 0)
            self.assertEqual(grouped_unpack_calls, 0)
            self.assertIsNotNone(gathered_refs[0]())
            result = real_grouped_unpack(*args)
            grouped_unpack_calls += 1
            return result

        with (
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=counts),
            patch.object(torch.distributed, "all_gather_into_tensor", new=fake_payload),
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_unpack_payload_trusted",
                new=tracked_unpack,
            ),
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_unpack_grouped_payload_trusted",
                new=tracked_grouped_unpack,
            ),
        ):
            fused_gather._FusedGaussianParameterGather.apply(
                layouts,
                fused_gather._PayloadIntegerRangeValidationState(),
                0,
                2,
                cast(torch.distributed.ProcessGroup, object()),
                None,
                *payloads,
            )

        self.assertEqual(single_unpack_calls, 0)
        self.assertEqual(grouped_unpack_calls, 1)
        self.assertEqual(len(local_payload_refs), 1)
        self.assertTrue(all(reference() is None for reference in local_payload_refs))
        self.assertEqual(len(gathered_refs), 1)
        self.assertTrue(all(reference() is None for reference in gathered_refs))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_releases_fallback_local_payloads_after_each_gather(self) -> None:
        device = torch.device("cuda")
        layer_count = fused_gather.gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS + 1
        payloads = tuple(torch.ones(index + 1, 2, device=device) for index in range(layer_count))
        counts = tuple((payload.shape[0], payload.shape[0]) for payload in payloads)
        layouts = tuple(
            fused_gather._make_layer_layout(f"node_{index}", {"value": payload})
            for index, payload in enumerate(payloads)
        )
        local_payload_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        unpack_calls = 0
        real_unpack = fused_gather.gaussian_collective_payload_ops._unpack_payload_trusted

        def fake_payload(output: torch.Tensor, padded: torch.Tensor, **_kwargs: Any) -> None:
            if local_payload_refs:
                self.assertIsNone(local_payload_refs[-1]())
            local_payload_refs.append(weakref.ref(padded))
            gathered_by_rank = output.reshape(2, -1)
            gathered_by_rank[0].copy_(padded.reshape(-1))
            gathered_by_rank[1].zero_()

        def tracked_unpack(*args):
            nonlocal unpack_calls
            self.assertTrue(all(reference() is None for reference in local_payload_refs))
            unpack_calls += 1
            return real_unpack(*args)

        with (
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=counts),
            patch.object(torch.distributed, "all_gather_into_tensor", new=fake_payload),
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_unpack_payload_trusted",
                new=tracked_unpack,
            ),
        ):
            fused_gather._FusedGaussianParameterGather.apply(
                layouts,
                fused_gather._PayloadIntegerRangeValidationState(),
                0,
                2,
                cast(torch.distributed.ProcessGroup, object()),
                None,
                *payloads,
            )

        self.assertEqual(len(local_payload_refs), len(payloads))
        self.assertTrue(all(reference() is None for reference in local_payload_refs))
        self.assertEqual(unpack_calls, len(payloads))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_gathers_inline_count_packet_then_preserves_payload_order(self) -> None:
        device = torch.device("cuda")
        payloads = (
            torch.ones(1, 2, device=device, requires_grad=True),
            torch.ones(2, 3, device=device),
            torch.empty(0, 1, device=device, requires_grad=True),
            torch.ones(3, 2, device=device),
        )
        counts = ((1, 2), (2, 1), (0, 1), (3, 2))
        layouts = tuple(
            fused_gather._make_layer_layout(node_id, {"value": payload})
            for node_id, payload in zip(_NODES, payloads, strict=True)
        )
        expected_schema_hash = fused_gather._communication_schema_hash(
            layouts,
            (True, False, True, False),
        )
        events: list[str] = []
        packed_ranges: list[tuple[str, int, int, int]] = []
        real_pack = fused_gather.gaussian_collective_payload_ops._pack_payload_trusted
        real_grouped_pack = fused_gather.gaussian_collective_payload_ops._pack_grouped_payload_trusted
        real_unpack = fused_gather.gaussian_collective_payload_ops._unpack_payload_trusted
        real_grouped_unpack = fused_gather.gaussian_collective_payload_ops._unpack_grouped_payload_trusted

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            if local.dtype == fused_gather._COUNT_METADATA_DTYPE:
                torch.testing.assert_close(
                    local,
                    torch.tensor(
                        _layer_count_packet_values(
                            (1, 2, 0, 3),
                            expected_schema_hash,
                            integer_ranges_cached=True,
                        ),
                        dtype=fused_gather._COUNT_METADATA_DTYPE,
                        device=device,
                    ),
                    rtol=0,
                    atol=0,
                )
                remote = torch.tensor(
                    _layer_count_packet_values(
                        (2, 1, 1, 2),
                        expected_schema_hash,
                        integer_ranges_cached=True,
                    ),
                    dtype=fused_gather._COUNT_METADATA_DTYPE,
                    device=device,
                )
                output.copy_(torch.stack((local, remote)).reshape(-1))
                events.append("metadata")
                return

            expected_grouped_span = sum(max(counts[layer]) * payloads[layer].shape[1] for layer in range(len(payloads)))
            self.assertEqual(tuple(local.shape), (expected_grouped_span,))
            self.assertEqual(tuple(output.shape), (2 * expected_grouped_span,))
            events.append("payload:grouped_layers")
            output.reshape(2, expected_grouped_span)[0].copy_(local)
            output.reshape(2, expected_grouped_span)[1].zero_()

        def tracked_pack(output: torch.Tensor, *args):
            storage_pointer = int(output.untyped_storage().data_ptr())
            storage_offset = int(output.storage_offset())
            output_numel = int(output.numel())
            packed_ranges.append(
                (
                    "single",
                    storage_pointer,
                    storage_offset,
                    storage_offset + output_numel,
                )
            )
            return real_pack(output, *args)

        def tracked_grouped_pack(output: torch.Tensor, *args):
            storage_pointer = int(output.untyped_storage().data_ptr())
            storage_offset = int(output.storage_offset())
            output_numel = int(output.numel())
            packed_ranges.append(
                (
                    "grouped",
                    storage_pointer,
                    storage_offset,
                    storage_offset + output_numel,
                )
            )
            return real_grouped_pack(output, *args)

        def tracked_unpack(*args):
            events.append("unpack:single")
            return real_unpack(*args)

        def tracked_grouped_unpack(*args):
            events.append("unpack:grouped")
            return real_grouped_unpack(*args)

        with (
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=fake_all_gather,
            ) as all_gather,
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_pack_payload_trusted",
                side_effect=tracked_pack,
            ),
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_pack_grouped_payload_trusted",
                side_effect=tracked_grouped_pack,
            ),
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_unpack_payload_trusted",
                side_effect=tracked_unpack,
            ),
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_unpack_grouped_payload_trusted",
                side_effect=tracked_grouped_unpack,
            ),
        ):
            forward_counts, *_gathered = fused_gather._FusedGaussianParameterGather.apply(
                layouts,
                fused_gather._PayloadIntegerRangeValidationState(),
                0,
                2,
                cast(torch.distributed.ProcessGroup, object()),
                None,
                *payloads,
            )

        self.assertEqual(all_gather.call_count, 2)
        # Unpacking is deferred past every payload gather, so no unpack
        # kernel interleaves into the metadata/payload collective sequence.
        self.assertEqual(
            events,
            [
                "metadata",
                "payload:grouped_layers",
                "unpack:grouped",
            ],
        )
        self.assertEqual([label for label, _storage, _start, _end in packed_ranges], ["grouped"])
        self.assertEqual(
            packed_ranges[0][3] - packed_ranges[0][2],
            sum(max(counts[layer]) * payloads[layer].shape[1] for layer in range(len(payloads))),
        )
        self.assertEqual(forward_counts, counts)
        for layer, (local, layer_counts, gathered) in enumerate(zip(payloads, counts, _gathered, strict=True)):
            with self.subTest(layer=layer):
                expected = torch.cat(
                    (
                        local.detach(),
                        torch.zeros(layer_counts[1], local.shape[1], dtype=local.dtype, device=local.device),
                    )
                )
                torch.testing.assert_close(gathered, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_groups_compatible_payloads_regardless_of_node_order(self) -> None:
        device = torch.device("cuda")
        node_order = ("background", "dynamic_rigids", "road", "dynamic_deformables")
        payloads = tuple(torch.ones(index + 1, 2, device=device) for index in range(len(node_order)))
        counts = tuple((payload.shape[0], payload.shape[0]) for payload in payloads)
        layouts = tuple(
            fused_gather._make_layer_layout(node_id, {"value": payload})
            for node_id, payload in zip(node_order, payloads, strict=True)
        )

        with (
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=counts),
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=_fake_two_rank_payload_gather,
            ) as payload_gather,
            patch.object(
                fused_gather, "_all_gather_grouped_payload", wraps=fused_gather._all_gather_grouped_payload
            ) as grouped,
        ):
            fused_gather._FusedGaussianParameterGather.apply(
                layouts,
                fused_gather._PayloadIntegerRangeValidationState(),
                0,
                2,
                cast(torch.distributed.ProcessGroup, object()),
                None,
                *payloads,
            )

        payload_gather.assert_called_once()
        grouped.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_groups_nonempty_background_with_empty_layers(self) -> None:
        device = torch.device("cuda")
        payloads = (
            torch.ones(1, 2, device=device),
            torch.empty(0, 3, device=device),
            torch.empty(0, 1, device=device),
            torch.empty(0, 2, device=device),
        )
        counts = ((1, 1), (0, 0), (0, 0), (0, 0))
        layouts = tuple(
            fused_gather._make_layer_layout(node_id, {"value": payload})
            for node_id, payload in zip(_NODES, payloads, strict=True)
        )

        with (
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=counts),
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=_fake_two_rank_payload_gather,
            ) as payload_gather,
        ):
            forward_counts, *gathered = fused_gather._FusedGaussianParameterGather.apply(
                layouts,
                fused_gather._PayloadIntegerRangeValidationState(),
                0,
                2,
                cast(torch.distributed.ProcessGroup, object()),
                None,
                *payloads,
            )

        self.assertEqual(payload_gather.call_count, 1)
        self.assertEqual(forward_counts, counts)
        self.assertEqual(tuple(field.numel() for field in gathered), (4, 0, 0, 0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_rejects_gathered_count_that_disagrees_with_local_rows(self) -> None:
        device = torch.device("cuda")
        payloads = (
            torch.ones(1, 1, device=device),
            torch.ones(1, 1, device=device),
        )
        layouts = tuple(
            fused_gather._make_layer_layout(node_id, {"value": payload})
            for node_id, payload in zip(("first", "second"), payloads, strict=True)
        )

        with (
            patch.object(
                fused_gather,
                "_all_gather_layer_counts",
                return_value=((1, 1), (5, 1)),
            ),
            patch.object(torch.distributed, "all_gather_into_tensor") as payload_gather,
            self.assertRaisesRegex(
                RuntimeError,
                "Gathered count mismatch for second: rank 0 count 5 != local count 1",
            ),
        ):
            fused_gather._FusedGaussianParameterGather.apply(
                layouts,
                fused_gather._PayloadIntegerRangeValidationState(),
                0,
                2,
                cast(torch.distributed.ProcessGroup, object()),
                None,
                *payloads,
            )

        payload_gather.assert_not_called()

    def test_forward_rejects_empty_layouts_before_count_collective(self) -> None:
        with (
            patch.object(fused_gather, "_all_gather_layer_counts") as gather_counts,
            self.assertRaisesRegex(RuntimeError, "requires at least one layer"),
        ):
            fused_gather._FusedGaussianParameterGather.apply(
                (),
                fused_gather._PayloadIntegerRangeValidationState(),
                0,
                2,
                cast(torch.distributed.ProcessGroup, object()),
                None,
            )

        gather_counts.assert_not_called()

    def test_payload_gather_writes_one_flat_rank_major_output(self) -> None:
        payload = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        def fake_payload(output: torch.Tensor, local_payload: torch.Tensor, **_kwargs: Any) -> None:
            self.assertTrue(output.is_contiguous())
            self.assertTrue(local_payload.is_contiguous())
            self.assertEqual(tuple(output.shape), (6, 3))
            output.view(3, 2, 3).copy_(torch.stack((local_payload, local_payload + 10, local_payload + 20)))

        with patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_payload) as gather:
            gathered = fused_gather._all_gather_padded_payload(3, payload)

        gather.assert_called_once()
        torch.testing.assert_close(
            gathered,
            torch.cat((payload, payload + 10, payload + 20)),
            rtol=0,
            atol=0,
        )

    def test_payload_gather_skips_globally_empty_payload(self) -> None:
        for payload in (torch.empty(0, 3), torch.empty(2, 0)):
            with patch.object(torch.distributed, "all_gather_into_tensor") as gather:
                gathered = fused_gather._all_gather_padded_payload(2, payload)

            gather.assert_not_called()
            self.assertEqual(tuple(gathered.shape), (2 * payload.shape[0], payload.shape[1]))

        with patch.object(torch.distributed, "all_gather_into_tensor") as gather:
            grouped = fused_gather._all_gather_grouped_payload(2, torch.empty(0))

        gather.assert_not_called()
        self.assertEqual(tuple(grouped.shape), (0,))

    def test_payload_unpack_fallback_metadata_cache_tracks_topology_and_stream(self) -> None:
        fused_gather._cached_payload_unpack_layout.cache_clear()
        args = ((2, 1), (2, 1), 2, 6, 0, torch.device("cuda"))
        try:
            with patch.object(fused_gather.torch, "tensor", return_value=torch.empty((2, 5))) as upload:
                first = fused_gather._cached_payload_unpack_layout(*args, 7)
                same = fused_gather._cached_payload_unpack_layout(*args, 7)
                changed_counts = fused_gather._cached_payload_unpack_layout(
                    (3, 1),
                    args[1],
                    3,
                    9,
                    0,
                    args[5],
                    7,
                )
                changed_stream = fused_gather._cached_payload_unpack_layout(*args, 8)

            self.assertIs(first, same)
            self.assertIsNot(first, changed_counts)
            self.assertIsNot(first, changed_stream)
            self.assertEqual(upload.call_count, 3)
            cache_info = fused_gather._cached_payload_unpack_layout.cache_info()
            self.assertEqual(cache_info.hits, 1)
            self.assertEqual(cache_info.misses, 3)
        finally:
            fused_gather._cached_payload_unpack_layout.cache_clear()

    def test_grouped_payload_unpack_cache_tracks_72_rank_topology_and_stream(self) -> None:
        fused_gather._cached_grouped_payload_unpack_layout.cache_clear()
        counts = tuple(tuple((rank + layer) % 3 for rank in range(72)) for layer in range(4))
        widths = ((2, 1), (3,), (1, 4), (2,))

        def make_grouped_layout(layer_counts: tuple[tuple[int, ...], ...]) -> Any:
            spans = tuple(
                max(count_vector) * sum(layer_widths)
                for count_vector, layer_widths in zip(layer_counts, widths, strict=True)
            )
            return fused_gather._GroupedPayloadLayout(
                layer_offsets=tuple(sum(spans[:layer]) for layer in range(4)),
                layer_spans=spans,
                rank_stride=sum(spans),
            )

        layout = make_grouped_layout(counts)
        permuted_counts = tuple(count_vector[1:] + count_vector[:1] for count_vector in counts)
        self.assertEqual(make_grouped_layout(permuted_counts), layout)
        changed_counts = ((3, *counts[0][1:]), *counts[1:])
        changed_layout = make_grouped_layout(changed_counts)
        device = torch.device("cuda")
        try:
            with patch.object(fused_gather.torch, "tensor", return_value=torch.empty((1, 5))) as upload:
                first = fused_gather._cached_grouped_payload_unpack_layout(counts, widths, layout, device, 7)
                same = fused_gather._cached_grouped_payload_unpack_layout(counts, widths, layout, device, 7)
                permuted = fused_gather._cached_grouped_payload_unpack_layout(
                    permuted_counts,
                    widths,
                    layout,
                    device,
                    7,
                )
                changed = fused_gather._cached_grouped_payload_unpack_layout(
                    changed_counts,
                    widths,
                    changed_layout,
                    device,
                    7,
                )
                changed_stream = fused_gather._cached_grouped_payload_unpack_layout(
                    counts,
                    widths,
                    layout,
                    device,
                    8,
                )

            self.assertIs(first, same)
            self.assertIsNot(first, permuted)
            self.assertIsNot(first, changed)
            self.assertIsNot(first, changed_stream)
            self.assertEqual(upload.call_count, 4)
            cache_info = fused_gather._cached_grouped_payload_unpack_layout.cache_info()
            self.assertEqual(cache_info.hits, 1)
            self.assertEqual(cache_info.misses, 4)
        finally:
            fused_gather._cached_grouped_payload_unpack_layout.cache_clear()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_mixed_trainable_payload_dtypes_fail_after_schema_agreement(self) -> None:
        parameters = {
            "first": {"value": torch.ones(1, 1, device="cuda", dtype=torch.float32, requires_grad=True)},
            "second": {"value": torch.ones(1, 1, device="cuda", dtype=torch.float64, requires_grad=True)},
        }
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(
                fused_gather,
                "_all_gather_layer_counts",
                return_value=((1, 1), (1, 1)),
            ) as gather_counts,
            self.assertRaisesRegex(RuntimeError, "one common trainable payload dtype"),
        ):
            fused_gather._fused_all_gather_gaussian_parameters(
                ["first", "second"],
                parameters,
                process_group=cast(torch.distributed.ProcessGroup, object()),
            )

        gather_counts.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_frozen_integer_only_layer_fails_after_schema_agreement(self) -> None:
        parameters = {
            "first": {"value": torch.ones(1, 1, device="cuda", dtype=torch.float32, requires_grad=True)},
            "second": {"ids": torch.ones(2, 1, device="cuda", dtype=torch.int64)},
        }
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(
                fused_gather,
                "_all_gather_layer_counts",
                return_value=((1, 1), (2, 2)),
            ) as gather_counts,
            self.assertRaisesRegex(RuntimeError, "requires floating-point layer payloads"),
        ):
            fused_gather._fused_all_gather_gaussian_parameters(
                ["first", "second"],
                parameters,
                process_group=cast(torch.distributed.ProcessGroup, object()),
            )

        gather_counts.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_unsupported_frozen_payload_dtype_fails_after_schema_agreement(self) -> None:
        parameters = {
            "first": {"value": torch.ones(1, 1, device="cuda", dtype=torch.float32, requires_grad=True)},
            "second": {"value": torch.empty(2, 1, device="cuda", dtype=torch.float8_e4m3fn)},
        }
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(
                fused_gather,
                "_all_gather_layer_counts",
                return_value=((1, 1), (2, 2)),
            ) as gather_counts,
            self.assertRaisesRegex(RuntimeError, "supports layer payload dtypes"),
        ):
            fused_gather._fused_all_gather_gaussian_parameters(
                ["first", "second"],
                parameters,
                process_group=cast(torch.distributed.ProcessGroup, object()),
            )

        gather_counts.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFusedParameterGatherCuda(unittest.TestCase):
    def test_integer_range_validation_cache_skips_stable_generation_and_revalidates_mutations(self) -> None:
        device = torch.device("cuda")
        values = torch.ones(1, 1, dtype=torch.float32, device=device)
        ids = torch.tensor([[1]], dtype=torch.int64, device=device)
        state = fused_gather._PayloadIntegerRangeValidationState()
        process_group = cast(torch.distributed.ProcessGroup, object())
        validations = 0
        real_validate = fused_gather._validate_payload_integer_ranges

        def validate(*args: Any) -> None:
            nonlocal validations
            validations += 1
            real_validate(*args)

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            if local.dtype == fused_gather._COUNT_METADATA_DTYPE:
                output.copy_(torch.cat((local, local)))
            else:
                _fake_two_rank_payload_gather(output, local)

        def run(current_ids: torch.Tensor) -> None:
            layout = fused_gather._make_layer_layout("node", {"value": values, "id": current_ids})
            fused_gather._FusedGaussianParameterGather.apply(
                (layout,),
                state,
                0,
                2,
                process_group,
                None,
                values,
                current_ids,
            )

        with (
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather),
            patch.object(torch.distributed, "all_reduce") as reduce_flag,
            patch.object(fused_gather, "_validate_payload_integer_ranges", side_effect=validate),
        ):
            run(ids)
            run(ids)
            ids.add_(1)
            run(ids)
            replacement = ids.clone()
            run(replacement)

        self.assertEqual(validations, 3)
        self.assertEqual(reduce_flag.call_count, 3)
        self.assertEqual(state.generation, 3)

    def test_integer_payload_range_covers_exact_limits_out_of_range_and_empty_local_rank(self) -> None:
        device = torch.device("cuda")

        def make_fields(
            ids: tuple[int, ...],
            payload_dtype: torch.dtype,
        ) -> tuple[tuple[Any, ...], tuple[tuple[torch.Tensor, ...], ...]]:
            fields = (
                torch.zeros((len(ids), 1), dtype=payload_dtype, device=device),
                torch.tensor(ids, dtype=torch.int64, device=device).reshape(-1, 1),
            )
            layout = fused_gather._make_layer_layout("node", {"value": fields[0], "id": fields[1]})
            self.assertEqual(layout.payload_dtype, payload_dtype)
            return (layout,), (fields,)

        for payload_dtype, exact_limit in fused_gather._FLOAT_INTEGER_EXACT_LIMIT.items():
            with self.subTest(payload_dtype=payload_dtype, case="exact limits"):
                layouts, fields_by_layer = make_fields((-exact_limit, exact_limit), payload_dtype)
                with patch.object(torch.distributed, "all_reduce") as reduce_flag:
                    fused_gather._validate_payload_integer_ranges(layouts, fields_by_layer)
                reduce_flag.assert_called_once()

            for out_of_range in (-exact_limit - 1, exact_limit + 1):
                with self.subTest(payload_dtype=payload_dtype, out_of_range=out_of_range):
                    layouts, fields_by_layer = make_fields((out_of_range,), payload_dtype)
                    with (
                        patch.object(torch.distributed, "all_reduce") as reduce_flag,
                        self.assertRaisesRegex(RuntimeError, "outside.*contiguous exact-integer range"),
                    ):
                        fused_gather._validate_payload_integer_ranges(layouts, fields_by_layer)
                    reduce_flag.assert_called_once()

        layouts, fields_by_layer = make_fields((), torch.float32)

        def inject_remote_violation(flag: torch.Tensor, **_kwargs: Any) -> None:
            flag.fill_(1)

        with (
            patch.object(torch.distributed, "all_reduce", side_effect=inject_remote_violation) as reduce_flag,
            self.assertRaisesRegex(RuntimeError, "outside.*contiguous exact-integer range"),
        ):
            fused_gather._validate_payload_integer_ranges(layouts, fields_by_layer)
        reduce_flag.assert_called_once()

        for safe_ids, payload_dtype, id_dtype in (
            ((0, 255), torch.float16, torch.uint8),
            ((-128, 127), torch.bfloat16, torch.int8),
            ((-32768, 32767), torch.float32, torch.int16),
            ((-(1 << 31), (1 << 31) - 1), torch.float64, torch.int32),
        ):
            with self.subTest(payload_dtype=payload_dtype, id_dtype=id_dtype, case="full dtype domain"):
                fields = (
                    torch.zeros((len(safe_ids), 1), dtype=payload_dtype, device=device),
                    torch.tensor(safe_ids, dtype=id_dtype, device=device).reshape(-1, 1),
                )
                layout = fused_gather._make_layer_layout("node", {"value": fields[0], "id": fields[1]})
                with patch.object(torch.distributed, "all_reduce") as reduce_flag:
                    fused_gather._validate_payload_integer_ranges((layout,), (fields,))
                reduce_flag.assert_not_called()

    def test_integer_payload_range_validation_precedes_grouped_and_per_layer_packing(self) -> None:
        device = torch.device("cuda")
        real_validate = fused_gather._validate_payload_integer_ranges
        real_grouped_pack = fused_gather.gaussian_collective_payload_ops._pack_grouped_payload_trusted
        real_layer_pack = fused_gather.gaussian_collective_payload_ops._pack_payload_trusted

        capacity = fused_gather.gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS
        for layer_count, expected_route in ((2, "grouped"), (5, "grouped"), (capacity + 1, "per-layer")):
            with self.subTest(route=expected_route):
                fields_by_layer = tuple(
                    (
                        torch.ones(1, 1, dtype=torch.float32, device=device),
                        torch.tensor([[layer]], dtype=torch.int64, device=device),
                    )
                    for layer in range(layer_count)
                )
                layouts = tuple(
                    fused_gather._make_layer_layout(
                        f"node_{layer}",
                        {"value": fields[0], "id": fields[1]},
                    )
                    for layer, fields in enumerate(fields_by_layer)
                )
                events: list[str] = []

                def validate(*args: Any) -> None:
                    events.append("validate")
                    real_validate(*args)

                def grouped_pack(*args: Any) -> None:
                    self.assertEqual(events, ["validate"])
                    events.append("pack:grouped")
                    real_grouped_pack(*args)

                def layer_pack(*args: Any) -> None:
                    self.assertEqual(events[0], "validate")
                    events.append("pack:layer")
                    real_layer_pack(*args)

                with (
                    patch.object(
                        fused_gather,
                        "_all_gather_layer_counts",
                        return_value=tuple((1, 1) for _ in range(layer_count)),
                    ),
                    patch.object(torch.distributed, "all_reduce"),
                    patch.object(
                        torch.distributed,
                        "all_gather_into_tensor",
                        side_effect=_fake_two_rank_payload_gather,
                    ),
                    patch.object(fused_gather, "_validate_payload_integer_ranges", side_effect=validate),
                    patch.object(
                        fused_gather.gaussian_collective_payload_ops,
                        "_pack_grouped_payload_trusted",
                        side_effect=grouped_pack,
                    ) as grouped,
                    patch.object(
                        fused_gather.gaussian_collective_payload_ops,
                        "_pack_payload_trusted",
                        side_effect=layer_pack,
                    ) as per_layer,
                ):
                    fused_gather._FusedGaussianParameterGather.apply(
                        layouts,
                        fused_gather._PayloadIntegerRangeValidationState(),
                        0,
                        2,
                        cast(torch.distributed.ProcessGroup, object()),
                        None,
                        *(field for fields in fields_by_layer for field in fields),
                    )

                if expected_route == "grouped":
                    grouped.assert_called_once()
                    per_layer.assert_not_called()
                    self.assertEqual(events, ["validate", "pack:grouped"])
                else:
                    grouped.assert_not_called()
                    self.assertEqual(per_layer.call_count, layer_count)
                    self.assertEqual(events, ["validate", *("pack:layer" for _ in range(layer_count))])

    def test_cached_integer_payload_is_structural_generation_state(self) -> None:
        device = torch.device("cuda")
        process_group = cast(torch.distributed.ProcessGroup, object())
        parameters = {
            node_id: {
                "value": torch.ones(1, 1, dtype=torch.float32, device=device, requires_grad=True),
                "id": torch.tensor([[index]], dtype=torch.int64, device=device),
            }
            for index, node_id in enumerate(("first", "second"))
        }

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
        ):
            prepared = fused_gather._prepare_fused_gather_inputs(
                list(parameters),
                parameters,
                process_group=process_group,
            )
            resolved = fused_gather._ResolvedLayerCountGather(
                counts_by_layer=((1, 1), (1, 1)),
                local_counts=prepared.local_counts,
                schema_hash=prepared.schema_hash,
                device=prepared.device,
                world_rank=prepared.world_rank,
                world_size=prepared.world_size,
            )
            with patch.object(torch.distributed, "all_reduce") as range_consensus:
                plan = fused_gather._make_fused_gather_generation_plan(
                    prepared,
                    resolved,
                    process_group=process_group,
                )
            range_consensus.assert_called_once()
            self.assertEqual(
                tuple((state.node_id, state.key) for state in plan.integer_payload_state),
                (("first", "id"), ("second", "id")),
            )

            steady_work = Mock()
            with (
                patch.object(torch.distributed, "all_reduce") as steady_consensus,
                patch.object(
                    fused_gather,
                    "_validate_integer_payload_generation_state",
                    wraps=fused_gather._validate_integer_payload_generation_state,
                ) as validate_generation,
                patch.object(
                    torch.distributed,
                    "all_gather_into_tensor",
                    return_value=steady_work,
                ) as payload_gather,
            ):
                steady_inputs = fused_gather._prepare_fused_gather_inputs_from_generation_plan(
                    list(parameters),
                    parameters,
                    plan,
                    process_group=process_group,
                )
                steady_pending = fused_gather._start_fused_payload_gather_from_generation_plan(
                    steady_inputs,
                    plan,
                    process_group=process_group,
                )
            steady_consensus.assert_not_called()
            validate_generation.assert_called_once_with(plan, steady_inputs.fields_by_layer)
            payload_gather.assert_called_once()
            fused_gather._discard_fused_payload_gather(steady_pending)
            steady_work.wait.assert_called_once_with()

            parameters["first"]["id"].add_(1)
            mutated_inputs = fused_gather._prepare_fused_gather_inputs_from_generation_plan(
                list(parameters),
                parameters,
                plan,
                process_group=process_group,
            )
            with (
                patch.object(torch.distributed, "all_reduce") as mutation_consensus,
                patch.object(torch.distributed, "all_gather_into_tensor") as mutation_payload_gather,
                self.assertRaisesRegex(RuntimeError, "integer payload 'first'.id changed outside a structural refresh"),
            ):
                fused_gather._start_fused_payload_gather_from_generation_plan(
                    mutated_inputs,
                    plan,
                    process_group=process_group,
                )
            mutation_consensus.assert_not_called()
            mutation_payload_gather.assert_not_called()

            refreshed_inputs = fused_gather._prepare_fused_gather_inputs(
                list(parameters),
                parameters,
                process_group=process_group,
            )
            refreshed_counts = replace(
                resolved,
                local_counts=refreshed_inputs.local_counts,
                schema_hash=refreshed_inputs.schema_hash,
            )
            with patch.object(torch.distributed, "all_reduce") as refresh_consensus:
                refreshed_plan = fused_gather._make_fused_gather_generation_plan(
                    refreshed_inputs,
                    refreshed_counts,
                    process_group=process_group,
                )
            refresh_consensus.assert_called_once()
            fused_gather._prepare_fused_gather_inputs_from_generation_plan(
                list(parameters),
                parameters,
                refreshed_plan,
                process_group=process_group,
            )

            parameters["second"]["id"] = parameters["second"]["id"].clone()
            replaced_inputs = fused_gather._prepare_fused_gather_inputs_from_generation_plan(
                list(parameters),
                parameters,
                refreshed_plan,
                process_group=process_group,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "integer payload 'second'.id changed outside a structural refresh",
            ):
                fused_gather._start_fused_payload_gather_from_generation_plan(
                    replaced_inputs,
                    refreshed_plan,
                    process_group=process_group,
                )

    def test_async_launch_wrapper_failures_drain_work_and_preserve_primary_errors(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        process_group = cast(torch.distributed.ProcessGroup, object())
        primary = RuntimeError("injected wrapper failure")
        drain = RuntimeError("injected drain failure")

        count_work = Mock()
        count_work.wait.side_effect = drain
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "all_gather_into_tensor", return_value=count_work),
            patch.object(fused_gather, "_PendingLayerCountGather", side_effect=primary),
            self.assertRaises(RuntimeError) as raised,
        ):
            fused_gather._start_layer_count_gather(
                2,
                (1,),
                device,
                tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT)),
            )
        self.assertIs(raised.exception, primary)
        count_work.wait.assert_called_once_with()
        self.assertTrue(any("count work" in note and "drain failure" in note for note in primary.__notes__))

        parameters = _make_grouped_nccl_parameters(0, device)
        expected_counts = tuple(_grouped_nccl_counts(0).values())
        payload_work = Mock()
        payload_work.wait.side_effect = drain
        payload_primary = RuntimeError("injected payload wrapper failure")

        def launch_payload(
            _output: torch.Tensor,
            _local: torch.Tensor,
            *,
            group: Optional[torch.distributed.ProcessGroup] = None,
            async_op: bool,
        ) -> Mock:
            self.assertIs(group, process_group)
            self.assertTrue(async_op)
            return payload_work

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=expected_counts),
        ):
            prepared = fused_gather._prepare_fused_gather_inputs(
                list(parameters),
                parameters,
                process_group=process_group,
            )
            plan = fused_gather._resolve_fused_gather_generation_plan(
                prepared,
                process_group=process_group,
            )
            with (
                patch.object(torch.distributed, "all_gather_into_tensor", side_effect=launch_payload),
                patch.object(fused_gather, "_PendingFusedPayloadGather", side_effect=payload_primary),
                self.assertRaises(RuntimeError) as raised,
            ):
                fused_gather._start_fused_payload_gather_from_generation_plan(
                    prepared,
                    plan,
                    process_group=process_group,
                )
        self.assertIs(raised.exception, payload_primary)
        payload_work.wait.assert_called_once_with()
        self.assertTrue(any("payload work" in note and "drain failure" in note for note in payload_primary.__notes__))

    def test_grouped_unpack_uses_one_metadata_kernel_for_every_rank_count(self) -> None:
        device = torch.device("cuda")
        layouts = tuple(
            fused_gather._make_layer_layout(node_id, {"value": torch.ones(1, layer + 1, device=device)})
            for layer, node_id in enumerate(_NODES)
        )

        def dynamic_unpack(outputs, *_args):
            return tuple(outputs)

        for world_size in (2, 72):
            with self.subTest(world_size=world_size):
                counts_by_layer = tuple(tuple((rank + layer) % 3 for rank in range(world_size)) for layer in range(4))
                grouped_layout = fused_gather._make_grouped_payload_layout(layouts, counts_by_layer)
                gathered = torch.zeros(world_size * grouped_layout.rank_stride, device=device)

                fused_gather._cached_grouped_payload_unpack_layout.cache_clear()
                try:
                    with patch.object(
                        fused_gather.gaussian_collective_payload_ops,
                        "_unpack_grouped_payload_trusted",
                        side_effect=dynamic_unpack,
                    ) as metadata_unpack:
                        result = fused_gather._unpack_grouped_layer_payloads(
                            layouts,
                            gathered,
                            counts_by_layer,
                            grouped_layout,
                        )

                    metadata_unpack.assert_called_once()
                    self.assertEqual(len(result), 4)
                    for layer, fields in enumerate(result):
                        self.assertEqual(fields["value"].shape, (sum(counts_by_layer[layer]), layer + 1))
                finally:
                    fused_gather._cached_grouped_payload_unpack_layout.cache_clear()

    def test_grouped_unpack_rejects_mispaired_72_rank_gathered_storage(self) -> None:
        device = torch.device("cuda")
        layouts = tuple(
            fused_gather._make_layer_layout(node_id, {"value": torch.ones(1, layer + 1, device=device)})
            for layer, node_id in enumerate(_NODES)
        )
        counts_by_layer = tuple(tuple((rank + layer) % 3 for rank in range(72)) for layer in range(4))
        grouped_layout = fused_gather._make_grouped_payload_layout(layouts, counts_by_layer)
        gathered = torch.zeros(72 * grouped_layout.rank_stride, device=device)

        for malformed in (gathered[:-1], torch.cat((gathered, gathered.new_zeros(1)))):
            with (
                self.subTest(numel=malformed.numel()),
                patch.object(
                    fused_gather.gaussian_collective_payload_ops,
                    "_unpack_grouped_payload_trusted",
                ) as scalable_unpack,
                self.assertRaisesRegex(
                    ValueError,
                    rf"has {malformed.numel()} values, expected {gathered.numel()}",
                ),
            ):
                fused_gather._unpack_grouped_layer_payloads(
                    layouts,
                    malformed,
                    counts_by_layer,
                    grouped_layout,
                )
            scalable_unpack.assert_not_called()

    def test_generation_plan_rejects_dynamic_mismatches_before_payload_launch(self) -> None:
        device = torch.device("cuda")
        process_group = cast(torch.distributed.ProcessGroup, object())
        parameters = _make_grouped_nccl_parameters(0, device)
        expected_counts = tuple(_grouped_nccl_counts(0).values())

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=expected_counts),
        ):
            prepared = fused_gather._prepare_fused_gather_inputs(
                list(parameters),
                parameters,
                process_group=process_group,
            )
            plan = fused_gather._resolve_fused_gather_generation_plan(
                prepared,
                process_group=process_group,
            )

            count_mismatch = {node_id: dict(fields) for node_id, fields in parameters.items()}
            background = parameters["background"]["value"]
            count_mismatch["background"]["value"] = background.new_zeros(
                (background.shape[0] + 1, *background.shape[1:]),
                requires_grad=background.requires_grad,
            )

            trailing_shape_mismatch = {node_id: dict(fields) for node_id, fields in parameters.items()}
            trailing_shape_mismatch["background"]["value"] = background.new_zeros(
                (background.shape[0], background.shape[1] + 1),
                requires_grad=background.requires_grad,
            )

            dtype_mismatch = {node_id: dict(fields) for node_id, fields in parameters.items()}
            dtype_mismatch["background"]["value"] = background.double()

            device_mismatch = {node_id: dict(fields) for node_id, fields in parameters.items()}
            device_mismatch["background"]["value"] = background.cpu()

            requires_grad_mismatch = {node_id: dict(fields) for node_id, fields in parameters.items()}
            requires_grad_mismatch["background"]["value"] = background.detach()

            field_order_mismatch = {node_id: dict(fields) for node_id, fields in parameters.items()}
            field_order_mismatch["road"] = {
                "float": parameters["road"]["float"],
                "half": parameters["road"]["half"],
            }

            cases = (
                ("count", count_mismatch, "row counts changed outside the structural boundary"),
                ("trailing shape", trailing_shape_mismatch, "field shape changed"),
                ("dtype", dtype_mismatch, "field dtype changed"),
                ("device", device_mismatch, "field device changed"),
                ("requires_grad", requires_grad_mismatch, "requires_grad schema changed"),
                ("field order", field_order_mismatch, "field order changed"),
            )
            for name, changed_parameters, error_pattern in cases:
                with (
                    self.subTest(name=name),
                    patch.object(fused_gather, "_start_fused_payload_gather_from_generation_plan") as start,
                    patch.object(torch.distributed, "all_gather_into_tensor") as payload_gather,
                    self.assertRaisesRegex(RuntimeError, error_pattern),
                ):
                    fresh = fused_gather._prepare_fused_gather_inputs_from_generation_plan(
                        list(changed_parameters),
                        changed_parameters,
                        plan,
                        process_group=process_group,
                    )
                    fused_gather._start_fused_payload_gather_from_generation_plan(
                        fresh,
                        plan,
                        process_group=process_group,
                    )

                start.assert_not_called()
                payload_gather.assert_not_called()

    def test_single_gpu_mocked_two_rank_empty_generation_uses_no_payload_collective(self) -> None:
        device = torch.device("cuda")
        process_group = cast(torch.distributed.ProcessGroup, object())
        field = torch.empty((0, 2), dtype=torch.float32, device=device, requires_grad=True)
        parameters = {"empty": {"value": field}}

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=((0, 0),)) as gather_counts,
            patch.object(torch.distributed, "all_gather_into_tensor") as payload_gather,
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_pack_gradients_trusted",
            ) as pack_gradients,
            patch.object(torch.distributed, "reduce_scatter_tensor") as reduce_scatter,
        ):
            prepared = fused_gather._prepare_fused_gather_inputs(
                ["empty"],
                parameters,
                process_group=process_group,
            )
            plan = fused_gather._resolve_fused_gather_generation_plan(
                prepared,
                process_group=process_group,
            )
            pending = fused_gather._start_fused_payload_gather_from_generation_plan(
                prepared,
                plan,
                process_group=process_group,
            )
            result = fused_gather._fused_all_gather_gaussian_parameters(
                ["empty"],
                parameters,
                process_group=process_group,
                prepared_inputs=prepared,
                generation_plan=plan,
                prefetched_payload_gather=pending,
            )
            result.parameters_by_node["empty"]["value"].sum().backward()

        gather_counts.assert_called_once()
        payload_gather.assert_not_called()
        pack_gradients.assert_not_called()
        reduce_scatter.assert_not_called()
        self.assertEqual(result.counts_by_node, {"empty": (0, 0)})
        self.assertEqual(tuple(result.parameters_by_node["empty"]["value"].shape), (0, 2))
        self.assertEqual(pending.state, "finished")
        self.assertIsNone(pending.work)
        self.assertIsNotNone(field.grad)
        assert field.grad is not None
        self.assertEqual(tuple(field.grad.shape), (0, 2))

    def test_grouped_payload_launches_before_independent_work_and_waits_at_resolver(self) -> None:
        device = torch.device("cuda")
        process_group = cast(torch.distributed.ProcessGroup, object())
        parameters = _make_grouped_nccl_parameters(0, device)
        expected_counts = (
            (1, 2),
            (2, 1),
            (0, 1),
            (1, 1),
        )
        events: list[str] = []
        work = Mock()
        work.wait.side_effect = lambda: events.append("wait")
        direct_range_state = Mock(spec=fused_gather._PayloadIntegerRangeValidationState)
        real_pack = fused_gather.gaussian_collective_payload_ops._pack_grouped_payload_trusted
        real_unpack = fused_gather.gaussian_collective_payload_ops._unpack_grouped_payload_trusted

        def tracked_pack(*args: Any, **kwargs: Any) -> Any:
            events.append("pack")
            return real_pack(*args, **kwargs)

        def async_payload(
            output: torch.Tensor,
            local: torch.Tensor,
            *,
            group: Optional[torch.distributed.ProcessGroup] = None,
            async_op: bool = False,
        ) -> Mock:
            self.assertIs(group, process_group)
            self.assertTrue(async_op)
            events.append("launch")
            _fake_two_rank_payload_gather(output, local)
            return work

        def tracked_unpack(*args: Any, **kwargs: Any) -> Any:
            events.append("unpack")
            return real_unpack(*args, **kwargs)

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=expected_counts) as gather_counts,
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=async_payload) as payload_gather,
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_pack_grouped_payload_trusted",
                side_effect=tracked_pack,
            ),
            patch.object(
                fused_gather.gaussian_collective_payload_ops,
                "_unpack_grouped_payload_trusted",
                side_effect=tracked_unpack,
            ),
        ):
            prepared = fused_gather._prepare_fused_gather_inputs(
                list(parameters),
                parameters,
                process_group=process_group,
            )
            plan = fused_gather._resolve_fused_gather_generation_plan(
                prepared,
                process_group=process_group,
            )
            pending = fused_gather._start_fused_payload_gather_from_generation_plan(
                prepared,
                plan,
                process_group=process_group,
            )
            self.assertEqual(events, ["pack", "launch"])
            work.wait.assert_not_called()

            # This models FreePose CUDA work enqueued after the asynchronous
            # payload launch and before the collector needs gathered fields.
            events.append("freepose")
            result = fused_gather._fused_all_gather_gaussian_parameters(
                list(parameters),
                parameters,
                process_group=process_group,
                integer_range_validation_state=direct_range_state,
                prepared_inputs=prepared,
                generation_plan=plan,
                prefetched_payload_gather=pending,
            )

        self.assertEqual(events, ["pack", "launch", "freepose", "wait", "unpack"])
        self.assertEqual(pending.state, "finished")
        self.assertIsNone(pending.local_grouped_payload)
        self.assertIsNone(pending.grouped_gathered_payload)
        self.assertIsNone(pending.work)
        gather_counts.assert_called_once()
        payload_gather.assert_called_once()
        direct_range_state.observe.assert_not_called()
        self.assertEqual(tuple(result.counts_by_node.values()), expected_counts)
        fused_gather._discard_fused_payload_gather(pending)
        work.wait.assert_called_once_with()

    def test_payload_prefetch_rejects_unsupported_schema_before_metadata_resolution(self) -> None:
        device = torch.device("cuda")
        process_group = cast(torch.distributed.ProcessGroup, object())
        parameters = _make_grouped_nccl_parameters(0, device)
        parameters["background"]["value"] = parameters["background"]["value"].detach().half()

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(fused_gather, "_all_gather_layer_counts") as gather_counts,
            patch.object(torch.distributed, "all_gather_into_tensor") as payload_gather,
        ):
            prepared = fused_gather._prepare_fused_gather_inputs(
                list(parameters),
                parameters,
                process_group=process_group,
            )
            self.assertFalse(fused_gather._can_prefetch_grouped_payload(prepared))
            with self.assertRaisesRegex(RuntimeError, "not supported by the captured schema"):
                fused_gather._resolve_fused_gather_generation_plan(
                    prepared,
                    process_group=process_group,
                )

        gather_counts.assert_not_called()
        payload_gather.assert_not_called()

    def test_prefetched_payload_drains_before_rejecting_supplied_plan_mismatch(self) -> None:
        device = torch.device("cuda")
        process_group = cast(torch.distributed.ProcessGroup, object())
        parameters = _make_grouped_nccl_parameters(0, device)
        expected_counts = tuple(_grouped_nccl_counts(0).values())
        work = Mock()

        def async_payload(
            output: torch.Tensor,
            local: torch.Tensor,
            *,
            group: Optional[torch.distributed.ProcessGroup] = None,
            async_op: bool = False,
        ) -> Mock:
            self.assertIs(group, process_group)
            self.assertTrue(async_op)
            _fake_two_rank_payload_gather(output, local)
            return work

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=expected_counts),
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=async_payload),
        ):
            prepared = fused_gather._prepare_fused_gather_inputs(
                list(parameters),
                parameters,
                process_group=process_group,
            )
            plan = fused_gather._resolve_fused_gather_generation_plan(
                prepared,
                process_group=process_group,
            )
            pending = fused_gather._start_fused_payload_gather_from_generation_plan(
                prepared,
                plan,
                process_group=process_group,
            )
            mismatched = replace(
                prepared,
                local_counts=(prepared.local_counts[0] + 1, *prepared.local_counts[1:]),
            )
            with self.assertRaisesRegex(RuntimeError, "do not match the published generation plan"):
                fused_gather._fused_all_gather_gaussian_parameters(
                    list(parameters),
                    parameters,
                    process_group=process_group,
                    prepared_inputs=mismatched,
                    generation_plan=plan,
                    prefetched_payload_gather=pending,
                )

        work.wait.assert_called_once_with()
        self.assertEqual(pending.state, "discarded")

    def test_prefetched_payload_contract_errors_always_drain_owned_work(self) -> None:
        device = torch.device("cuda")
        process_group = cast(torch.distributed.ProcessGroup, object())
        parameters = _make_grouped_nccl_parameters(0, device)
        expected_counts = tuple(_grouped_nccl_counts(0).values())

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
            patch.object(fused_gather, "_all_gather_layer_counts", return_value=expected_counts),
        ):
            prepared = fused_gather._prepare_fused_gather_inputs(
                list(parameters),
                parameters,
                process_group=process_group,
            )
            plan = fused_gather._resolve_fused_gather_generation_plan(
                prepared,
                process_group=process_group,
            )
            mismatched = replace(prepared, local_counts=(prepared.local_counts[0] + 1, *prepared.local_counts[1:]))
            cases = (
                ("payload-without-plan", list(parameters), prepared, None, "supplied together"),
                ("plan-without-fresh-inputs", list(parameters), None, plan, "explicitly captured fresh inputs"),
                ("node-order", list(reversed(parameters)), prepared, plan, "node order"),
                ("generation-mismatch", list(parameters), mismatched, plan, "published generation plan"),
            )
            for name, nodes, supplied_prepared, supplied_plan, message in cases:
                with self.subTest(name=name):
                    work = Mock()
                    pending = fused_gather._PendingFusedPayloadGather(
                        local_grouped_payload=torch.empty(0, device=device),
                        grouped_gathered_payload=torch.empty(0, device=device),
                        work=work,
                    )
                    with self.assertRaisesRegex(RuntimeError, message):
                        fused_gather._fused_all_gather_gaussian_parameters(
                            nodes,
                            parameters,
                            process_group=process_group,
                            prepared_inputs=supplied_prepared,
                            generation_plan=supplied_plan,
                            prefetched_payload_gather=pending,
                        )
                    work.wait.assert_called_once_with()
                    self.assertEqual(pending.state, "discarded")

            with self.assertRaisesRegex(RuntimeError, "supplied together"):
                fused_gather._fused_all_gather_gaussian_parameters(
                    list(parameters),
                    parameters,
                    process_group=process_group,
                    prepared_inputs=prepared,
                    generation_plan=plan,
                )

    def test_async_count_gather_waits_once_and_resolves_one_shot(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        local_counts = (2, 3)
        remote_counts = (5, 4)
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))
        captured: dict[str, torch.Tensor] = {}
        work = Mock()

        def wait() -> bool:
            remote = captured["local"].clone()
            inline_start = fused_gather._LAYER_COUNT_PACKET_INLINE_COUNTS_START
            remote[inline_start : inline_start + len(remote_counts)].copy_(
                torch.tensor(remote_counts, dtype=torch.int64, device=device)
            )
            captured["output"].copy_(torch.stack((captured["local"], remote)).reshape(-1))
            return True

        work.wait.side_effect = wait

        def fake_all_gather(
            output: torch.Tensor,
            local: torch.Tensor,
            *,
            group: Optional[torch.distributed.ProcessGroup] = None,
            async_op: bool,
        ) -> Mock:
            self.assertIsNone(group)
            self.assertTrue(async_op)
            captured["local"] = local
            captured["output"] = output
            return work

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather) as gather,
        ):
            pending = fused_gather._start_layer_count_gather(
                2,
                local_counts,
                device,
                schema_hash,
            )
            work.wait.assert_not_called()
            self.assertIs(pending.local_packet, captured["local"])
            self.assertIs(pending.gathered_packet, captured["output"])
            self.assertIs(pending.work, work)
            resolved = fused_gather._finish_layer_count_gather(pending)

        gather.assert_called_once()
        work.wait.assert_called_once_with()
        self.assertEqual(resolved.counts_by_layer, ((2, 5), (3, 4)))
        self.assertEqual(pending.state, "finished")
        with self.assertRaisesRegex(RuntimeError, "already finished"):
            fused_gather._finish_layer_count_gather(pending)
        fused_gather._discard_layer_count_gather(pending)
        work.wait.assert_called_once_with()

    def test_async_count_gather_discard_is_idempotent_and_never_starts_fallback(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        work = Mock()

        def fake_all_gather(
            _output: torch.Tensor,
            _local: torch.Tensor,
            *,
            group: Optional[torch.distributed.ProcessGroup] = None,
            async_op: bool,
        ) -> Mock:
            self.assertIsNone(group)
            self.assertTrue(async_op)
            return work

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather) as gather,
        ):
            pending = fused_gather._start_layer_count_gather(
                2,
                tuple(
                    range(
                        1,
                        fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY + 2,
                    )
                ),
                device,
                tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT)),
            )
            fused_gather._discard_layer_count_gather(pending)
            fused_gather._discard_layer_count_gather(pending)

        gather.assert_called_once()
        work.wait.assert_called_once_with()
        self.assertEqual(pending.state, "discarded")
        with self.assertRaisesRegex(RuntimeError, "already discarded"):
            fused_gather._finish_layer_count_gather(pending)

    def test_layer_count_gather_rejects_rank_asymmetric_schema(self) -> None:
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))

        def fake_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            remote = local.clone()
            remote[fused_gather._LAYER_COUNT_PACKET_SCHEMA_HASH_START] += 1
            output.copy_(torch.stack((local, remote)).reshape(-1))

        with (
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_gather) as gather,
            self.assertRaisesRegex(RuntimeError, "schema differs across ranks"),
        ):
            fused_gather._all_gather_layer_counts(
                2,
                (2,),
                torch.device("cuda"),
                schema_hash,
            )

        gather.assert_called_once()

    def test_layer_count_gather_rejects_rank_asymmetric_arity_before_counts(self) -> None:
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))

        def fake_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            remote = local.clone()
            remote[fused_gather._LAYER_COUNT_PACKET_LAYER_COUNT_INDEX] += 1
            output.copy_(torch.stack((local, remote)).reshape(-1))

        with (
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_gather) as gather,
            self.assertRaisesRegex(RuntimeError, "layer count differs across ranks"),
        ):
            fused_gather._all_gather_layer_counts(
                2,
                (2,),
                torch.device("cuda"),
                schema_hash,
            )

        gather.assert_called_once()

    def test_layer_counts_use_one_fixed_control_packet_at_capacity(self) -> None:
        device = torch.device("cuda")
        local_counts = (2, 0, 3, 1, 7, 0, 2, 9)
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))
        remote_counts = (5, 4, 1, 6, 8, 3, 0, 4)
        self.assertEqual(len(local_counts), fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY)

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            self.assertEqual(local.dtype, fused_gather._COUNT_METADATA_DTYPE)
            torch.testing.assert_close(
                local,
                torch.tensor(
                    _layer_count_packet_values(local_counts, schema_hash),
                    dtype=fused_gather._COUNT_METADATA_DTYPE,
                    device=device,
                ),
                rtol=0,
                atol=0,
            )
            remote = torch.tensor(
                _layer_count_packet_values(remote_counts, schema_hash),
                dtype=fused_gather._COUNT_METADATA_DTYPE,
                device=device,
            )
            output.copy_(torch.stack((local, remote)).reshape(-1))

        with patch.object(
            torch.distributed,
            "all_gather_into_tensor",
            side_effect=fake_all_gather,
        ) as all_gather:
            counts_by_layer = fused_gather._all_gather_layer_counts(2, local_counts, device, schema_hash)

        all_gather.assert_called_once()
        self.assertEqual(counts_by_layer, tuple(zip(local_counts, remote_counts, strict=True)))

    def test_layer_counts_ignore_unused_control_packet_slots(self) -> None:
        device = torch.device("cuda")
        local_counts = (2, 3)
        remote_counts = (5, 4)
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            torch.testing.assert_close(
                local,
                torch.tensor(
                    _layer_count_packet_values(local_counts, schema_hash),
                    dtype=fused_gather._COUNT_METADATA_DTYPE,
                    device=device,
                ),
                rtol=0,
                atol=0,
            )
            remote = torch.tensor(
                _layer_count_packet_values(
                    remote_counts,
                    schema_hash,
                    unused_inline_values=(-1, -2),
                ),
                dtype=fused_gather._COUNT_METADATA_DTYPE,
                device=device,
            )
            output.copy_(torch.stack((local, remote)).reshape(-1))

        with patch.object(
            torch.distributed,
            "all_gather_into_tensor",
            side_effect=fake_all_gather,
        ) as all_gather:
            counts_by_layer = fused_gather._all_gather_layer_counts(2, local_counts, device, schema_hash)

        all_gather.assert_called_once()
        self.assertEqual(counts_by_layer, ((2, 5), (3, 4)))

    def test_layer_counts_track_dynamic_values_across_supported_world_sizes(self) -> None:
        device = torch.device("cuda")
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))

        for world_size in (2, 3, 4, 5, 8, 9, 16, 72):
            for layer_count in (2, 3, 4, 5, 8, 9):
                for generation in (0, 1):
                    counts_by_rank = tuple(
                        tuple((generation * 3 + rank * 2 + layer) % 7 for layer in range(layer_count))
                        for rank in range(world_size)
                    )
                    local_counts = counts_by_rank[0]
                    gather_calls = 0

                    def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
                        nonlocal gather_calls
                        if gather_calls == 0:
                            packets = []
                            for counts in counts_by_rank:
                                packets.append(
                                    torch.tensor(
                                        _layer_count_packet_values(counts, schema_hash),
                                        dtype=fused_gather._COUNT_METADATA_DTYPE,
                                        device=device,
                                    )
                                )
                            torch.testing.assert_close(local, packets[0], rtol=0, atol=0)
                            output.copy_(torch.stack(packets).reshape(-1))
                        else:
                            full_counts = torch.tensor(
                                tuple(count for counts in counts_by_rank for count in counts),
                                dtype=fused_gather._COUNT_METADATA_DTYPE,
                                device=device,
                            )
                            torch.testing.assert_close(
                                local,
                                full_counts[:layer_count],
                                rtol=0,
                                atol=0,
                            )
                            output.copy_(full_counts)
                        gather_calls += 1

                    with patch.object(
                        torch.distributed,
                        "all_gather_into_tensor",
                        side_effect=fake_all_gather,
                    ):
                        result = fused_gather._all_gather_layer_counts(
                            world_size,
                            local_counts,
                            device,
                            schema_hash,
                        )

                    expected = tuple(
                        tuple(counts_by_rank[rank][layer] for rank in range(world_size)) for layer in range(layer_count)
                    )
                    self.assertEqual(result, expected)
                    self.assertEqual(
                        gather_calls,
                        1 if layer_count <= fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY else 2,
                    )

    def test_layer_counts_preserve_values_above_int32_range(self) -> None:
        device = torch.device("cuda")
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))
        local_counts = (torch.iinfo(torch.int32).max + 1, 0)
        remote_counts = (0, torch.iinfo(torch.int32).max + 2)

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            self.assertEqual(local.dtype, torch.int64)
            remote = torch.tensor(
                _layer_count_packet_values(remote_counts, schema_hash),
                dtype=fused_gather._COUNT_METADATA_DTYPE,
                device=device,
            )
            output.copy_(torch.stack((local, remote)).reshape(-1))

        with patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather) as all_gather:
            result = fused_gather._all_gather_layer_counts(2, local_counts, device, schema_hash)

        all_gather.assert_called_once()
        self.assertEqual(result, ((local_counts[0], 0), (0, remote_counts[1])))

    def test_layer_count_fallback_preserves_values_above_int32_range(self) -> None:
        device = torch.device("cuda")
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))
        int32_max = torch.iinfo(torch.int32).max
        local_counts = (int32_max + 1, 0, 3, int32_max + 4, 5, 0, 1, 2, int32_max + 9)
        remote_counts = (0, int32_max + 2, int32_max + 3, 4, int32_max + 5, 1, 0, 3, int32_max + 10)
        gather_calls = 0

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            nonlocal gather_calls
            self.assertEqual(local.dtype, torch.int64)
            if gather_calls == 0:
                remote_packet = torch.tensor(
                    _layer_count_packet_values(remote_counts, schema_hash),
                    dtype=fused_gather._COUNT_METADATA_DTYPE,
                    device=device,
                )
                output.copy_(torch.stack((local, remote_packet)).reshape(-1))
            else:
                torch.testing.assert_close(
                    local,
                    torch.tensor(local_counts, dtype=fused_gather._COUNT_METADATA_DTYPE, device=device),
                    rtol=0,
                    atol=0,
                )
                output.copy_(
                    torch.tensor(
                        (*local_counts, *remote_counts),
                        dtype=fused_gather._COUNT_METADATA_DTYPE,
                        device=device,
                    )
                )
            gather_calls += 1

        with patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather):
            result = fused_gather._all_gather_layer_counts(2, local_counts, device, schema_hash)

        self.assertEqual(gather_calls, 2)
        self.assertEqual(result, tuple(zip(local_counts, remote_counts, strict=True)))

    def test_layer_counts_reject_negative_inline_count(self) -> None:
        device = torch.device("cuda")
        local_counts = (2, 3, 0, 1, 2, 3, 4, 5)
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            remote_counts = (2, 3, 0, 1, 2, 3, 4, -1)
            remote = torch.tensor(
                _layer_count_packet_values(remote_counts, schema_hash),
                dtype=fused_gather._COUNT_METADATA_DTYPE,
                device=device,
            )
            output.copy_(torch.stack((local, remote)).reshape(-1))

        with (
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather) as all_gather,
            self.assertRaisesRegex(RuntimeError, r"negative layer counts: \(\(1, 7, -1\),\)"),
        ):
            fused_gather._all_gather_layer_counts(2, local_counts, device, schema_hash)

        all_gather.assert_called_once()

    def test_layer_counts_retain_full_vector_fallback_above_capacity(self) -> None:
        device = torch.device("cuda")
        local_counts = (2, 0, 3, 1, 7, 0, 2, 9, 6)
        remote_counts = (5, 4, 1, 6, 8, 3, 0, 4, 2)
        self.assertEqual(len(local_counts), fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY + 1)
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            self.assertEqual(local.dtype, fused_gather._COUNT_METADATA_DTYPE)
            if local.numel() == fused_gather._LAYER_COUNT_PACKET_WIDTH:
                torch.testing.assert_close(
                    local,
                    torch.tensor(
                        _layer_count_packet_values(local_counts, schema_hash),
                        dtype=fused_gather._COUNT_METADATA_DTYPE,
                        device=device,
                    ),
                    rtol=0,
                    atol=0,
                )
                remote = torch.tensor(
                    _layer_count_packet_values(remote_counts, schema_hash),
                    dtype=fused_gather._COUNT_METADATA_DTYPE,
                    device=device,
                )
                output.copy_(torch.stack((local, remote)).reshape(-1))
                return
            torch.testing.assert_close(
                local,
                torch.tensor(local_counts, dtype=fused_gather._COUNT_METADATA_DTYPE, device=device),
                rtol=0,
                atol=0,
            )
            output.copy_(
                torch.tensor((*local_counts, *remote_counts), dtype=fused_gather._COUNT_METADATA_DTYPE, device=device)
            )

        with patch.object(
            torch.distributed,
            "all_gather_into_tensor",
            side_effect=fake_all_gather,
        ) as all_gather:
            counts_by_layer = fused_gather._all_gather_layer_counts(2, local_counts, device, schema_hash)

        self.assertEqual(all_gather.call_count, 2)
        self.assertEqual(counts_by_layer, tuple(zip(local_counts, remote_counts, strict=True)))

    def test_layer_counts_reject_arity_crossing_capacity_before_fallback(self) -> None:
        device = torch.device("cuda")
        local_counts = (2,) * fused_gather._LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            remote = local.clone()
            remote[fused_gather._LAYER_COUNT_PACKET_LAYER_COUNT_INDEX] += 1
            output.copy_(torch.stack((local, remote)).reshape(-1))

        with (
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather) as all_gather,
            self.assertRaisesRegex(RuntimeError, "layer count differs across ranks"),
        ):
            fused_gather._all_gather_layer_counts(2, local_counts, device, schema_hash)

        all_gather.assert_called_once()

    def test_layer_counts_reject_negative_count_from_fallback_vector(self) -> None:
        device = torch.device("cuda")
        local_counts = (2, 0, 3, 1, 7, 0, 2, 9, 6)
        schema_hash = tuple(range(fused_gather._COMMUNICATION_SCHEMA_HASH_WORD_COUNT))

        def fake_all_gather(output: torch.Tensor, local: torch.Tensor, **_kwargs: Any) -> None:
            if local.numel() == fused_gather._LAYER_COUNT_PACKET_WIDTH:
                output.copy_(torch.stack((local, local)).reshape(-1))
                return
            remote = torch.tensor(
                (*local_counts[:-1], -1),
                dtype=fused_gather._COUNT_METADATA_DTYPE,
                device=device,
            )
            output.copy_(torch.stack((local, remote)).reshape(-1))

        with (
            patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather) as all_gather,
            self.assertRaisesRegex(RuntimeError, r"negative layer counts: \(\(1, 8, -1\),\)"),
        ):
            fused_gather._all_gather_layer_counts(2, local_counts, device, schema_hash)

        self.assertEqual(all_gather.call_count, 2)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Cannot test multi-GPU correctness on a single GPU")
    def test_two_rank_nccl_covers_inline_fallback_grouped_prefetch_empty_and_error_routes(self) -> None:
        world_size = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = str(Path(temp_dir) / "nccl-init")
            workers = [
                subprocess.Popen(
                    [sys.executable, __file__, "--nccl-oracle-worker", str(rank), str(world_size), init_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for rank in range(world_size)
            ]
            outputs: list[str] = []
            try:
                deadline = time.monotonic() + 45
                for worker in workers:
                    output, _ = worker.communicate(timeout=max(0.0, deadline - time.monotonic()))
                    outputs.append(output)
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        worker.kill()
                        worker.wait()

            for rank, (worker, output) in enumerate(zip(workers, outputs, strict=True)):
                self.assertEqual(worker.returncode, 0, f"NCCL oracle rank {rank} failed:\n{output}")

    def test_non_nccl_backend_is_rejected_before_collectives(self) -> None:
        parameters = {
            "first": {"value": torch.ones(1, 1, device="cuda", requires_grad=True)},
            "second": {"value": torch.ones(1, 1, device="cuda", requires_grad=True)},
        }
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_backend", return_value="not-nccl"),
            patch.object(fused_gather, "_all_gather_layer_counts") as gather_counts,
            self.assertRaisesRegex(RuntimeError, "requires the NCCL backend"),
        ):
            fused_gather._fused_all_gather_gaussian_parameters(
                ["first", "second"],
                parameters,
                process_group=cast(torch.distributed.ProcessGroup, object()),
            )

        gather_counts.assert_not_called()

    def test_higher_order_backward_fails_instead_of_using_python_pack(self) -> None:
        value = torch.ones(1, 1, device="cuda", requires_grad=True)
        layout = fused_gather._make_layer_layout("layer", {"value": value})

        def fake_payload(output: torch.Tensor, padded: torch.Tensor, **_kwargs: Any) -> None:
            _fake_two_rank_payload_gather(output, padded)

        fused_gather._cached_gradient_pack_plan.cache_clear()
        try:
            with (
                patch.object(fused_gather, "_all_gather_layer_counts", return_value=((1, 1),)),
                patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_payload),
                patch.object(fused_gather.gaussian_collective_payload_ops, "_pack_gradients_trusted") as pack_gradients,
                patch.object(torch.distributed, "reduce_scatter_tensor") as reduce_scatter,
                self.assertRaisesRegex(RuntimeError, "create_graph=True is not supported"),
            ):
                _counts, gathered = fused_gather._FusedGaussianParameterGather.apply(
                    (layout,),
                    fused_gather._PayloadIntegerRangeValidationState(),
                    0,
                    2,
                    cast(torch.distributed.ProcessGroup, object()),
                    None,
                    value,
                )
                torch.autograd.grad(gathered.square().sum(), value, create_graph=True)

            pack_gradients.assert_not_called()
            reduce_scatter.assert_not_called()
        finally:
            fused_gather._cached_gradient_pack_plan.cache_clear()

    def test_cuda_pack_failure_propagates_before_reduce_scatter(self) -> None:
        value = torch.ones(1, 1, device="cuda", requires_grad=True)
        layout = fused_gather._make_layer_layout("layer", {"value": value})

        def fake_payload(output: torch.Tensor, padded: torch.Tensor, **_kwargs: Any) -> None:
            _fake_two_rank_payload_gather(output, padded)

        fused_gather._cached_gradient_pack_plan.cache_clear()
        try:
            with (
                patch.object(fused_gather, "_all_gather_layer_counts", return_value=((1, 1),)),
                patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_payload),
                patch.object(
                    fused_gather.gaussian_collective_payload_ops,
                    "_pack_gradients_trusted",
                    side_effect=RuntimeError("injected CUDA pack failure"),
                ) as pack_gradients,
                patch.object(torch.distributed, "reduce_scatter_tensor") as reduce_scatter,
                self.assertRaisesRegex(RuntimeError, "injected CUDA pack failure"),
            ):
                _counts, gathered = fused_gather._FusedGaussianParameterGather.apply(
                    (layout,),
                    fused_gather._PayloadIntegerRangeValidationState(),
                    0,
                    2,
                    cast(torch.distributed.ProcessGroup, object()),
                    None,
                    value,
                )
                gathered.sum().backward()

            pack_gradients.assert_called_once()
            reduce_scatter.assert_not_called()
        finally:
            fused_gather._cached_gradient_pack_plan.cache_clear()

    def test_cuda_packer_preserves_unused_gradient_semantics(self) -> None:
        device = torch.device("cuda")
        used = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
        unused = torch.tensor([[5.0], [6.0]], device=device, requires_grad=True)
        globally_unused = torch.tensor([[7.0, 8.0]], device=device, requires_grad=True)
        layouts = (
            fused_gather._make_layer_layout("active", {"used": used, "unused": unused}),
            fused_gather._make_layer_layout("globally_unused", {"value": globally_unused}),
        )
        counts = ((2, 1), (1, 2))
        packed_gradient: Optional[torch.Tensor] = None
        real_pack_gradients = fused_gather.gaussian_collective_payload_ops._pack_gradients_trusted

        def fake_payload(output: torch.Tensor, padded: torch.Tensor, **_kwargs: Any) -> None:
            _fake_two_rank_payload_gather(output, padded)

        def fake_reduce_scatter(
            output: torch.Tensor,
            rank_major_gradients: torch.Tensor,
            **_kwargs,
        ) -> None:
            nonlocal packed_gradient
            packed_gradient = rank_major_gradients.detach().clone()
            output.copy_(rank_major_gradients[: output.numel()])

        fused_gather._cached_gradient_pack_plan.cache_clear()
        try:
            with (
                patch.object(fused_gather, "_all_gather_layer_counts", return_value=counts),
                patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_payload),
                patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
                patch.object(torch.distributed, "reduce_scatter_tensor", side_effect=fake_reduce_scatter),
                patch.object(
                    fused_gather.gaussian_collective_payload_ops,
                    "_pack_gradients_trusted",
                    side_effect=real_pack_gradients,
                ) as pack_gradients,
            ):
                _counts, gathered_used, _gathered_unused, _gathered_globally_unused = (
                    fused_gather._FusedGaussianParameterGather.apply(
                        layouts,
                        fused_gather._PayloadIntegerRangeValidationState(),
                        0,
                        2,
                        cast(torch.distributed.ProcessGroup, object()),
                        None,
                        used,
                        unused,
                        globally_unused,
                    )
                )
                gathered_used.sum().backward()

            pack_gradients.assert_called_once()
            self.assertIsNotNone(packed_gradient)
            torch.testing.assert_close(
                packed_gradient,
                torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    device=device,
                ),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(used.grad, torch.ones_like(used), rtol=0, atol=0)
            self.assertIsNotNone(unused.grad)
            torch.testing.assert_close(unused.grad, torch.zeros_like(unused), rtol=0, atol=0)
            self.assertIsNone(globally_unused.grad)
        finally:
            fused_gather._cached_gradient_pack_plan.cache_clear()

    def test_cuda_packer_preserves_frozen_field_autograd_semantics(self) -> None:
        device = torch.device("cuda")
        active = torch.tensor([[1.0]], device=device, requires_grad=True)
        frozen = torch.tensor([[2.0]], device=device)
        fully_frozen = torch.tensor([[3.0]], device=device)
        active_reference = active.detach().clone().requires_grad_()
        frozen_reference = frozen.detach().clone()
        layouts = (
            fused_gather._make_layer_layout("mixed", {"active": active, "frozen": frozen}),
            fused_gather._make_layer_layout("fully_frozen", {"value": fully_frozen}),
        )
        packed_gradient: Optional[torch.Tensor] = None
        real_pack_gradients = fused_gather.gaussian_collective_payload_ops._pack_gradients_trusted

        def fake_payload(output: torch.Tensor, padded: torch.Tensor, **_kwargs: Any) -> None:
            _fake_two_rank_payload_gather(output, padded)

        def fake_reduce_scatter(
            output: torch.Tensor,
            rank_major_gradients: torch.Tensor,
            **_kwargs,
        ) -> None:
            nonlocal packed_gradient
            packed_gradient = rank_major_gradients.detach().clone()
            output.copy_(rank_major_gradients[: output.numel()])

        reference_payload = torch.cat((active_reference, frozen_reference), dim=-1)
        reference_gathered = torch.cat((reference_payload, torch.zeros_like(reference_payload)), dim=0)
        _reference_active, reference_frozen = torch.split(reference_gathered, [1, 1], dim=-1)

        fused_gather._cached_gradient_pack_plan.cache_clear()
        try:
            with (
                patch.object(fused_gather, "_all_gather_layer_counts", return_value=((1, 1), (1, 1))),
                patch.object(torch.distributed, "all_gather_into_tensor", side_effect=fake_payload),
                patch.object(torch.distributed, "get_backend", return_value=torch.distributed.Backend.NCCL),
                patch.object(torch.distributed, "reduce_scatter_tensor", side_effect=fake_reduce_scatter),
                patch.object(
                    fused_gather.gaussian_collective_payload_ops,
                    "_pack_gradients_trusted",
                    side_effect=real_pack_gradients,
                ) as pack_gradients,
            ):
                _counts, gathered_active, gathered_frozen, gathered_fully_frozen = (
                    fused_gather._FusedGaussianParameterGather.apply(
                        layouts,
                        fused_gather._PayloadIntegerRangeValidationState(),
                        0,
                        2,
                        cast(torch.distributed.ProcessGroup, object()),
                        None,
                        active,
                        frozen,
                        fully_frozen,
                    )
                )
                self.assertTrue(gathered_active.requires_grad)
                self.assertTrue(gathered_frozen.requires_grad)
                self.assertEqual(gathered_frozen.requires_grad, reference_frozen.requires_grad)
                self.assertFalse(gathered_fully_frozen.requires_grad)
                gathered_frozen.sum().backward()

            reference_frozen.sum().backward()
            torch.cuda.synchronize(device)
            pack_gradients.assert_called_once()
            self.assertIsNotNone(packed_gradient)
            torch.testing.assert_close(packed_gradient, torch.zeros(2, device=device), rtol=0, atol=0)
            self.assertIsNotNone(active.grad)
            self.assertIsNotNone(active_reference.grad)
            torch.testing.assert_close(active.grad, active_reference.grad, rtol=0, atol=0)
            torch.testing.assert_close(active.grad, torch.zeros_like(active), rtol=0, atol=0)
            self.assertIsNone(frozen.grad)
            self.assertIsNone(frozen_reference.grad)
            self.assertIsNone(fully_frozen.grad)
        finally:
            fused_gather._cached_gradient_pack_plan.cache_clear()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Cannot test multi-GPU correctness on a single GPU")
    def test_two_rank_nccl_selective_adam_visibility_union_fixes_owner_skip(self) -> None:
        world_size = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = str(Path(temp_dir) / "selective-adam-nccl-init")
            workers = [
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--selective-adam-visibility-worker",
                        str(rank),
                        str(world_size),
                        init_file,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for rank in range(world_size)
            ]
            outputs: list[str] = []
            try:
                deadline = time.monotonic() + 120
                for worker in workers:
                    output, _ = worker.communicate(timeout=max(0.0, deadline - time.monotonic()))
                    outputs.append(output)
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        worker.kill()
                        worker.wait()

            for rank, (worker, output) in enumerate(zip(workers, outputs, strict=True)):
                self.assertEqual(worker.returncode, 0, f"SelectiveAdam NCCL rank {rank} failed:\n{output}")
                self.assertNotIn("gloo", output.lower(), f"SelectiveAdam rank {rank} used Gloo:\n{output}")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--nccl-oracle-worker":
        _two_rank_nccl_oracle_worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif len(sys.argv) == 5 and sys.argv[1] == "--selective-adam-visibility-worker":
        _selective_adam_visibility_regression_worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        unittest.main()
