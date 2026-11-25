# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cuboid-local donors and growth, with one native layer-wide row publication."""

import heapq

import torch

from gsplat.strategy import MCMCOperation, MCMCRowPlan

from nurec.models.gaussians.construction_barriers_model import ConstructionBarriersGaussianModel
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import (
    _apply_and_publish,
    _build_owners,
    mcmc_dead_mask,
)
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes


def growth_allocations(counts: list[int], cap: int, *, eligible: list[bool] | None = None) -> list[int]:
    """Grow eligible cuboids within equal final shares and the native layer-wide 5% limit."""
    if not counts:
        return []
    if type(cap) is not int or cap <= 0 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("Cuboid growth requires positive counts and capacity")
    if eligible is None:
        eligible = [True] * len(counts)
    if len(eligible) != len(counts) or any(type(value) is not bool for value in eligible):
        raise ValueError("Cuboid growth eligibility requires one bool per cuboid")
    base, remainder = divmod(cap, len(counts))
    quotas = [base + (i < remainder) for i in range(len(counts))]
    if any(n > q for n, q in zip(counts, quotas)):
        raise ValueError("Initial cuboid population exceeds its equal capacity share")
    current = sum(counts)
    headroom = sum(q - n for n, q, enabled in zip(counts, quotas, eligible) if enabled)
    requested = min(headroom, max(0, min(cap, int(1.05 * current)) - current))
    additions = [0] * len(counts)
    candidates = [(n, i) for i, n in enumerate(counts) if eligible[i] and n < quotas[i]]
    heapq.heapify(candidates)
    # Only scalar counts live in this heap, never per-point tensors.
    for _ in range(requested):
        n, i = heapq.heappop(candidates)
        additions[i] += 1
        if n + 1 < quotas[i]:
            heapq.heappush(candidates, (n + 1, i))
    return additions


def _batched_topology_plan(
    order: torch.Tensor,
    counts: list[int],
    opacities: torch.Tensor,
    scales: torch.Tensor,
    dead: torch.Tensor,
    *,
    binoms: torch.Tensor,
    min_opacity: float,
    generator: torch.Generator,
    operation: MCMCOperation,
    cap: int,
) -> MCMCRowPlan | None:
    """Build a layer-wide relocation or growth plan with track-local donors.

    Args:
        order: CUDA int64 row permutation, stably sorted by cuboid identity.
        counts: Positive cuboid sizes partitioning ``order`` in track order.
        opacities: Activated float32 CUDA opacities with shape ``[N]``.
        scales: Activated float32 scales with shape ``[N, 3]`` on the same device.
        dead: Boolean CUDA relocation mask with shape ``[N]`` in original row order.
        binoms: Native sampler's square float32 CUDA binomial table.
        min_opacity: Native sampler's lower source-opacity clamp.
        generator: CUDA RNG used for donor draws.
        operation: Relocation or addition.
        cap: Layer capacity for addition; unused for relocation.

    Returns:
        A layer-wide topology plan, or ``None`` when no track requests valid draws.
        Tracks without live, nonzero-weight donors retain every row and
        consume no random draws.

    Inputs are not mutated. The caller guarantees finite activated opacities in
    [0, 1], positive finite scales and a valid binomial table, and runs under
    ``torch.no_grad()``. Correction stays on the GPU without value readbacks.
    CUDA uses one batched uniform draw, changing RNG consumption from native
    multinomial. Track metadata and candidate compaction still synchronize.
    """
    from nurec.libs.cuboid_relocation import correct_relocation, prepare_tracks, sample_tracks

    sorted_dead, group, metadata_tensor = prepare_tracks(order, counts, opacities, dead)
    metadata = metadata_tensor.tolist()
    candidate_counts = [size - n for size, (n, _) in zip(counts, metadata)]
    if operation == MCMCOperation.ADDITION:
        additions = growth_allocations(counts, cap, eligible=[n > 0 for n in candidate_counts])
        draws = [(n, nonzero) for n, (_, nonzero) in zip(additions, metadata)]
    else:
        draws = metadata
    if not any(n and nonzero for n, nonzero in draws):
        return None

    # Compact live donors once for the layer, for both growth and relocation.
    candidates = order[~sorted_dead]
    candidate_weights = opacities[candidates]
    source_indices = sample_tracks(candidates, candidate_weights, candidate_counts, draws, generator)
    keep, revised_opacities, revised_scales = correct_relocation(
        order, sorted_dead, group, metadata_tensor, source_indices, opacities, scales, binoms, min_opacity
    )
    if operation == MCMCOperation.ADDITION:
        keep.fill_(True)
    return MCMCRowPlan(
        operation=operation,
        keep=keep,
        source_indices=source_indices,
        revised_opacities=revised_opacities,
        revised_scales=revised_scales,
        storage_min_opacity=float(min_opacity),
    )


@torch.no_grad()
def cuboid_topology(
    selected: SelectedGaussianNodes,
    *,
    operation: MCMCOperation,
    cap: int,
    binoms: torch.Tensor,
    min_opacity: float,
    generator: torch.Generator,
    invisible_steps: torch.nn.ParameterDict | None,
    max_invisible_steps: int | None = None,
) -> int:
    """Relocate or grow within each cuboid, then publish one layer-wide row plan.

    The caller must enforce the strategy's single-GPU contract. Donors never
    cross track boundaries; row identities, optimizer moments and visibility
    counters follow the same native publication. Cuboids without a live donor
    retain their rows unchanged, without relocation or growth. Return the
    number of donor rows sampled, or zero when every group is a no-op.
    """
    owners, snapshots = _build_owners(selected, invisible_steps)
    if len(owners) != 1:
        raise ValueError("Cuboid-local topology requires one independent layer owner")
    scene = owners[0].scene
    model = selected.gaussians_nodes[scene.id]
    if not isinstance(model, ConstructionBarriersGaussianModel):
        raise TypeError("Cuboid-local topology requires a construction-barrier layer")
    ids = getattr(model, "barrier_object_id", None)
    count = scene.num_gaussians()
    if ids is None or ids.shape != (count,) or ids.dtype != torch.int64 or not ids.is_cuda:
        raise ValueError("Cuboid-local MCMC requires row-aligned CUDA int64 barrier_object_id")
    if ids.device != scene.splats["means"].device:
        raise ValueError("Cuboid identities and parameters must share a device")
    if count == 0:
        return 0
    order = torch.argsort(ids, stable=True)
    group_ids, sizes = torch.unique_consecutive(ids[order], return_counts=True)
    counts = sizes.tolist()
    expected = sorted(row["track_id"] for row in model._cuboid_records)
    if group_ids.tolist() != expected:
        raise RuntimeError("A source cuboid identity disappeared before grouped topology")
    opacities = torch.sigmoid(scene.splats["opacities"].view(-1))
    scales = torch.exp(scene.splats["scales"]).clamp_min_(torch.finfo(torch.float32).tiny)
    visibility = None if invisible_steps is None or max_invisible_steps is None else invisible_steps[scene.id].view(-1)
    dead = mcmc_dead_mask(opacities, visibility, min_opacity=min_opacity, max_invisible_steps=max_invisible_steps)
    if operation not in (MCMCOperation.RELOCATION, MCMCOperation.ADDITION):
        raise ValueError("Unsupported cuboid topology operation")
    plan = _batched_topology_plan(
        order,
        counts,
        opacities,
        scales,
        dead,
        binoms=binoms,
        min_opacity=min_opacity,
        generator=generator,
        operation=operation,
        cap=cap,
    )
    if plan is None:
        return 0
    # Parameters, Adam state, track IDs and visibility all share one row map.
    _apply_and_publish(selected, owners, snapshots, plan, invisible_steps)
    return plan.source_indices.numel()
