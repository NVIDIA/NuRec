# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cuboid-local donors and growth, with one native layer-wide row publication."""

import heapq

import torch

from gsplat.strategy import MCMCOperation, MCMCRowPlan, sample_mcmc_relocation

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
    groups = order.split(counts)
    opacities = torch.sigmoid(scene.splats["opacities"].view(-1))
    scales = torch.exp(scene.splats["scales"]).clamp_min_(torch.finfo(torch.float32).tiny)
    keep = torch.ones(count, dtype=torch.bool, device=ids.device)
    visibility = None if invisible_steps is None else invisible_steps[scene.id].view(-1)
    dead = mcmc_dead_mask(opacities, visibility, min_opacity=min_opacity, max_invisible_steps=max_invisible_steps)
    candidates_per_group = [rows[~dead[rows]] for rows in groups]
    relocation = operation == MCMCOperation.RELOCATION
    if relocation:
        additions = None
    elif operation == MCMCOperation.ADDITION:
        additions = growth_allocations(counts, cap, eligible=[rows.numel() > 0 for rows in candidates_per_group])
        if not sum(additions):
            return 0
    else:
        raise ValueError("Unsupported cuboid topology operation")

    sources, revised_opacities, revised_scales = [], [], []
    for index, (rows, candidates) in enumerate(zip(groups, candidates_per_group)):
        if candidates.numel() == 0:
            # A weak fallback donor would be lifted by gsplat's opacity floor.
            # Leave suppressed cuboids unchanged for both relocation and growth.
            continue
        if relocation:
            dead_rows = rows[dead[rows]]
            sample_count = dead_rows.numel()
        else:
            assert additions is not None
            sample_count = additions[index]
        if sample_count == 0:
            continue
        samples = sample_mcmc_relocation(
            opacities=opacities[candidates],
            scales=scales[candidates],
            num_samples=sample_count,
            binoms=binoms,
            min_opacity=min_opacity,
            generator=generator,
        )
        if samples.sampled_indices.numel() == 0:
            # Native zero-weight behavior: retain this cuboid's existing rows.
            continue
        if samples.sampled_indices.numel() != sample_count:
            raise RuntimeError("Cuboid sampler returned a partial sample")
        sources.append(candidates[samples.sampled_indices])
        revised_opacities.append(samples.revised_opacities)
        revised_scales.append(samples.revised_scales)
        if relocation:
            keep[dead_rows] = False
    if not sources:
        return 0
    plan = MCMCRowPlan(
        operation=operation,
        keep=keep,
        source_indices=torch.cat(sources),
        revised_opacities=torch.cat(revised_opacities),
        revised_scales=torch.cat(revised_scales),
        storage_min_opacity=float(min_opacity),
    )
    # Parameters, Adam state, track IDs and visibility all share one row map.
    _apply_and_publish(selected, owners, snapshots, plan, invisible_steps)
    return plan.source_indices.numel()
