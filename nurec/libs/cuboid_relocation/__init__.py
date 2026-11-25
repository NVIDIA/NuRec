# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA track preparation, weighted sampling and MCMC relocation correction."""

import importlib
import itertools

from collections.abc import Sequence
from typing import Any

import torch


_cuda: Any = importlib.import_module("nurec.libs.cuboid_relocation.libcuboid_relocation_cuda_cc")
constrain_perturbation = _cuda.constrain_perturbation


def _offsets(counts: list[int], device: torch.device) -> torch.Tensor:
    """Return contiguous int64 CUDA offsets [G+1] for G nonnegative counts on device."""
    return torch.tensor([0, *itertools.accumulate(counts)], dtype=torch.int64, device=device)


def prepare_tracks(
    order: torch.Tensor, counts: list[int], opacities: torch.Tensor, dead: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute track masks and counts on the current CUDA stream, without gradients.

    Args:
        order: Int64 [N] row permutation grouping tracks, N < 2**31.
        counts: Positive track lengths summing to N.
        opacities: Float32 [N] finite, nonnegative weights in original row order.
        dead: Bool [N] relocation mask in original row order.

    All tensors share a CUDA device; strides are supported. Inputs are unchanged.
    The caller guarantees a valid permutation and partition.

    Returns:
        Bool [N] sorted dead mask, int64 [N] sorted track indices, and int64
        [G,2] (dead count, has nonzero live weight), on the input device.

    Raises:
        RuntimeError: Invalid tensor device, dtype or shape.
    """
    sorted_dead = torch.empty_like(dead)
    group = torch.empty_like(order)
    metadata = torch.empty((len(counts), 2), dtype=torch.int64, device=order.device)
    _cuda.prepare_tracks(order, _offsets(counts, order.device), opacities, dead, sorted_dead, metadata, group)
    return sorted_dead, group, metadata


def sample_tracks(
    rows: torch.Tensor,
    weights: torch.Tensor,
    counts: list[int],
    metadata: Sequence[Sequence[int]],
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample track-local donors on the current CUDA stream, without gradients.

    Args:
        rows: Int64 [C] candidate IDs grouped by track, C < 2**31.
        weights: Float32 [C] finite, nonnegative weights aligned with rows.
        counts: Nonnegative track lengths summing to C.
        metadata: Per-track (requested draws, has nonzero donor weight).
            Tracks with draws require candidates; zero weights consume no RNG.
        generator: CUDA generator on the input device.

    Tensors share a CUDA device; strides are supported. The caller guarantees
    valid counts/metadata. Inputs are unchanged.

    Returns:
        Int64 CUDA [S] donor IDs in track order; no draws returns an empty tensor.

    Raises:
        RuntimeError: Invalid tensor device, dtype or shape for nonzero work.
    """
    draws = [n if nonzero else 0 for n, nonzero in metadata]
    total = sum(draws)
    if not total:
        return rows.new_empty(0)
    # Batched uniforms change RNG consumption and the training trajectory versus
    # per-track multinomial. Full-training quality/speedup need Qingjie's validation.
    uniforms = torch.rand(total, dtype=torch.float64, device=rows.device, generator=generator)
    offsets = _offsets(counts, rows.device)
    cdf = torch.empty_like(weights, dtype=torch.float64)
    _cuda.track_cdf(weights=weights, offsets=offsets, cdf=cdf)
    sources = rows.new_empty(total)
    _cuda.sample_tracks(rows, cdf, offsets, _offsets(draws, rows.device), uniforms, sources)
    return sources


def correct_relocation(
    order: torch.Tensor,
    sorted_dead: torch.Tensor,
    group: torch.Tensor,
    metadata: torch.Tensor,
    sources: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    binoms: torch.Tensor,
    min_opacity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the keep mask and corrected donor values on CUDA, without gradients.

    Args:
        order: Int64 [N] row permutation; N and source count S must be < 2**31.
        sorted_dead: Bool [N] dead mask in order's row order.
        group: Int64 [N] track indices in order's row order.
        metadata: Int64 [G,2] (dead count, has nonzero live weight).
        sources: Int64 [S] sampled original row IDs.
        opacities: Float32 [N] finite activated opacities in [0,1].
        scales: Float32 [N,3] positive finite activated scales.
        binoms: Legacy float32 [B,B] table, B > 0; B caps ratios, values are unused.
        min_opacity: Source-opacity floor in (0,1).

    Inputs share a CUDA device and may be strided. The caller guarantees valid
    values/indices; inputs are unchanged and no GPU values are read back.

    Returns:
        CUDA keep bool [N], revised opacities float32 [S], scales float32 [S,3].

    Raises:
        RuntimeError: Invalid tensor metadata or min_opacity.
    """
    occurrences = torch.zeros_like(order, dtype=torch.int32)
    keep = torch.empty_like(sorted_dead)
    revised_opacities = opacities.new_empty(sources.numel())
    revised_scales = scales.new_empty((sources.numel(), 3))
    _cuda.correct_relocation(
        order,
        sorted_dead,
        group,
        metadata,
        sources,
        opacities,
        scales,
        binoms,
        min_opacity,
        occurrences,
        keep,
        revised_opacities,
        revised_scales,
    )
    return keep, revised_opacities, revised_scales
