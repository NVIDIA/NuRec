# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Morton (Z-order) ordering for per-Gaussian rows.

Shared by the compact-artifact codec and by the checkpoint-capture reorder of the live model.
"""

from __future__ import annotations

import typing

import torch


__all__ = ["VOXEL_SIZE", "MortonOrder", "morton_sort_order"]

# Voxel edge length, in meters.
VOXEL_SIZE = 2.0

_MORTON_AXIS_BITS = 21
_MORTON_AXIS_MASK = (1 << _MORTON_AXIS_BITS) - 1
_MORTON_MAX_COORD = 1 << (2 * _MORTON_AXIS_BITS)


class MortonOrder(typing.NamedTuple):
    """A Morton permutation together with the codes it was derived from.

    codes_low and codes_high are in the original row order, not permuted. codes_high is None unless the scene needs
    more than 21 bits per axis.
    """

    order: torch.Tensor
    codes_low: torch.Tensor
    codes_high: torch.Tensor | None


def morton_sort_order(
    positions: torch.Tensor,
    voxel_size: float = VOXEL_SIZE,
    label: str = "positions",
) -> MortonOrder:
    """Return the permutation that sorts positions into Morton order.

    Positions are voxelized, then the three voxel coordinates are bit-interleaved into one 63-bit code. Scenes wider
    than 21 bits on an axis get a second code segment which is sorted as the high key. Output is deterministic on
    CUDA.

    Args:
        positions: (N, 3) float tensor.
        voxel_size: Voxel edge length in meters. Must be positive.
        label: Name used in error messages, e.g. the layer prefix.

    Returns:
        The permutation and the Morton codes, in original row order.

    Raises:
        ValueError: If positions are malformed, non-finite, or exceed the two-segment Morton range.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{label} must be (N, 3), got {tuple(positions.shape)}")

    empty = torch.empty(0, dtype=torch.long, device=positions.device)
    if positions.shape[0] == 0:
        return MortonOrder(order=empty, codes_low=empty, codes_high=None)

    voxel_coords_f = torch.floor(positions / voxel_size)

    # NaN/inf would silently corrupt voxel coords
    is_finite = torch.isfinite(voxel_coords_f).all()
    extent = (voxel_coords_f.amax(dim=0).double() - voxel_coords_f.amin(dim=0).double()).max()

    # Batch the transfer
    guards = torch.stack((is_finite.double(), extent)).cpu()
    if not bool(guards[0]):
        raise ValueError(f"{label} contains non-finite values")

    max_coord = int(guards[1])
    if max_coord >= _MORTON_MAX_COORD:
        raise ValueError(
            f"{label}: voxel coord {max_coord} exceeds 42-bit Morton limit. "
            f"Scene extent > {_MORTON_MAX_COORD * voxel_size:.0f} m along some axis.",
        )

    # Shift all coords to positive while keeping the spatial ordering
    voxel_coords = voxel_coords_f.long()
    voxel_coords -= voxel_coords.min(dim=0).values

    codes_low = _encode_morton_3d(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2])
    # Stable sort makes equal-code rows reproducible.
    order = codes_low.argsort(stable=True)
    codes_high = None
    if max_coord >= (1 << _MORTON_AXIS_BITS):
        high_coords = voxel_coords >> _MORTON_AXIS_BITS
        codes_high = _encode_morton_3d(high_coords[:, 0], high_coords[:, 1], high_coords[:, 2])
        order = order[codes_high[order].argsort(stable=True)]

    return MortonOrder(order=order, codes_low=codes_low, codes_high=codes_high)


def _encode_morton_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Interleave three 21-bit integer coordinates into a 63-bit Morton code."""
    return _part_1by2(x) | (_part_1by2(y) << 1) | (_part_1by2(z) << 2)


def _part_1by2(n: torch.Tensor) -> torch.Tensor:
    """Spread each of 21 low bits of n with two zeros between them."""
    n = n.long() & _MORTON_AXIS_MASK
    n = (n | (n << 32)) & 0x1F00000000FFFF
    n = (n | (n << 16)) & 0x1F0000FF0000FF
    n = (n | (n << 8)) & 0x100F00F00F00F00F
    n = (n | (n << 4)) & 0x10C30C30C30C30C3
    n = (n | (n << 2)) & 0x1249249249249249
    return n
