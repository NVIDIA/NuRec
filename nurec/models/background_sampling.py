# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from typing import Any, NamedTuple

import torch

from gsplat.scene.kernels.env_map_sample_ops import sample_env_map_cubemap, sample_env_map_cubemap_vjp
from torch.autograd.function import once_differentiable

from nurec.models.background_sampling_cuda import background_sampling_cuda


# This cache is process-local. A DDP rank normally uses one CUDA device and
# one cubemap size; four entries leave headroom for tools processing several
# sizes. Larger per-process working sets evict and rebuild the topology,
# affecting performance only.
_TOPOLOGY_CACHE_MAX_ENTRIES = 4

_CUBEMAP_FACE_COUNT = 6
_CUBEMAP_EDGES_PER_FACE = 4
_CUBEMAP_CORNERS_PER_FACE = 4
_CUBEMAP_CORNER_SOURCE_COUNT = 3

# OpenGL cubemap face order expected by gsplat's sampler.
(
    _POSITIVE_X_FACE,
    _NEGATIVE_X_FACE,
    _POSITIVE_Y_FACE,
    _NEGATIVE_Y_FACE,
    _POSITIVE_Z_FACE,
    _NEGATIVE_Z_FACE,
) = range(_CUBEMAP_FACE_COUNT)

_RAY_COMPONENT_COUNT = 3

_SEAMLESS_BORDER_WIDTH = 1
_SEAMLESS_PADDING_PER_AXIS = 2 * _SEAMLESS_BORDER_WIDTH

_FUSED_SAMPLER_CHANNEL_COUNT = 3


class _CubemapPaddingTopology(NamedTuple):
    border_source_indices: torch.Tensor
    border_destination_indices: torch.Tensor
    corner_destination_indices: torch.Tensor
    single_border_source_indices: torch.Tensor
    single_border_destination_indices: torch.Tensor
    repeated_border_source_indices: torch.Tensor
    repeated_border_destination_indices: torch.Tensor
    repeated_edge_source_count: int


@lru_cache(maxsize=_TOPOLOGY_CACHE_MAX_ENTRIES)
def _cached_cubemap_padding_topology(
    cubemap_size: int, device_type: str, device_index: int | None
) -> _CubemapPaddingTopology:
    """Build and cache the immutable source mapping for a padded cubemap."""
    if cubemap_size < 1:
        raise ValueError(f"Cubemap size must be positive, got {cubemap_size}")

    # Build the topology on the CPU once. Each DDP rank has its own process-local
    # cache, and the device is part of the key, so cached CUDA tensors cannot be
    # reused by another rank or device.
    source = torch.arange(_CUBEMAP_FACE_COUNT * cubemap_size * cubemap_size).reshape(
        _CUBEMAP_FACE_COUNT, cubemap_size, cubemap_size
    )
    top = torch.stack(
        [
            source[_POSITIVE_Y_FACE, :, -1].flip(0),
            source[_POSITIVE_Y_FACE, :, 0],
            source[_NEGATIVE_Z_FACE, 0].flip(0),
            source[_POSITIVE_Z_FACE, -1],
            source[_POSITIVE_Y_FACE, -1],
            source[_POSITIVE_Y_FACE, 0].flip(0),
        ]
    )
    right = torch.stack(
        [
            source[_NEGATIVE_Z_FACE, :, 0],
            source[_POSITIVE_Z_FACE, :, 0],
            source[_POSITIVE_X_FACE, 0].flip(0),
            source[_POSITIVE_X_FACE, -1],
            source[_POSITIVE_X_FACE, :, 0],
            source[_NEGATIVE_X_FACE, :, 0],
        ]
    )
    bottom = torch.stack(
        [
            source[_NEGATIVE_Y_FACE, :, -1],
            source[_NEGATIVE_Y_FACE, :, 0].flip(0),
            source[_POSITIVE_Z_FACE, 0],
            source[_NEGATIVE_Z_FACE, -1].flip(0),
            source[_NEGATIVE_Y_FACE, 0],
            source[_NEGATIVE_Y_FACE, -1].flip(0),
        ]
    )
    left = torch.stack(
        [
            source[_POSITIVE_Z_FACE, :, -1],
            source[_NEGATIVE_Z_FACE, :, -1],
            source[_NEGATIVE_X_FACE, 0],
            source[_NEGATIVE_X_FACE, -1].flip(0),
            source[_NEGATIVE_X_FACE, :, -1],
            source[_POSITIVE_X_FACE, :, -1],
        ]
    )

    padded_size = cubemap_size + _SEAMLESS_PADDING_PER_AXIS
    padded = torch.arange(_CUBEMAP_FACE_COUNT * padded_size * padded_size).reshape(
        _CUBEMAP_FACE_COUNT, padded_size, padded_size
    )
    interior = slice(_SEAMLESS_BORDER_WIDTH, -_SEAMLESS_BORDER_WIDTH)
    edge_source_indices = torch.cat([top.flatten(), right.flatten(), bottom.flatten(), left.flatten()])
    edge_destination_indices = torch.cat(
        [
            padded[:, 0, interior].flatten(),
            padded[:, interior, -1].flatten(),
            padded[:, -1, interior].flatten(),
            padded[:, interior, 0].flatten(),
        ]
    )

    # At a cube corner there is no fourth face. Preserve seamless filtering by
    # recording the three texels that meet there in a fixed addition order.
    corner_source_indices = torch.stack(
        [
            torch.stack(
                [
                    top[:, 0],
                    left[:, 0],
                    source[:, 0, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    top[:, -1],
                    right[:, 0],
                    source[:, 0, -1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    bottom[:, 0],
                    left[:, -1],
                    source[:, -1, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    bottom[:, -1],
                    right[:, -1],
                    source[:, -1, -1],
                ],
                dim=-1,
            ),
        ],
        dim=1,
    ).reshape(_CUBEMAP_FACE_COUNT * _CUBEMAP_CORNERS_PER_FACE, _CUBEMAP_CORNER_SOURCE_COUNT)
    corner_destination_indices = torch.stack(
        [
            padded[:, 0, 0],
            padded[:, 0, -1],
            padded[:, -1, 0],
            padded[:, -1, -1],
        ],
        dim=1,
    ).flatten()
    border_source_indices = torch.cat([edge_source_indices, corner_source_indices.flatten()])
    border_destination_indices = torch.cat(
        [
            edge_destination_indices,
            corner_destination_indices.repeat_interleave(_CUBEMAP_CORNER_SOURCE_COUNT),
        ]
    )

    # Most source texels appear once in the border mapping, but each face-corner
    # texel appears repeatedly. Group those repeated entries so backward can
    # reduce them in a fixed order before updating each source exactly once.
    unique_source_indices, inverse_indices, source_counts = torch.unique(
        border_source_indices,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    repeated_source_groups = torch.nonzero(source_counts > 1, as_tuple=False).flatten()
    single_entry_mask = source_counts[inverse_indices] == 1
    single_border_positions = torch.nonzero(single_entry_mask, as_tuple=False).flatten()
    single_border_source_indices = border_source_indices.index_select(0, single_border_positions)
    single_border_destination_indices = border_destination_indices.index_select(0, single_border_positions)
    repeated_border_source_indices = unique_source_indices.index_select(0, repeated_source_groups)
    # Cube symmetry gives every repeated source the same total multiplicity,
    # including when S=1. Validate it explicitly because the grouped tensor and
    # backward's shared edge/corner split both depend on this invariant.
    assert source_counts.index_select(0, repeated_source_groups).unique().numel() == 1, (
        "Repeated cubemap border sources must have uniform multiplicity"
    )
    repeated_border_positions = torch.stack(
        [
            torch.nonzero(inverse_indices == source_group, as_tuple=False).flatten()
            for source_group in repeated_source_groups
        ]
    )
    repeated_edge_source_counts = (repeated_border_positions < edge_source_indices.numel()).sum(dim=1)
    assert repeated_edge_source_counts.unique().numel() == 1, (
        "Repeated cubemap border sources must have uniform edge multiplicity"
    )
    repeated_edge_source_count = int(repeated_edge_source_counts[0].item())
    repeated_border_destination_indices = border_destination_indices.index_select(
        0, repeated_border_positions.flatten()
    ).reshape(repeated_border_positions.shape)
    device = torch.device(device_type) if device_index is None else torch.device(device_type, device_index)
    return _CubemapPaddingTopology(
        border_source_indices.to(device),
        border_destination_indices.to(device),
        corner_destination_indices.to(device),
        single_border_source_indices.to(device),
        single_border_destination_indices.to(device),
        repeated_border_source_indices.to(device),
        repeated_border_destination_indices.to(device),
        repeated_edge_source_count,
    )


def _cubemap_padding_topology(cubemap_size: int, device: torch.device) -> _CubemapPaddingTopology:
    """Return the process-local cached topology for ``cubemap_size`` and ``device``."""
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _cached_cubemap_padding_topology(cubemap_size, device.type, device.index)


class _PadCubemapForSeamlessSampling(torch.autograd.Function):
    """Pure-ATen reference implementation of seamless cubemap padding.

    Building all 24 edge strips and 24 averaged corners from ordinary PyTorch tensor
    operations creates a large retained graph, plus repeated launch and copy
    overhead on every SkyEnvMap call. This function preserves the same forward
    mapping while grouping repeated corner contributions before its indexed VJP,
    so every accumulation destination is unique. Only ``cubemap`` is
    differentiable, and the cached topology tensors are immutable indices.
    """

    @staticmethod
    def forward(
        ctx,
        cubemap: torch.Tensor,
        topology: _CubemapPaddingTopology,
    ) -> torch.Tensor:
        border_source_indices = topology.border_source_indices
        border_destination_indices = topology.border_destination_indices
        corner_destination_indices = topology.corner_destination_indices
        _, height, width, n_channels = cubemap.shape
        edge_count = _CUBEMAP_FACE_COUNT * _CUBEMAP_EDGES_PER_FACE * height
        interior = slice(_SEAMLESS_BORDER_WIDTH, -_SEAMLESS_BORDER_WIDTH)
        contiguous_cubemap = cubemap.contiguous()
        flat_cubemap = contiguous_cubemap.view(-1, n_channels)
        padded = cubemap.new_empty(
            (
                _CUBEMAP_FACE_COUNT,
                height + _SEAMLESS_PADDING_PER_AXIS,
                width + _SEAMLESS_PADDING_PER_AXIS,
                n_channels,
            )
        )
        padded[:, interior, interior].copy_(contiguous_cubemap)
        flat_padded = padded.view(-1, n_channels)

        edge_texels = flat_cubemap.index_select(0, border_source_indices[:edge_count])
        flat_padded.index_copy_(0, border_destination_indices[:edge_count], edge_texels)

        corner_texels = flat_cubemap.index_select(0, border_source_indices[edge_count:]).reshape(
            _CUBEMAP_FACE_COUNT * _CUBEMAP_CORNERS_PER_FACE,
            _CUBEMAP_CORNER_SOURCE_COUNT,
            n_channels,
        )
        first_corner_source, second_corner_source, third_corner_source = corner_texels.unbind(dim=1)
        corner_values = (
            first_corner_source + second_corner_source + third_corner_source
        ) / _CUBEMAP_CORNER_SOURCE_COUNT
        flat_padded.index_copy_(0, corner_destination_indices, corner_values)

        _stash_padding_context(ctx, cubemap, topology)
        return padded

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        (
            single_border_source_indices,
            single_border_destination_indices,
            repeated_border_source_indices,
            repeated_border_destination_indices,
        ) = ctx.saved_tensors
        n_channels = grad_output.shape[-1]
        interior = slice(_SEAMLESS_BORDER_WIDTH, -_SEAMLESS_BORDER_WIDTH)
        flat_grad_output = grad_output.reshape(-1, n_channels)
        grad_cubemap = grad_output[:, interior, interior].contiguous()
        flat_grad_cubemap = grad_cubemap.view(-1, n_channels)

        if single_border_source_indices.numel() > 0:
            single_border_grad = flat_grad_output.index_select(0, single_border_destination_indices)
            flat_grad_cubemap.index_add_(0, single_border_source_indices, single_border_grad)
        repeated_border_grad = flat_grad_output.index_select(0, repeated_border_destination_indices.flatten()).reshape(
            repeated_border_destination_indices.shape[0],
            repeated_border_destination_indices.shape[1],
            n_channels,
        )
        repeated_border_grad[:, ctx.repeated_edge_source_count :].div_(_CUBEMAP_CORNER_SOURCE_COUNT)
        flat_grad_cubemap.index_add_(
            0,
            repeated_border_source_indices,
            repeated_border_grad.sum(dim=1),
        )
        return grad_cubemap.reshape(ctx.cubemap_shape), None


def _stash_padding_context(
    ctx: Any,
    cubemap: torch.Tensor,
    topology: _CubemapPaddingTopology,
) -> None:
    """Save the topology fields shared by the ATen and CUDA backward paths."""
    ctx.cubemap_shape = cubemap.shape
    ctx.repeated_edge_source_count = topology.repeated_edge_source_count
    ctx.save_for_backward(
        topology.single_border_source_indices,
        topology.single_border_destination_indices,
        topology.repeated_border_source_indices,
        topology.repeated_border_destination_indices,
    )


class _PadCubemapForSeamlessSamplingCuda(torch.autograd.Function):
    """CUDA fast path for float32 seamless padding with first-order differentiation only."""

    @staticmethod
    def forward(
        ctx,
        cubemap: torch.Tensor,
        topology: _CubemapPaddingTopology,
    ) -> torch.Tensor:
        contiguous_cubemap = cubemap.contiguous()
        padded = background_sampling_cuda.pad_cubemap_forward(
            contiguous_cubemap,
            topology.border_source_indices,
            topology.border_destination_indices,
            topology.corner_destination_indices,
        )

        _stash_padding_context(ctx, cubemap, topology)
        return padded

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        (
            single_border_source_indices,
            single_border_destination_indices,
            repeated_border_source_indices,
            repeated_border_destination_indices,
        ) = ctx.saved_tensors
        grad_cubemap = background_sampling_cuda.pad_cubemap_backward(
            grad_output.contiguous(),
            single_border_source_indices,
            single_border_destination_indices,
            repeated_border_source_indices,
            repeated_border_destination_indices,
            ctx.repeated_edge_source_count,
        )
        return grad_cubemap.reshape(ctx.cubemap_shape), None


def _pad_cubemap_for_seamless_sampling(cubemap: torch.Tensor) -> torch.Tensor:
    """Add neighboring-face texels for seamless bilinear filtering."""
    topology = _cubemap_padding_topology(cubemap.shape[1], cubemap.device)
    if cubemap.is_cuda and cubemap.dtype == torch.float32:
        return _PadCubemapForSeamlessSamplingCuda.apply(cubemap, topology)
    return _PadCubemapForSeamlessSampling.apply(cubemap, topology)


def _prepare_sampling_rays(rays: torch.Tensor, cubemap_size: int) -> tuple[torch.Tensor, float]:
    """Return ``(rays, face_coordinate_scale)`` to pass to the fused cubemap sampler.

    The scale maps face coordinates into the interior of the seamlessly padded faces.
    It is applied here for non-float32 rays and left to the kernel for float32 rays.
    """
    face_coordinate_scale = cubemap_size / (cubemap_size + _SEAMLESS_PADDING_PER_AXIS)
    if rays.dtype == torch.float32:
        return rays.contiguous(), face_coordinate_scale

    # Preserve the established dtype-before-cast behavior for uncommon ray
    # dtypes; casting first can otherwise change the scaled coordinates.
    major_axis = rays.abs().argmax(dim=-1)
    major_axis_mask = torch.nn.functional.one_hot(major_axis, num_classes=_RAY_COMPONENT_COUNT).bool()
    sampling_rays = torch.where(major_axis_mask, rays, rays * face_coordinate_scale).float()
    return sampling_rays, 1.0


def cubemap_coverage_seamless(cubemap_size: int, rays: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return per-texel coverage for seamless cubemap sampling.

    ``rays`` is ``[N, 3]`` and ``weights`` is ``[N]``; the result is ``[6, H, W]``.
    This equals ``autograd.grad(sample_cubemap_seamless(cubemap, rays), cubemap,
    weights[:, None].expand(-1, 3))[0].mean(dim=-1)`` without constructing an
    autograd graph.
    """
    if cubemap_size < 1:
        raise ValueError(f"Cubemap size must be positive, got {cubemap_size}")
    if rays.ndim != 2 or rays.shape[-1] != _RAY_COMPONENT_COUNT:
        raise ValueError(f"Expected rays shape [N, 3], got {tuple(rays.shape)}")
    if weights.ndim != 1 or weights.shape[0] != rays.shape[0]:
        raise ValueError(f"Expected weights shape [{rays.shape[0]}], got {tuple(weights.shape)}")
    if weights.device != rays.device:
        raise RuntimeError(
            f"Rays and weights must be on the same device; got rays on {rays.device} and weights on {weights.device}"
        )
    if not rays.is_cuda:
        raise RuntimeError(f"Cubemap coverage requires CUDA rays, got {rays.device}")
    if rays.shape[0] == 0:
        return torch.zeros((_CUBEMAP_FACE_COUNT, cubemap_size, cubemap_size), device=rays.device)

    sampling_rays, face_coordinate_scale = _prepare_sampling_rays(rays, cubemap_size)
    # The fused VJP is RGB-only; all channels receive the same weight, so keep one.
    grad_out = weights.reshape(-1, 1).expand(-1, _FUSED_SAMPLER_CHANNEL_COUNT).float()
    padded_size = cubemap_size + _SEAMLESS_PADDING_PER_AXIS
    padded_coverage = sample_env_map_cubemap_vjp(
        sampling_rays, grad_out, padded_size, padded_size, face_coordinate_scale
    )
    # Fold the padded borders back onto their source texels (the padding backward),
    # dropping the batch dim and keeping a single channel: [1, 6, Hp, Wp, 3] -> [6, Hp, Wp, 1].
    topology = _cubemap_padding_topology(cubemap_size, rays.device)
    coverage = background_sampling_cuda.pad_cubemap_backward(
        padded_coverage[0, ..., :1].contiguous(),
        topology.single_border_source_indices,
        topology.single_border_destination_indices,
        topology.repeated_border_source_indices,
        topology.repeated_border_destination_indices,
        topology.repeated_edge_source_count,
    )
    return coverage.squeeze(-1)


def sample_cubemap_seamless(cubemap: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    """Sample an OpenGL cubemap with seamless cross-face bilinear filtering.

    Args:
        cubemap: Cubemap faces shaped ``[6, S, S, C]``.
        rays: Direction vectors shaped ``[N, 3]`` on the same CUDA device as ``cubemap``.

    Returns:
        Sampled values shaped ``[N, C]`` with the cubemap's input dtype.

    Raises:
        RuntimeError: If either input is not a CUDA tensor or the inputs are on different devices.
    """
    if cubemap.ndim != 4:
        raise ValueError(f"Expected cubemap shape [6, S, S, C], got {tuple(cubemap.shape)}")
    face_count, cubemap_height, cubemap_width, n_channels = cubemap.shape
    if face_count != _CUBEMAP_FACE_COUNT or cubemap_height != cubemap_width:
        raise ValueError(f"Expected cubemap shape [6, S, S, C], got {tuple(cubemap.shape)}")
    if n_channels < 1:
        raise ValueError("Cubemap must contain at least one channel")
    if rays.ndim != 2 or rays.shape[-1] != _RAY_COMPONENT_COUNT:
        raise ValueError(f"Expected rays shape [N, 3], got {tuple(rays.shape)}")
    if cubemap.device != rays.device:
        raise RuntimeError(
            f"Cubemap and rays must be on the same CUDA device; got cubemap on {cubemap.device} "
            f"and rays on {rays.device}"
        )
    if not cubemap.is_cuda or not rays.is_cuda:
        raise RuntimeError(
            f"Cubemap sampling requires CUDA tensors; got cubemap on {cubemap.device} and rays on {rays.device}"
        )

    cubemap_size = cubemap_height
    if rays.numel() == 0:
        return cubemap.reshape(-1, n_channels)[:0]

    # The fused sampler clamps bilinear taps to the selected face. Add a
    # one-texel border from neighboring faces and contract the face coordinates
    # to preserve seamless cube filtering.
    channel_padded_cubemap = cubemap.float()
    channel_padding = -n_channels % _FUSED_SAMPLER_CHANNEL_COUNT
    if channel_padding:
        channel_padded_cubemap = torch.nn.functional.pad(channel_padded_cubemap, (0, channel_padding))
    seamless_cubemap = _pad_cubemap_for_seamless_sampling(channel_padded_cubemap)
    sampling_rays, sampler_face_coordinate_scale = _prepare_sampling_rays(rays, cubemap_size)

    # The fused sampler processes RGB textures. Sample channels in groups of
    # three so masks and arbitrary feature maps remain supported.
    if seamless_cubemap.shape[-1] == _FUSED_SAMPLER_CHANNEL_COUNT:
        sampled_channels = sample_env_map_cubemap(
            sampling_rays,
            seamless_cubemap[None],
            face_coordinate_scale=sampler_face_coordinate_scale,
        )
    else:
        sampled_channels = torch.cat(
            [
                sample_env_map_cubemap(
                    sampling_rays,
                    seamless_cubemap[..., channel_start : channel_start + _FUSED_SAMPLER_CHANNEL_COUNT][None],
                    face_coordinate_scale=sampler_face_coordinate_scale,
                )
                for channel_start in range(0, seamless_cubemap.shape[-1], _FUSED_SAMPLER_CHANNEL_COUNT)
            ],
            dim=-1,
        )
    return sampled_channels[..., :n_channels].to(cubemap.dtype)
