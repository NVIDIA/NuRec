# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import cv2
import numpy as np
import pytest
import torch

from nurec.models import background_sampling
from nurec.models.background_sampling import (
    _SEAMLESS_PADDING_PER_AXIS,
    _cubemap_padding_topology,
    _pad_cubemap_for_seamless_sampling,
    _PadCubemapForSeamlessSampling,
    _PadCubemapForSeamlessSamplingCuda,
    cubemap_coverage_seamless,
    sample_cubemap_seamless,
)
from nurec.models.background_sampling_cuda import background_sampling_cuda


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cubemap sampling requires CUDA")


_CUBEMAP_FACE_COUNT = 6
_CUBEMAP_CORNERS_PER_FACE = 4
_CUBEMAP_CORNER_COUNT = _CUBEMAP_FACE_COUNT * _CUBEMAP_CORNERS_PER_FACE
_CUBEMAP_CORNER_SOURCE_COUNT = 3
_CUBEMAP_EDGES_PER_FACE = 4
_CUBEMAP_EDGE_TEXELS_PER_SIZE = _CUBEMAP_FACE_COUNT * _CUBEMAP_EDGES_PER_FACE
_RAY_COMPONENT_COUNT = 3

_CUBEMAP_LAYOUT_ROWS = 2
_CUBEMAP_LAYOUT_COLUMNS = 3
_INPAINT_THRESHOLD = 5e-2
_INPAINT_KERNEL_SIZE = 10

_DTYPE_TOLERANCES = [
    # dtype, forward tolerance, texture-gradient tolerance
    (torch.float32, 2e-5, 2e-5),
    (torch.float16, 5e-3, 1e-2),
    (torch.bfloat16, 5e-2, 5e-2),
]

_PADDING_TEST_CONFIGURATIONS = [
    pytest.param(torch.float32, 2e-5, False, id="float32-cuda-dispatch"),
    pytest.param(torch.float32, 2e-5, True, id="float32-aten-reference"),
    pytest.param(torch.float16, 1e-2, False, id="float16-aten-dispatch"),
    pytest.param(torch.bfloat16, 5e-2, False, id="bfloat16-aten-dispatch"),
]


def _dominant_axis_to_face_uv(rays: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return OpenGL cubemap face indices and normalized face coordinates."""
    x, y, z = rays.unbind(-1)
    ax, ay, az = x.abs(), y.abs(), z.abs()
    face = torch.zeros(rays.shape[0], dtype=torch.long, device=rays.device)
    sc = torch.zeros_like(x)
    tc = torch.zeros_like(x)
    ma = torch.ones_like(x)

    x_major = (ax >= ay) & (ax >= az)
    y_major = (~x_major) & (ay >= az)
    z_major = (~x_major) & (~y_major)
    masks = [
        x_major & (x > 0),
        x_major & (x <= 0),
        y_major & (y > 0),
        y_major & (y <= 0),
        z_major & (z > 0),
        z_major & (z <= 0),
    ]
    coordinates = [
        (-z, -y, ax),
        (z, -y, ax),
        (x, z, ay),
        (x, -z, ay),
        (x, -y, az),
        (-x, -y, az),
    ]
    for face_index, (mask, (face_s, face_t, face_ma)) in enumerate(zip(masks, coordinates)):
        face[mask] = face_index
        sc[mask] = face_s[mask]
        tc[mask] = face_t[mask]
        ma[mask] = face_ma[mask]
    return face, sc / ma, tc / ma


def _reference_border(cubemap: torch.Tensor) -> torch.Tensor:
    """Build the reference sampler's one-texel OpenGL face border."""
    top_edges = [
        (2, "column", -1, True),
        (2, "column", 0, False),
        (5, "row", 0, True),
        (4, "row", -1, False),
        (2, "row", -1, False),
        (2, "row", 0, True),
    ]
    right_edges = [
        (5, "column", 0, False),
        (4, "column", 0, False),
        (0, "row", 0, True),
        (0, "row", -1, False),
        (0, "column", 0, False),
        (1, "column", 0, False),
    ]
    bottom_edges = [
        (3, "column", -1, False),
        (3, "column", 0, True),
        (4, "row", 0, False),
        (5, "row", -1, True),
        (3, "row", 0, False),
        (3, "row", -1, True),
    ]
    left_edges = [
        (4, "column", -1, False),
        (5, "column", -1, False),
        (1, "row", 0, False),
        (1, "row", -1, True),
        (1, "column", -1, False),
        (0, "column", -1, False),
    ]

    def edge_values(entries: list[tuple[int, str, int, bool]]) -> torch.Tensor:
        values = []
        for face, axis, index, flip in entries:
            value = cubemap[face, index] if axis == "row" else cubemap[face, :, index]
            values.append(value.flip(0) if flip else value)
        return torch.stack(values)

    top = edge_values(top_edges)
    right = edge_values(right_edges)
    bottom = edge_values(bottom_edges)
    left = edge_values(left_edges)
    corners = [
        (top[:, 0] + left[:, 0] + cubemap[:, 0, 0]) / 3,
        (top[:, -1] + right[:, 0] + cubemap[:, 0, -1]) / 3,
        (bottom[:, 0] + left[:, -1] + cubemap[:, -1, 0]) / 3,
        (bottom[:, -1] + right[:, -1] + cubemap[:, -1, -1]) / 3,
    ]
    top = torch.cat([corners[0][:, None], top, corners[1][:, None]], dim=1)
    middle = torch.cat([left[:, :, None], cubemap, right[:, :, None]], dim=2)
    bottom = torch.cat([corners[2][:, None], bottom, corners[3][:, None]], dim=1)
    return torch.cat([top[:, None], middle, bottom[:, None]], dim=1)


def _sample_reference(cubemap: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    face, u, v = _dominant_axis_to_face_uv(rays)
    scale = cubemap.shape[1] / (cubemap.shape[1] + 2)
    grid = torch.stack([u * scale, v * scale], dim=-1)
    padded = _reference_border(cubemap.float())
    sampled = torch.empty((rays.shape[0], cubemap.shape[-1]), device=rays.device)
    for face_index in range(_CUBEMAP_FACE_COUNT):
        mask = face == face_index
        face_texture = padded[face_index].permute(2, 0, 1)[None]
        values = torch.nn.functional.grid_sample(
            face_texture,
            grid[mask][None, None],
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        sampled[mask] = values.squeeze(0).squeeze(1).T
    return sampled.to(cubemap.dtype)


def _face_ray(face: int, u: float, v: float, device: torch.device) -> torch.Tensor:
    rays = [
        [1.0, -v, -u],
        [-1.0, -v, u],
        [u, 1.0, v],
        [u, -1.0, -v],
        [u, -v, 1.0],
        [-u, -v, -1.0],
    ]
    return torch.tensor(rays[face], device=device)


def _edge_and_corner_rays(size: int, device: torch.device) -> torch.Tensor:
    edge = 1 - 0.5 / size
    rays = []
    for face in range(_CUBEMAP_FACE_COUNT):
        rays.extend(
            [
                _face_ray(face, -edge, 0.37, device),
                _face_ray(face, edge, 0.37, device),
                _face_ray(face, -0.29, -edge, device),
                _face_ray(face, -0.29, edge, device),
            ]
        )
        for u in (-edge, edge):
            for v in (-edge, edge):
                rays.append(_face_ray(face, u, v, device))
    return torch.stack(rays)


def _encoded_cubemap(
    size: int,
    n_channels: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
    requires_grad: bool = False,
) -> torch.Tensor:
    values = torch.arange(
        _CUBEMAP_FACE_COUNT * size * size * n_channels,
        device=device,
        dtype=torch.float32,
    )
    return (
        (values / values.numel())
        .reshape(_CUBEMAP_FACE_COUNT, size, size, n_channels)
        .to(dtype)
        .requires_grad_(requires_grad)
    )


def _final_inpainting_mask(texture_gradients: torch.Tensor) -> torch.Tensor:
    """Apply SkyEnvMapBackground's threshold, layout, and dilation."""
    cubemap_size = texture_gradients.shape[1]
    flattened = (
        texture_gradients.reshape(
            _CUBEMAP_LAYOUT_ROWS,
            _CUBEMAP_LAYOUT_COLUMNS,
            cubemap_size,
            cubemap_size,
        )
        .permute(0, 2, 1, 3)
        .reshape(_CUBEMAP_LAYOUT_ROWS * cubemap_size, _CUBEMAP_LAYOUT_COLUMNS * cubemap_size)
    )
    thresholded = (flattened < _INPAINT_THRESHOLD).byte().cpu().numpy()
    kernel = np.ones((_INPAINT_KERNEL_SIZE, _INPAINT_KERNEL_SIZE), np.uint8)
    return torch.from_numpy(cv2.dilate(thresholded, kernel, iterations=1)).bool()


def _apply_padding_for_test(cubemap: torch.Tensor, use_aten_reference: bool) -> torch.Tensor:
    if not use_aten_reference:
        return _pad_cubemap_for_seamless_sampling(cubemap)
    topology = _cubemap_padding_topology(cubemap.shape[1], cubemap.device)
    return _PadCubemapForSeamlessSampling.apply(cubemap, topology)


@pytest.mark.parametrize("size", [1, 7])
@pytest.mark.parametrize("n_channels", [1, 3, 5])
@pytest.mark.parametrize(
    ("dtype", "gradient_tolerance", "use_aten_reference"),
    _PADDING_TEST_CONFIGURATIONS,
)
def test_padding_matches_legacy_output_and_gradients(
    size: int,
    n_channels: int,
    dtype: torch.dtype,
    gradient_tolerance: float,
    use_aten_reference: bool,
) -> None:
    torch.manual_seed(29)
    actual_cubemap = torch.randn(
        (_CUBEMAP_FACE_COUNT, size, size, n_channels),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    expected_cubemap = actual_cubemap.detach().clone().requires_grad_(True)

    actual = _apply_padding_for_test(actual_cubemap, use_aten_reference)
    expected = _reference_border(expected_cubemap)

    assert actual.dtype == dtype
    assert actual.is_contiguous()
    # The CUDA float32 implementation deliberately matches the ATen
    # reference's left-associated corner sum and reciprocal multiplication.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    # SkyEnvMapBackground may traverse this graph once for radiance statistics
    # and again for the training loss. Exercise both backwards on one forward.
    tracking_weights = torch.randn_like(actual)
    training_weights = torch.randn_like(actual)
    actual_tracking_grad = torch.autograd.grad(actual, actual_cubemap, tracking_weights, retain_graph=True)[0]
    expected_tracking_grad = torch.autograd.grad(expected, expected_cubemap, tracking_weights, retain_graph=True)[0]
    actual_training_grad = torch.autograd.grad(actual, actual_cubemap, training_weights)[0]
    expected_training_grad = torch.autograd.grad(expected, expected_cubemap, training_weights)[0]

    torch.testing.assert_close(
        actual_tracking_grad,
        expected_tracking_grad,
        atol=gradient_tolerance,
        rtol=gradient_tolerance,
    )
    torch.testing.assert_close(
        actual_training_grad,
        expected_training_grad,
        atol=gradient_tolerance,
        rtol=gradient_tolerance,
    )


@pytest.mark.parametrize("size", [1, 7])
@pytest.mark.parametrize("n_channels", [1, 3, 5])
@pytest.mark.parametrize(
    ("dtype", "gradient_tolerance", "use_aten_reference"),
    _PADDING_TEST_CONFIGURATIONS,
)
def test_padding_backward_is_bitwise_repeatable_for_accumulated_corners(
    size: int,
    n_channels: int,
    dtype: torch.dtype,
    gradient_tolerance: float,
    use_aten_reference: bool,
) -> None:
    torch.manual_seed(47 + size + n_channels)
    actual_cubemap = torch.randn(
        (_CUBEMAP_FACE_COUNT, size, size, n_channels),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    expected_cubemap = actual_cubemap.detach().clone().requires_grad_(True)
    actual = _apply_padding_for_test(actual_cubemap, use_aten_reference)
    expected = _reference_border(expected_cubemap)
    upstream = torch.randn_like(actual)

    actual_gradient = torch.autograd.grad(actual, actual_cubemap, upstream, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, expected_cubemap, upstream)[0]

    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        atol=gradient_tolerance,
        rtol=gradient_tolerance,
    )
    for _ in range(20):
        repeated_gradient = torch.autograd.grad(actual, actual_cubemap, upstream, retain_graph=True)[0]
        torch.testing.assert_close(repeated_gradient, actual_gradient, atol=0, rtol=0)


@pytest.mark.parametrize("size", [1, 7])
def test_each_padded_corner_distributes_its_gradient_to_all_three_sources(size: int) -> None:
    actual_cubemap = torch.randn(
        (_CUBEMAP_FACE_COUNT, size, size, 1),
        device="cuda",
        requires_grad=True,
    )
    expected_cubemap = actual_cubemap.detach().clone().requires_grad_(True)
    actual = _pad_cubemap_for_seamless_sampling(actual_cubemap)
    expected = _reference_border(expected_cubemap)
    corner_destinations = _cubemap_padding_topology(size, actual_cubemap.device).corner_destination_indices.tolist()

    assert len(corner_destinations) == _CUBEMAP_CORNER_COUNT
    for corner_index, corner_destination in enumerate(corner_destinations):
        upstream = torch.zeros_like(actual)
        upstream.reshape(-1)[corner_destination] = 1
        retain_graph = corner_index + 1 < _CUBEMAP_CORNER_COUNT

        actual_gradient = torch.autograd.grad(
            actual,
            actual_cubemap,
            upstream,
            retain_graph=retain_graph,
        )[0]
        expected_gradient = torch.autograd.grad(
            expected,
            expected_cubemap,
            upstream,
            retain_graph=retain_graph,
        )[0]

        torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)
        torch.testing.assert_close(actual_gradient.sum(), torch.ones((), device="cuda"), atol=2e-7, rtol=0)


def test_padding_topology_cache_reuses_indices_for_size_and_device() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    cubemap_size = 11

    first = _cubemap_padding_topology(cubemap_size, device)
    second = _cubemap_padding_topology(cubemap_size, device)
    other_size = _cubemap_padding_topology(cubemap_size + 1, device)

    assert first is second
    assert first is not other_size
    assert first.border_source_indices.device == device
    edge_texel_count = _CUBEMAP_EDGE_TEXELS_PER_SIZE * cubemap_size
    assert first.border_source_indices.numel() == (
        edge_texel_count + _CUBEMAP_CORNER_COUNT * _CUBEMAP_CORNER_SOURCE_COUNT
    )
    assert first.border_destination_indices[:edge_texel_count].unique().numel() == edge_texel_count
    assert first.corner_destination_indices.unique().numel() == _CUBEMAP_CORNER_COUNT
    assert first.border_destination_indices[edge_texel_count:].reshape(
        _CUBEMAP_CORNER_COUNT, _CUBEMAP_CORNER_SOURCE_COUNT
    ).unique(dim=1).shape == (_CUBEMAP_CORNER_COUNT, 1)
    assert first.border_source_indices.min() >= 0
    assert first.border_source_indices.max() < _CUBEMAP_FACE_COUNT * cubemap_size * cubemap_size
    assert first.single_border_source_indices.unique().numel() == first.single_border_source_indices.numel()
    assert first.repeated_border_source_indices.unique().numel() == first.repeated_border_source_indices.numel()
    backward_source_indices = torch.cat([first.single_border_source_indices, first.repeated_border_source_indices])
    assert backward_source_indices.unique().numel() == backward_source_indices.numel()
    assert first.single_border_destination_indices.numel() + first.repeated_border_destination_indices.numel() == (
        first.border_source_indices.numel()
    )


def test_padding_dispatches_float32_to_cuda_extension() -> None:
    with (
        mock.patch.object(
            background_sampling_cuda,
            "pad_cubemap_forward",
            wraps=background_sampling_cuda.pad_cubemap_forward,
        ) as forward,
        mock.patch.object(
            background_sampling_cuda,
            "pad_cubemap_backward",
            wraps=background_sampling_cuda.pad_cubemap_backward,
        ) as backward,
    ):
        float_cubemap = _encoded_cubemap(7, 3, requires_grad=True)
        _pad_cubemap_for_seamless_sampling(float_cubemap).sum().backward()
        forward.assert_called_once()
        backward.assert_called_once()

        forward.reset_mock()
        backward.reset_mock()
        half_cubemap = _encoded_cubemap(7, 3, dtype=torch.float16, requires_grad=True)
        _pad_cubemap_for_seamless_sampling(half_cubemap).sum().backward()
        forward.assert_not_called()
        backward.assert_not_called()


def test_cuda_padding_backward_rejects_topology_for_different_cubemap_size() -> None:
    grad_output_cubemap_size = 8
    mismatched_topology = _cubemap_padding_topology(16, torch.device("cuda"))
    grad_output = torch.zeros(
        (
            _CUBEMAP_FACE_COUNT,
            grad_output_cubemap_size + _SEAMLESS_PADDING_PER_AXIS,
            grad_output_cubemap_size + _SEAMLESS_PADDING_PER_AXIS,
            3,
        ),
        device="cuda",
    )

    with pytest.raises(RuntimeError, match="backward topology size does not match grad_output cubemap size"):
        background_sampling_cuda.pad_cubemap_backward(
            grad_output=grad_output,
            single_border_source_indices=mismatched_topology.single_border_source_indices,
            single_border_destination_indices=mismatched_topology.single_border_destination_indices,
            repeated_border_source_indices=mismatched_topology.repeated_border_source_indices,
            repeated_border_destination_indices=mismatched_topology.repeated_border_destination_indices,
            repeated_edge_source_count=mismatched_topology.repeated_edge_source_count,
        )


def test_cuda_padding_supports_noncontiguous_input_and_grad_output() -> None:
    torch.manual_seed(53)
    cubemap_size = 7
    n_channels = 3
    actual_cubemap = torch.randn(
        (cubemap_size, _CUBEMAP_FACE_COUNT, cubemap_size, n_channels),
        device="cuda",
    ).permute(1, 0, 2, 3)
    actual_cubemap.requires_grad_(True)
    expected_cubemap = actual_cubemap.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    topology = _cubemap_padding_topology(cubemap_size, actual_cubemap.device)
    actual = _PadCubemapForSeamlessSamplingCuda.apply(actual_cubemap, topology)
    expected = _PadCubemapForSeamlessSampling.apply(expected_cubemap, topology)
    grad_output = torch.randn(
        (
            cubemap_size + _SEAMLESS_PADDING_PER_AXIS,
            _CUBEMAP_FACE_COUNT,
            cubemap_size + _SEAMLESS_PADDING_PER_AXIS,
            n_channels,
        ),
        device="cuda",
    ).permute(1, 0, 2, 3)

    assert not actual_cubemap.is_contiguous()
    assert not grad_output.is_contiguous()
    actual_gradient = torch.autograd.grad(actual, actual_cubemap, grad_output)[0]
    expected_gradient = torch.autograd.grad(expected, expected_cubemap, grad_output)[0]

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=2e-5)


def test_cuda_padding_is_explicitly_first_order_only() -> None:
    cubemap = _encoded_cubemap(7, 3, requires_grad=True)
    padded = _pad_cubemap_for_seamless_sampling(cubemap)
    upstream = torch.ones_like(padded, requires_grad=True)
    first_gradient = torch.autograd.grad(
        padded,
        cubemap,
        upstream,
        create_graph=True,
    )[0]

    with pytest.raises(RuntimeError, match="once_differentiable|differentiate twice"):
        first_gradient.sum().backward()


def test_sampler_passes_float32_coordinate_scale_to_every_channel_group() -> None:
    cubemap = _encoded_cubemap(7, 5)
    contiguous_float_rays = _edge_and_corner_rays(7, cubemap.device)
    noncontiguous_float_rays = torch.randn(
        (_RAY_COMPONENT_COUNT, contiguous_float_rays.shape[0]),
        device=cubemap.device,
    ).T
    assert not noncontiguous_float_rays.is_contiguous()
    face_coordinate_scale = 7 / 9

    with mock.patch.object(
        background_sampling,
        "sample_env_map_cubemap",
        wraps=background_sampling.sample_env_map_cubemap,
    ) as sampler:
        sample_cubemap_seamless(cubemap, contiguous_float_rays)
        assert sampler.call_count == 2
        first_group_rays = sampler.call_args_list[0].args[0]
        assert first_group_rays is sampler.call_args_list[1].args[0]
        assert first_group_rays.is_contiguous()
        torch.testing.assert_close(first_group_rays, contiguous_float_rays, atol=0, rtol=0)
        assert all(call.kwargs["face_coordinate_scale"] == face_coordinate_scale for call in sampler.call_args_list)

        sample_cubemap_seamless(cubemap, noncontiguous_float_rays)
        assert sampler.call_count == 4
        second_group_rays = sampler.call_args_list[2].args[0]
        assert second_group_rays is sampler.call_args_list[3].args[0]
        assert second_group_rays.is_contiguous()
        torch.testing.assert_close(second_group_rays, noncontiguous_float_rays, atol=0, rtol=0)
        assert all(call.kwargs["face_coordinate_scale"] == face_coordinate_scale for call in sampler.call_args_list[2:])


@pytest.mark.parametrize("ray_dtype", [torch.float16, torch.bfloat16])
def test_pinned_gsplat_scale_matches_uncommon_dtype_dispatch_at_routing_boundaries(
    ray_dtype: torch.dtype,
) -> None:
    """Lock NuRec's pinned gsplat scale contract across its two dtype paths."""
    torch.manual_seed(43)
    # 14 / (14 + 2) == 7 / 8 is exact in float32, float16, and bfloat16,
    # keeping this a dispatch-contract test rather than a rounding comparison.
    cubemap_size = 14
    signs = (-1.0, 1.0)
    axes = torch.cat(
        [
            torch.eye(_RAY_COMPONENT_COUNT, device="cuda"),
            -torch.eye(_RAY_COMPONENT_COUNT, device="cuda"),
        ]
    )
    edge_ties = torch.tensor(
        [[x, y, 0.0] for x in signs for y in signs]
        + [[x, 0.0, z] for x in signs for z in signs]
        + [[0.0, y, z] for y in signs for z in signs],
        device="cuda",
    )
    corner_ties = torch.tensor(
        [[x, y, z] for x in signs for y in signs for z in signs],
        device="cuda",
    )
    rays = torch.cat([axes, edge_ties, corner_ties])
    fused_rays = rays.detach().clone().requires_grad_(True)
    uncommon_dtype_rays = rays.to(ray_dtype).requires_grad_(True)

    fused_cubemap = torch.rand(
        (_CUBEMAP_FACE_COUNT, cubemap_size, cubemap_size, 3),
        device="cuda",
        requires_grad=True,
    )
    uncommon_dtype_cubemap = fused_cubemap.detach().clone().requires_grad_(True)
    gradient_weights = torch.randn((rays.shape[0], 3), device="cuda")

    fused = sample_cubemap_seamless(fused_cubemap, fused_rays)
    uncommon_dtype = sample_cubemap_seamless(uncommon_dtype_cubemap, uncommon_dtype_rays)
    fused_gradient, fused_ray_gradient = torch.autograd.grad(
        fused,
        (fused_cubemap, fused_rays),
        gradient_weights,
        allow_unused=True,
    )
    uncommon_dtype_gradient, uncommon_dtype_ray_gradient = torch.autograd.grad(
        uncommon_dtype,
        (uncommon_dtype_cubemap, uncommon_dtype_rays),
        gradient_weights,
        allow_unused=True,
    )

    torch.testing.assert_close(fused, uncommon_dtype, atol=0, rtol=0)
    torch.testing.assert_close(fused_gradient, uncommon_dtype_gradient, atol=2e-5, rtol=2e-5)
    assert fused_ray_gradient is None
    assert uncommon_dtype_ray_gradient is None


@pytest.mark.parametrize("ray_dtype", [torch.float16, torch.bfloat16])
def test_sampler_scales_uncommon_ray_dtypes_before_float32_cast(ray_dtype: torch.dtype) -> None:
    cubemap_size = 7
    cubemap = _encoded_cubemap(cubemap_size, 3)
    rays = _edge_and_corner_rays(cubemap_size, cubemap.device).to(ray_dtype)
    face_coordinate_scale = cubemap_size / (cubemap_size + _SEAMLESS_PADDING_PER_AXIS)
    major_axis = rays.abs().argmax(dim=-1)
    major_axis_mask = torch.nn.functional.one_hot(major_axis, num_classes=_RAY_COMPONENT_COUNT).bool()
    expected_sampling_rays = torch.where(major_axis_mask, rays, rays * face_coordinate_scale).float()

    with mock.patch.object(
        background_sampling,
        "sample_env_map_cubemap",
        wraps=background_sampling.sample_env_map_cubemap,
    ) as sampler:
        sample_cubemap_seamless(cubemap, rays)

    sampler.assert_called_once()
    sampling_rays = sampler.call_args.args[0]
    assert sampling_rays.dtype == torch.float32
    assert sampling_rays.is_contiguous()
    assert sampler.call_args.kwargs["face_coordinate_scale"] == 1.0
    torch.testing.assert_close(sampling_rays, expected_sampling_rays, atol=0, rtol=0)


@pytest.mark.parametrize("n_channels", [1, 3, 5])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(dtype, forward_tolerance) for dtype, forward_tolerance, _ in _DTYPE_TOLERANCES],
)
def test_sample_cubemap_matches_grid_sample_reference_at_every_face_edge_and_corner(
    n_channels: int,
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    cubemap = _encoded_cubemap(32, n_channels, dtype=dtype)
    rays = _edge_and_corner_rays(32, cubemap.device)

    actual = sample_cubemap_seamless(cubemap, rays)
    expected = _sample_reference(cubemap, rays)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_sample_cubemap_blends_known_positive_x_negative_z_seam() -> None:
    cubemap = torch.zeros((6, 4, 4, 1), device="cuda")
    cubemap[0] = 10.0  # +X face
    cubemap[5] = 30.0  # -Z face
    seam_ray = torch.tensor([[1.0, 0.0, -1.0]], device="cuda")

    actual = sample_cubemap_seamless(cubemap, seam_ray)

    # The ray lies exactly on the +X/-Z boundary, so seamless bilinear
    # filtering must blend the two constant faces equally.
    torch.testing.assert_close(actual, torch.tensor([[20.0]], device="cuda"), atol=2e-6, rtol=0)


@pytest.mark.parametrize("n_channels", [1, 3])
def test_sample_cubemap_matches_reference_for_discontinuous_texture(n_channels: int) -> None:
    torch.manual_seed(17)
    cubemap = torch.rand((6, 256, 256, n_channels), device="cuda")
    random_rays = torch.nn.functional.normalize(torch.randn((4096, 3), device="cuda"), dim=-1)
    rays = torch.cat([random_rays, _edge_and_corner_rays(256, cubemap.device)])

    actual = sample_cubemap_seamless(cubemap, rays)
    expected = _sample_reference(cubemap, rays)

    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=2e-5)


@pytest.mark.parametrize("n_channels", [1, 3, 5])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(dtype, gradient_tolerance) for dtype, _, gradient_tolerance in _DTYPE_TOLERANCES],
)
def test_sample_cubemap_preserves_texture_gradients(
    n_channels: int,
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    torch.manual_seed(23)
    rays = torch.nn.functional.normalize(
        torch.randn((512, _RAY_COMPONENT_COUNT), device="cuda"),
        dim=-1,
    )
    rays = torch.cat([rays, _edge_and_corner_rays(32, rays.device)])
    weights = torch.randn((rays.shape[0], n_channels), device="cuda", dtype=dtype)
    actual_cubemap = _encoded_cubemap(32, n_channels, dtype=dtype, requires_grad=True)
    expected_cubemap = actual_cubemap.detach().clone().requires_grad_(True)

    (sample_cubemap_seamless(actual_cubemap, rays) * weights).sum().backward()
    (_sample_reference(expected_cubemap, rays) * weights).sum().backward()

    torch.testing.assert_close(actual_cubemap.grad, expected_cubemap.grad, atol=tolerance, rtol=tolerance)


def test_sample_cubemap_supports_two_backward_traversals_from_one_retained_graph() -> None:
    torch.manual_seed(31)
    rays = torch.nn.functional.normalize(
        torch.randn((512, _RAY_COMPONENT_COUNT), device="cuda"),
        dim=-1,
    )
    rays = torch.cat([rays, _edge_and_corner_rays(32, rays.device)])
    actual_cubemap = _encoded_cubemap(32, 3, requires_grad=True)
    expected_cubemap = actual_cubemap.detach().clone().requires_grad_(True)
    actual = sample_cubemap_seamless(actual_cubemap, rays)
    expected = _sample_reference(expected_cubemap, rays)
    tracking_weights = torch.rand_like(actual)
    training_weights = torch.randn_like(actual)

    actual_tracking_gradient = torch.autograd.grad(
        actual,
        actual_cubemap,
        tracking_weights,
        retain_graph=True,
    )[0]
    expected_tracking_gradient = torch.autograd.grad(
        expected,
        expected_cubemap,
        tracking_weights,
        retain_graph=True,
    )[0]
    actual_training_gradient = torch.autograd.grad(actual, actual_cubemap, training_weights)[0]
    expected_training_gradient = torch.autograd.grad(expected, expected_cubemap, training_weights)[0]

    torch.testing.assert_close(actual_tracking_gradient, expected_tracking_gradient, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_training_gradient, expected_training_gradient, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("n_channels", [3, 5])
def test_sample_cubemap_supports_noncontiguous_cubemap_input(n_channels: int) -> None:
    torch.manual_seed(37)
    size = 32
    actual_cubemap = torch.rand(
        (size, _CUBEMAP_FACE_COUNT, size, n_channels),
        device="cuda",
    ).permute(1, 0, 2, 3)
    actual_cubemap.requires_grad_(True)
    expected_cubemap = actual_cubemap.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    rays = torch.nn.functional.normalize(
        torch.randn((512, _RAY_COMPONENT_COUNT), device="cuda"),
        dim=-1,
    )
    weights = torch.randn((rays.shape[0], n_channels), device="cuda")

    assert actual_cubemap.is_leaf
    assert not actual_cubemap.is_contiguous()
    actual = sample_cubemap_seamless(actual_cubemap, rays)
    expected = _sample_reference(expected_cubemap, rays)
    actual_gradient = torch.autograd.grad(actual, actual_cubemap, weights)[0]
    expected_gradient = torch.autograd.grad(expected, expected_cubemap, weights)[0]

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=2e-5)


def test_sample_cubemap_supports_empty_rays_and_zero_texture_gradient() -> None:
    cubemap = _encoded_cubemap(16, 5, dtype=torch.float16, requires_grad=True)
    rays = torch.empty((0, _RAY_COMPONENT_COUNT), device=cubemap.device, requires_grad=True)

    sampled = sample_cubemap_seamless(cubemap, rays)
    sampled.sum().backward()

    assert sampled.shape == (0, 5)
    assert sampled.dtype == cubemap.dtype
    assert sampled.device == cubemap.device
    assert cubemap.grad is not None
    assert torch.count_nonzero(cubemap.grad) == 0
    assert rays.grad is None


def test_padding_topology_cache_does_not_cache_cubemap_values() -> None:
    cubemap = _encoded_cubemap(16, 3)
    rays = torch.nn.functional.normalize(
        torch.randn((128, _RAY_COMPONENT_COUNT), device=cubemap.device),
        dim=-1,
    )

    before_update = sample_cubemap_seamless(cubemap, rays)
    with torch.no_grad():
        cubemap.add_(1)
    after_update = sample_cubemap_seamless(cubemap, rays)

    torch.testing.assert_close(after_update, before_update + 1, atol=2e-6, rtol=2e-6)


def test_sample_cubemap_preserves_final_gradient_based_inpainting_mask() -> None:
    torch.manual_seed(41)
    cubemap_size = 32
    n_channels = 3
    rays = torch.nn.functional.normalize(
        torch.randn((4096, _RAY_COMPONENT_COUNT), device="cuda"),
        dim=-1,
    )
    rays = torch.cat([rays, _edge_and_corner_rays(cubemap_size, rays.device)])
    opacity = torch.linspace(0.0, 0.9, rays.shape[0], device=rays.device)
    gradient_weights = (1 - opacity[:, None]).expand(-1, n_channels)
    actual_cubemap = _encoded_cubemap(cubemap_size, n_channels, requires_grad=True)
    expected_cubemap = actual_cubemap.detach().clone().requires_grad_(True)

    actual_gradient = torch.autograd.grad(
        sample_cubemap_seamless(actual_cubemap, rays),
        actual_cubemap,
        gradient_weights,
    )[0]
    expected_gradient = torch.autograd.grad(
        _sample_reference(expected_cubemap, rays),
        expected_cubemap,
        gradient_weights,
    )[0]
    actual_mask = _final_inpainting_mask(actual_gradient.mean(dim=-1))
    expected_mask = _final_inpainting_mask(expected_gradient.mean(dim=-1))

    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_mask, expected_mask, atol=0, rtol=0)


@pytest.mark.parametrize("cubemap_size", [7, 32])
@pytest.mark.parametrize("ray_dtype", [torch.float32, torch.float16])
def test_cubemap_coverage_matches_autograd_reference(cubemap_size: int, ray_dtype: torch.dtype) -> None:
    torch.manual_seed(47)
    random_rays = torch.nn.functional.normalize(torch.randn(257, 3, device="cuda"), dim=-1)
    rays = torch.cat([random_rays, _edge_and_corner_rays(cubemap_size, random_rays.device)]).to(ray_dtype)
    opacity = torch.linspace(0.0, 0.9, rays.shape[0], device=rays.device)
    weights = 1 - opacity

    cubemap = torch.zeros(6, cubemap_size, cubemap_size, 3, device="cuda", requires_grad=True)
    reference = torch.autograd.grad(
        sample_cubemap_seamless(cubemap, rays),
        cubemap,
        weights[:, None].expand(-1, 3),
    )[0].mean(dim=-1)
    actual = cubemap_coverage_seamless(cubemap_size, rays, weights)

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        _final_inpainting_mask(actual),
        _final_inpainting_mask(reference),
        atol=0,
        rtol=0,
    )


def test_cubemap_coverage_supports_empty_rays() -> None:
    actual = cubemap_coverage_seamless(7, torch.empty(0, 3, device="cuda"), torch.empty(0, device="cuda"))

    assert actual.shape == (6, 7, 7)
    assert torch.count_nonzero(actual) == 0


def test_cubemap_coverage_validates_inputs_before_empty_ray_fast_path() -> None:
    rays = torch.empty(0, 3, device="cuda")

    with pytest.raises(ValueError, match="Cubemap size must be positive"):
        cubemap_coverage_seamless(0, rays, torch.empty(0, device="cuda"))
    with pytest.raises(ValueError, match=r"Expected weights shape \[0\]"):
        cubemap_coverage_seamless(7, rays, torch.ones(1, device="cuda"))
    with pytest.raises(ValueError, match=r"Expected weights shape \[0\]"):
        cubemap_coverage_seamless(7, rays, torch.empty(0, 1, device="cuda"))
    with pytest.raises(RuntimeError, match="Rays and weights must be on the same device"):
        cubemap_coverage_seamless(7, rays, torch.empty(0))


def test_cubemap_coverage_rejects_cpu_rays() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA rays"):
        cubemap_coverage_seamless(7, torch.empty(1, 3), torch.ones(1))


def test_sample_cubemap_preserves_fused_sampler_ray_gradient_behavior() -> None:
    cubemap = _encoded_cubemap(16, 3, requires_grad=True)
    rays = torch.randn((32, 3), device="cuda", requires_grad=True)

    ray_grad = torch.autograd.grad(sample_cubemap_seamless(cubemap, rays).sum(), rays, allow_unused=True)[0]

    # gsplat's fused cubemap backward intentionally returns no ray gradient.
    assert ray_grad is None


@pytest.mark.parametrize(
    ("cubemap", "rays", "message"),
    [
        (torch.empty(5, 4, 4, 3), torch.empty(1, 3), "cubemap shape"),
        (torch.empty(6, 4, 5, 3), torch.empty(1, 3), "cubemap shape"),
        (torch.empty(6, 4, 4, 0), torch.empty(1, 3), "at least one channel"),
        (torch.empty(6, 4, 4, 3), torch.empty(1, 4), "rays shape"),
    ],
)
def test_sample_cubemap_rejects_invalid_shapes(cubemap: torch.Tensor, rays: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_cubemap_seamless(cubemap, rays)


@pytest.mark.parametrize(
    ("cpu_input", "expected_message"),
    [
        (
            "cubemap",
            "Cubemap and rays must be on the same CUDA device; got cubemap on cpu and rays on cuda:0",
        ),
        (
            "rays",
            "Cubemap and rays must be on the same CUDA device; got cubemap on cuda:0 and rays on cpu",
        ),
    ],
)
def test_sample_cubemap_rejects_mixed_cpu_and_cuda_devices(cpu_input: str, expected_message: str) -> None:
    cubemap = torch.empty(_CUBEMAP_FACE_COUNT, 4, 4, 3, device="cuda")
    rays = torch.empty(1, _RAY_COMPONENT_COUNT, device="cuda")
    if cpu_input == "cubemap":
        cubemap = cubemap.cpu()
    else:
        rays = rays.cpu()

    with pytest.raises(RuntimeError) as error:
        sample_cubemap_seamless(cubemap, rays)

    assert str(error.value) == expected_message


def test_sample_cubemap_rejects_cpu_tensors_on_the_same_device() -> None:
    cubemap = torch.empty(_CUBEMAP_FACE_COUNT, 4, 4, 3)
    rays = torch.empty(1, _RAY_COMPONENT_COUNT)

    with pytest.raises(RuntimeError) as error:
        sample_cubemap_seamless(cubemap, rays)

    assert str(error.value) == "Cubemap sampling requires CUDA tensors; got cubemap on cpu and rays on cpu"
