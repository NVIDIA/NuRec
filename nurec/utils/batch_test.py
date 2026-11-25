# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from nurec.utils.batch import _pixel_ray_angle, compute_pixel_footprint


def _reference_footprint(rays, mode="cone", offset=10):
    """Independent float64 reference, including both replicated image borders."""
    rays = rays.detach().cpu().numpy().astype(np.float64)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    def pair_value(a, b):
        if mode == "plane_intersect":
            return np.linalg.norm(a / a[..., 2:] - b / b[..., 2:], axis=-1) / offset
        # Cross/dot atan2 is independent of the production chord-length formula.
        return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))

    bottom_right = pair_value(rays[offset:, offset:], rays[:-offset, :-offset])
    top_right = pair_value(rays[offset:, :-offset], rays[:-offset, offset:])
    average = (
        np.pad(bottom_right, ((0, offset), (0, offset)), mode="edge")
        + np.pad(top_right, ((offset, 0), (0, offset)), mode="edge")
    ) / 2
    return 2 * np.tan(average / 2) / offset if mode == "cone" else average


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["cone", "plane_intersect"])
@pytest.mark.parametrize("num_threads", [1, 8])
def test_pixel_footprint_matches_float64_reference(precision, dtype, mode, num_threads):
    """Protect the precision-sensitive angular footprint and its padding edges."""
    previous_precision = torch.get_float32_matmul_precision()
    previous_threads = torch.get_num_threads()
    generator = torch.Generator().manual_seed(91)
    rays = torch.randn(96, 128, 3, dtype=dtype, generator=generator)
    rays[..., 2] = rays[..., 2].abs() + 2
    try:
        torch.set_num_threads(num_threads)
        torch.set_float32_matmul_precision(precision)
        expected = torch.from_numpy(_reference_footprint(rays, mode=mode)).to(dtype)
        actual = compute_pixel_footprint(rays, mode=mode)
        atol = 1e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(actual, expected, rtol=0, atol=atol)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("focal_length", [1000.0, 100000.0])
def test_pixel_footprint_preserves_small_angles(device, focal_length):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    y, x = torch.meshgrid(torch.arange(96), torch.arange(128), indexing="ij")
    rays = torch.stack([(x - 64) / focal_length, (y - 48) / focal_length, torch.ones_like(x)], dim=-1)
    expected = torch.from_numpy(_reference_footprint(rays)).float().to(device)
    rays = rays.to(device).requires_grad_()
    actual = compute_pixel_footprint(rays)
    assert torch.isfinite(actual).all() and (actual > 0).all()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-4)
    actual.sum().backward()
    assert rays.grad is not None
    assert torch.isfinite(rays.grad).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pixel_ray_angle_endpoints_and_zero_rays(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    a = torch.tensor([[0.0, 0.0, 1.0]] * 4 + [[0.0, 0.0, 0.0]], device=device, requires_grad=True)
    b = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        device=device,
        requires_grad=True,
    )
    angles = _pixel_ray_angle(a, b)
    expected = torch.tensor([0.0, np.pi, np.pi / 2, np.pi / 2, np.pi / 2], device=device)
    torch.testing.assert_close(angles, expected, atol=2e-7, rtol=0)
    angles.sum().backward()
    assert a.grad is not None and b.grad is not None
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_pixel_footprint_gradients():
    generator = torch.Generator().manual_seed(7)
    rays = torch.randn(3, 4, 3, dtype=torch.float64, generator=generator)
    rays[..., 2] = rays[..., 2].abs() + 2
    rays.requires_grad_()
    assert torch.autograd.gradcheck(lambda r: compute_pixel_footprint(r, pixel_offset=1), (rays,))


@pytest.mark.parametrize("mode", ["cone", "plane_intersect"])
def test_pixel_footprint_numpy(mode):
    generator = torch.Generator().manual_seed(5)
    rays = torch.randn(16, 18, 3, generator=generator)
    rays[..., 2] = rays[..., 2].abs() + 2
    result = compute_pixel_footprint(rays.numpy(), mode=mode)
    assert isinstance(result, np.ndarray) and result.dtype == np.float32
    np.testing.assert_allclose(result, _reference_footprint(rays, mode=mode), atol=1e-6, rtol=0)
