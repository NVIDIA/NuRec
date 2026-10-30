# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU K-nearest-neighbour kernel checked against a brute-force reference.

Covers both the forward index/distance search and the backward gradient
accumulation, for the L1 and L2 norms.
"""

import pytest
import torch

from nurec.libs.pytorch3d_knn.interface import pytorch3d_knn


def _reference_knn(p1: torch.Tensor, p2: torch.Tensor, k: int, norm: int):
    """Brute-force nearest neighbours computed with plain torch ops."""
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = diff.abs().sum(-1) if norm == 1 else (diff * diff).sum(-1)
    return dist.topk(k, dim=-1, largest=False)


def _inputs(n=2, p1=7, p2=9, d=3, seed=0):
    torch.manual_seed(seed)
    a = torch.rand(n, p1, d, dtype=torch.float32)
    b = torch.rand(n, p2, d, dtype=torch.float32)
    lengths1 = torch.full((n,), p1, dtype=torch.int64)
    lengths2 = torch.full((n,), p2, dtype=torch.int64)
    return a, b, lengths1, lengths2


@pytest.mark.parametrize("norm", [1, 2])
def test_forward_matches_brute_force(norm):
    a, b, lengths1, lengths2 = _inputs()
    k = 4

    idx, dists = pytorch3d_knn.knn_points_idx(a, b, lengths1, lengths2, norm, k, -1)
    expected = _reference_knn(a, b, k, norm)

    # Sort both sides by distance: ties may be ordered differently.
    got = torch.sort(dists, dim=-1).values
    assert torch.allclose(got, expected.values, atol=1e-5), f"distances differ: {(got - expected.values).abs().max()}"
    assert idx.shape == (a.shape[0], a.shape[1], k)
    # Every returned index must point at a real point in p2.
    assert int(idx.min()) >= 0 and int(idx.max()) < b.shape[1]


@pytest.mark.parametrize("norm", [1, 2])
def test_backward_matches_autograd(norm):
    a, b, lengths1, lengths2 = _inputs()
    k = 3

    idx, _ = pytorch3d_knn.knn_points_idx(a, b, lengths1, lengths2, norm, k, -1)
    grad_dists = torch.rand(a.shape[0], a.shape[1], k, dtype=torch.float32)
    grad_a, grad_b = pytorch3d_knn.knn_points_backward(a, b, lengths1, lengths2, idx, norm, grad_dists)

    # Reference: differentiate the same gathered distances with autograd.
    ra = a.clone().requires_grad_(True)
    rb = b.clone().requires_grad_(True)
    diff = ra[:, :, None, :] - rb[:, None, :, :]
    full = diff.abs().sum(-1) if norm == 1 else (diff * diff).sum(-1)
    gathered = torch.gather(full, 2, idx)
    gathered.backward(grad_dists)

    assert torch.allclose(grad_a, ra.grad, atol=1e-4), f"grad p1 differs: {(grad_a - ra.grad).abs().max()}"
    assert torch.allclose(grad_b, rb.grad, atol=1e-4), f"grad p2 differs: {(grad_b - rb.grad).abs().max()}"


def test_ragged_lengths_only_use_valid_points():
    """Points beyond `lengths` must not contribute; shortening p2 changes results."""
    a, b, lengths1, lengths2 = _inputs()
    k = 2

    full_idx, full_dists = pytorch3d_knn.knn_points_idx(a, b, lengths1, lengths2, 2, k, -1)
    short = torch.full_like(lengths2, 3)
    short_idx, short_dists = pytorch3d_knn.knn_points_idx(a, b, lengths1, short, 2, k, -1)

    assert int(short_idx.max()) < 3, "neighbours must come from the first `lengths2` points"
    assert torch.all(short_dists >= full_dists - 1e-6), "restricting candidates cannot reduce distances"
