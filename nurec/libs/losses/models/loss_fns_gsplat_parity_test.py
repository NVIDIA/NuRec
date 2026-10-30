# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins the gsplat Tier-1 loss functions against the implementations they replaced.

Each reference below is the implementation this repository carried before it
started calling gsplat's. They are kept verbatim so a silent change on the gsplat
side fails here rather than in training.
"""

import unittest

import torch
import torch.nn.functional as F

from gsplat.losses import BG_GRID_NUM_COLS, BG_GRID_NUM_ROWS

import nurec.libs.losses.models.loss_fns  # noqa: F401  -- registers the functions under test

from nurec.libs.losses.models.registry import loss_fns


def _ref_bce_clipped(input: torch.Tensor, target: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        return F.binary_cross_entropy(input.float().clip(eps, 1 - eps), target.float(), reduction="none")


def _ref_weights_reg(weights_list: list[torch.Tensor], dim: int = 1) -> torch.Tensor:
    return torch.mean(torch.cat([(weights**2).sum(dim) for weights in weights_list]))


def _ref_relu_sum(value: torch.Tensor, eps: float) -> torch.Tensor:
    return F.relu(value - eps).sum()


def _ref_log_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.log1p((pred - target).abs())


def _ref_total_variation_temporal(x: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    tv_t = x.diff(dim=0).square().mean(dim=(1, 2, 3, 4)) if x.shape[0] > 1 else torch.zeros(1, device=x.device)
    return tv_t * loss_mask


def _ref_identity_distance(grid: torch.Tensor) -> torch.Tensor:
    reshaped_grid = grid.view(grid.shape[0], BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, *grid.shape[2:])
    identity = torch.eye(BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, device=grid.device)
    identity = identity.view(1, BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS, *([1] * len(grid.shape[2:])))
    return torch.norm(reshaped_grid - identity, p="fro", dim=(1, 2))


def _ref_depth_inverse_mse(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    return (torch.clamp(pred, min=eps).reciprocal() - torch.clamp(target, min=eps).reciprocal()).pow(2)


def _ref_normal_cosine(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.sum(pred_normal * gt_normal, dim=-1)


class TestGsplatTier1Parity(unittest.TestCase):
    """Every registered function that now resolves to gsplat matches its predecessor."""

    def setUp(self):
        torch.manual_seed(0)

    def test_bce_clipped(self):
        pred = torch.rand(4, 5)
        target = torch.rand(4, 5)
        torch.testing.assert_close(loss_fns["bce_clipped"](pred, target), _ref_bce_clipped(pred, target))

    def test_bce_clipped_honours_eps_clipping(self):
        # Values outside (eps, 1 - eps) are where clipping is observable at all.
        pred = torch.tensor([0.0, 1.0, 0.5])
        target = torch.tensor([1.0, 0.0, 0.5])
        got = loss_fns["bce_clipped"](pred, target, eps=0.01)
        torch.testing.assert_close(got, _ref_bce_clipped(pred, target, eps=0.01))
        self.assertTrue(torch.isfinite(got).all(), "clipping must keep BCE finite at 0 and 1")

    def test_weights_reg(self):
        ws = [torch.randn(3, 4), torch.randn(2, 4)]
        torch.testing.assert_close(loss_fns["weights_reg"](ws), _ref_weights_reg(ws))

    def test_relu_sum(self):
        v = torch.randn(6, 7)
        torch.testing.assert_close(loss_fns["relu_sum"](v, 0.25), _ref_relu_sum(v, 0.25))

    def test_identity_distance(self):
        channels = BG_GRID_NUM_ROWS * BG_GRID_NUM_COLS
        grid = torch.randn(3, channels, 2, 5, 5)
        torch.testing.assert_close(loss_fns["identity_distance"](grid), _ref_identity_distance(grid))

    def test_identity_distance_is_zero_on_the_identity(self):
        # Guards the row/col order: a transposed reshape would not vanish here.
        identity = torch.eye(BG_GRID_NUM_ROWS, BG_GRID_NUM_COLS).reshape(1, -1, 1, 1, 1)
        grid = identity.expand(2, BG_GRID_NUM_ROWS * BG_GRID_NUM_COLS, 1, 3, 3).contiguous()
        torch.testing.assert_close(loss_fns["identity_distance"](grid), torch.zeros(2, 1, 3, 3), atol=1e-6, rtol=0)

    def test_depth_inverse_mse(self):
        pred = torch.rand(4, 6) * 50.0
        target = torch.rand(4, 6) * 50.0
        torch.testing.assert_close(
            loss_fns["depth_inverse_mse"](pred, target, 1e-3), _ref_depth_inverse_mse(pred, target, 1e-3)
        )

    def test_depth_inverse_mse_accepts_registry_keyword_extras(self):
        # The registry hands loss fns keyword arguments that gsplat's signature does not take.
        pred, target = torch.rand(2, 2) + 1.0, torch.rand(2, 2) + 1.0
        loss_fns["depth_inverse_mse"](pred, target, 1e-3, unused_extra=object())

    def test_normal_cosine(self):
        pred = F.normalize(torch.randn(7, 3), dim=-1)
        gt = F.normalize(torch.randn(7, 3), dim=-1)
        torch.testing.assert_close(loss_fns["normal_cosine"](pred, gt), _ref_normal_cosine(pred, gt))

    def test_log_l1(self):
        pred = torch.randn(4, 5)
        target = torch.randn(4, 5)
        torch.testing.assert_close(loss_fns["log_l1"](pred, target), _ref_log_l1(pred, target))

    def test_log_l1_keeps_residuals_below_the_float32_rounding_threshold(self):
        # The case gsplat!493 fixed: 1 + x rounds to 1 for x < 2**-24, log1p does not.
        pred = torch.tensor([1e-8], dtype=torch.float32)
        target = torch.zeros_like(pred)
        got = loss_fns["log_l1"](pred, target)
        self.assertGreater(got.item(), 0.0)
        torch.testing.assert_close(got, pred, rtol=1e-3, atol=0)

    def test_total_variation_temporal(self):
        x = torch.randn(5, 3, 2, 4, 4)
        mask = torch.rand(4)
        torch.testing.assert_close(
            loss_fns["total_variation_temporal"](x, mask), _ref_total_variation_temporal(x, mask)
        )

    def test_total_variation_temporal_single_frame_keeps_the_empty_mask(self):
        # The case gsplat!493 fixed: a per-transition mask is empty at B=1, so the
        # result must stay empty rather than contributing a zero to a downstream mean.
        x = torch.randn(1, 3, 2, 4, 4)
        empty_mask = torch.rand(0)
        got = loss_fns["total_variation_temporal"](x, empty_mask)
        self.assertEqual(got.shape, (0,))
        torch.testing.assert_close(got, _ref_total_variation_temporal(x, empty_mask))

    def test_gradients_flow_through_the_gsplat_functions(self):
        for name, args in (
            ("relu_sum", (torch.randn(5, requires_grad=True), 0.1)),
            (
                "normal_cosine",
                (F.normalize(torch.randn(4, 3), dim=-1).requires_grad_(True), F.normalize(torch.randn(4, 3), dim=-1)),
            ),
        ):
            with self.subTest(fn=name):
                out = loss_fns[name](*args)
                out.sum().backward()
                self.assertIsNotNone(args[0].grad)
                self.assertTrue(torch.isfinite(args[0].grad).all())


class TestDocumentedDivergences(unittest.TestCase):
    """The functions deliberately kept local, with the reason made executable.

    If an assertion here starts failing, gsplat has converged on this repository's
    semantics and the local copy should be deleted in favour of it, as was done for
    log_l1 and total_variation_temporal.
    """

    def test_gsplat_total_variation_has_a_different_contract(self):
        from gsplat.losses.functional import total_variation_loss

        x = torch.randn(4, 3, 2, 5, 5)
        self.assertEqual(loss_fns["total_variation_spatial"](x).shape, (4,))
        self.assertEqual(total_variation_loss(x).shape, ())


if __name__ == "__main__":
    unittest.main()
