# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stateless road-pixel quality helpers in ``road_utils``."""

import math
import unittest

from unittest.mock import patch

import cv2
import numpy as np
import torch

from nurec.metrics.impl.utils import road_utils


class _RoadUtilsBase(unittest.TestCase):
    """Shared fixture: a deterministic 64x96 image + bottom-half road mask."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.height = 64
        self.width = 96
        yy, xx = np.meshgrid(np.linspace(0.1, 0.9, self.height), np.linspace(0.1, 0.9, self.width), indexing="ij")
        base = np.stack([yy, xx, (yy + xx) / 2.0], axis=-1).astype(np.float32)
        base += rng.normal(0.0, 0.02, base.shape).astype(np.float32)
        self.gt = np.clip(base, 0.0, 1.0).astype(np.float32)
        self.rendered = np.clip(self.gt + rng.normal(0.0, 0.05, self.gt.shape).astype(np.float32), 0.0, 1.0).astype(
            np.float32
        )
        self.valid_mask = np.zeros((self.height, self.width), dtype=bool)
        self.valid_mask[self.height // 2 :, :] = True
        self.empty_mask = np.zeros((self.height, self.width), dtype=bool)


class TestComputeRef(_RoadUtilsBase):
    """ref — Road Edge Fidelity (Canny P/R/F1 + Chamfer)."""

    def test_keys(self) -> None:
        out = road_utils.compute_ref(self.gt, self.rendered, self.valid_mask)
        self.assertEqual(set(out.keys()), {"ref_precision", "ref_recall", "ref_f1", "ref_chamfer"})

    def test_identical_inputs_perfect_match(self) -> None:
        out = road_utils.compute_ref(self.gt, self.gt, self.valid_mask)
        self.assertAlmostEqual(out["ref_f1"], 1.0, places=5)
        self.assertAlmostEqual(out["ref_chamfer"], 0.0, places=5)

    def test_blurred_render_degrades_ref(self) -> None:
        # Discriminative oracle: a strongly-blurred copy of GT must score
        # strictly worse than GT-vs-GT (F1 < 1, chamfer > 0). Guards against
        # the "identical-inputs = perfect score" tautology where production
        # code could be replaced with `return {...: 1.0}` and tests still pass.
        # The base fixture has no edges Canny can find, so paint a sharp bar
        # across the road region first.
        gt_with_edge = self.gt.copy()
        gt_with_edge[self.height // 2 + 5 : self.height // 2 + 15, :, :] = 0.95
        blurred = cv2.GaussianBlur(gt_with_edge, (0, 0), sigmaX=4.0)
        out = road_utils.compute_ref(gt_with_edge, blurred, self.valid_mask)
        self.assertLess(out["ref_f1"], 1.0, "Blurred render must produce ref_f1 < 1")
        self.assertGreater(out["ref_chamfer"], 0.0, "Blurred render must produce ref_chamfer > 0")

    def test_empty_mask_returns_nan(self) -> None:
        out = road_utils.compute_ref(self.gt, self.rendered, self.empty_mask)
        for v in out.values():
            self.assertTrue(math.isnan(v))

    def test_textureless_render_keeps_chamfer_finite_and_bounded(self) -> None:
        # GT with a sharp horizontal bar → Canny detects edges.
        # Flat-grey render → zero Canny edges. ref_chamfer must stay finite
        # AND bounded (10x tolerance_px, resolution-independent) so the
        # penalty hurts the per-camera mean without dominating it the way an
        # image-diagonal sentinel would.
        gt_with_edges = np.full_like(self.gt, 0.3)
        gt_with_edges[self.height // 2 + 5 : self.height // 2 + 15, :, :] = 0.9
        flat_rendered = np.full_like(self.rendered, 0.5)
        out = road_utils.compute_ref(gt_with_edges, flat_rendered, self.valid_mask)
        self.assertTrue(math.isfinite(out["ref_chamfer"]))
        self.assertAlmostEqual(out["ref_chamfer"], 10.0 * 3, places=5)  # default tolerance_px=3
        self.assertEqual(out["ref_recall"], 0.0)


class TestComputeRlcf(_RoadUtilsBase):
    """rlcf — Road Local Contrast Fidelity."""

    def test_keys(self) -> None:
        out = road_utils.compute_rlcf(self.gt, self.rendered, self.valid_mask)
        self.assertEqual(
            set(out.keys()),
            {"rlcf_correlation", "rlcf_preservation", "rlcf_high_contrast_gt", "rlcf_high_contrast_rendered"},
        )

    def test_identical_inputs_perfect_match(self) -> None:
        out = road_utils.compute_rlcf(self.gt, self.gt, self.valid_mask)
        self.assertAlmostEqual(out["rlcf_correlation"], 1.0, places=5)
        self.assertAlmostEqual(out["rlcf_preservation"], 1.0, places=5)

    def test_empty_mask_returns_nan(self) -> None:
        out = road_utils.compute_rlcf(self.gt, self.rendered, self.empty_mask)
        for v in out.values():
            self.assertTrue(math.isnan(v))

    def test_flat_gt_with_textured_render_preservation_is_finite(self) -> None:
        # GT is uniform (no local contrast on road) but render is textured.
        # preservation was returning NaN here; it must now collapse to a
        # finite value so the aggregated rlcf_preservation stays usable.
        flat_gt = np.zeros_like(self.gt)
        out = road_utils.compute_rlcf(flat_gt, self.rendered, self.valid_mask)
        self.assertTrue(math.isfinite(out["rlcf_preservation"]))
        self.assertEqual(out["rlcf_preservation"], 0.0)

    def test_rejects_non_positive_window_size(self) -> None:
        # Config errors must raise loudly, not surface as caught cv2.error -> all-NaN frames.
        with self.assertRaises(ValueError):
            road_utils.compute_rlcf(self.gt, self.rendered, self.valid_mask, window_size=0)


class TestComputeLm(_RoadUtilsBase):
    """lm — Lane Marking Detection (P/R/F1/IoU)."""

    def test_keys(self) -> None:
        out = road_utils.compute_lm(self.gt, self.rendered, self.valid_mask)
        self.assertEqual(set(out.keys()), {"lm_f1", "lm_iou", "lm_precision", "lm_recall"})

    def test_identical_inputs_perfect_match(self) -> None:
        out = road_utils.compute_lm(self.gt, self.gt, self.valid_mask)
        self.assertAlmostEqual(out["lm_f1"], 1.0, places=5)
        self.assertAlmostEqual(out["lm_iou"], 1.0, places=5)

    def test_blurred_render_degrades_lm(self) -> None:
        # Discriminative oracle: blurred render must produce strictly lower
        # lm_iou than identical inputs (1.0). Avoids the perfect-match tautology.
        # Paint a sharp bright line across the road region to give the
        # lane-marking detector something to find.
        gt_with_marking = self.gt.copy()
        gt_with_marking[self.height // 2 + 5 : self.height // 2 + 10, :, :] = 0.95
        blurred = cv2.GaussianBlur(gt_with_marking, (0, 0), sigmaX=4.0)
        out = road_utils.compute_lm(gt_with_marking, blurred, self.valid_mask)
        self.assertLess(out["lm_iou"], 1.0, "Blurred render must produce lm_iou < 1")

    def test_empty_mask_returns_nan(self) -> None:
        out = road_utils.compute_lm(self.gt, self.rendered, self.empty_mask)
        for v in out.values():
            self.assertTrue(math.isnan(v))

    def test_rejects_invalid_config_params(self) -> None:
        # Config errors must raise loudly, not surface as caught cv2.error -> all-NaN frames.
        with self.assertRaises(ValueError):
            road_utils.compute_lm(self.gt, self.rendered, self.valid_mask, morph_ksize=0)
        with self.assertRaises(ValueError):
            road_utils.compute_lm(self.gt, self.rendered, self.valid_mask, blur_sigma=0.0)


class TestComputeRmtf(_RoadUtilsBase):
    """rmtf — Road Multi-Scale Texture Fidelity (DoG)."""

    def test_keys(self) -> None:
        out = road_utils.compute_rmtf(self.gt, self.rendered, self.valid_mask)
        self.assertIn("rmtf_aggregate", out)
        self.assertEqual(len(out), 5, "4 DoG ratios + aggregate")

    def test_identical_inputs_aggregate_one(self) -> None:
        out = road_utils.compute_rmtf(self.gt, self.gt, self.valid_mask)
        self.assertAlmostEqual(out["rmtf_aggregate"], 1.0, places=5)

    def test_empty_mask_returns_nan(self) -> None:
        out = road_utils.compute_rmtf(self.gt, self.rendered, self.empty_mask)
        for v in out.values():
            self.assertTrue(math.isnan(v))

    def test_flat_gt_with_textured_render_keeps_outputs_finite(self) -> None:
        # A flat-GT band used to produce NaN ratios and poison the
        # weighted aggregate. The aggregate and every per-band ratio
        # must now stay finite (degenerate band hits the 2.0 cap).
        flat_gt = np.zeros_like(self.gt)
        out = road_utils.compute_rmtf(flat_gt, self.rendered, self.valid_mask)
        self.assertTrue(math.isfinite(out["rmtf_aggregate"]))
        self.assertGreater(out["rmtf_aggregate"], 0.0)
        for key, val in out.items():
            self.assertTrue(math.isfinite(val), f"{key} must be finite, got {val}")
        self.assertEqual(out["rmtf_ratio_s1"], 2.0)

    def test_rejects_n_scales_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            road_utils.compute_rmtf(self.gt, self.rendered, self.valid_mask, n_scales=5)
        with self.assertRaises(ValueError):
            road_utils.compute_rmtf(self.gt, self.rendered, self.valid_mask, n_scales=0)

    def test_rejects_non_positive_base_sigma(self) -> None:
        # Config errors must raise loudly, not surface as caught cv2.error -> all-NaN frames.
        with self.assertRaises(ValueError):
            road_utils.compute_rmtf(self.gt, self.rendered, self.valid_mask, base_sigma=0.0)

    def test_band_keys_unique_for_sub_unit_base_sigma(self) -> None:
        # int-truncated labels would collide here (int(0.1*2**i) = 0 for i<4);
        # the keys must remain distinct so per-band ratios don't overwrite.
        out = road_utils.compute_rmtf(self.gt, self.rendered, self.valid_mask, base_sigma=0.1)
        band_keys = [k for k in out if k != "rmtf_aggregate"]
        self.assertEqual(len(band_keys), 4)
        self.assertEqual(len(set(band_keys)), 4, f"band keys must be unique, got {band_keys}")


class TestHelperLibraryErrorHandling(_RoadUtilsBase):
    """compute_*/debug_* helpers catch degenerate-input library errors and return NaN dict / {}."""

    def test_compute_ref_catches_cv2_error_returns_nan_dict(self) -> None:
        # Simulate cv2.Canny raising mid-pipeline on a degenerate frame.
        # compute_ref must catch it and return its NaN-dict (same schema as
        # the empty-mask early-return path), NOT propagate.
        with patch.object(road_utils, "_canny_with_adaptive_thresholds", side_effect=cv2.error("boom")):
            out = road_utils.compute_ref(self.gt, self.rendered, self.valid_mask)
        self.assertEqual(set(out.keys()), {"ref_chamfer", "ref_precision", "ref_recall", "ref_f1"})
        for v in out.values():
            self.assertTrue(math.isnan(v))

    def test_compute_ref_does_not_catch_value_error(self) -> None:
        # Config errors (ValueError) must propagate -- not be swallowed as
        # a library failure.
        with patch.object(road_utils, "_canny_with_adaptive_thresholds", side_effect=ValueError("bad config")):
            with self.assertRaises(ValueError):
                road_utils.compute_ref(self.gt, self.rendered, self.valid_mask)

    def test_debug_ref_catches_cv2_error_returns_empty_dict(self) -> None:
        with patch.object(road_utils, "_canny_with_adaptive_thresholds", side_effect=cv2.error("boom")):
            out = road_utils.debug_ref(self.gt, self.rendered, self.valid_mask)
        self.assertEqual(out, {})


class TestSelectVizFrames(unittest.TestCase):
    """Clip-level sampling helper for the debug_* visualisations."""

    def test_small_clips_are_visualised_in_full(self) -> None:
        names = [str(n) for n in range(10)]
        self.assertEqual(road_utils.select_viz_frames(names), set(names))

    def test_zero_disables_sampling(self) -> None:
        names = [str(n) for n in range(100)]
        self.assertEqual(road_utils.select_viz_frames(names, every_nth=0), set(names))

    def test_samples_one_in_every_nth(self) -> None:
        names = [str(116930000000 + i * 33333) for i in range(600)]
        selected = road_utils.select_viz_frames(names)  # default: 1 in 30
        self.assertEqual(len(selected), 20)
        self.assertTrue(selected.issubset(set(names)))

    def test_sampling_is_deterministic_across_calls(self) -> None:
        names = [str(116930000000 + i * 33333) for i in range(600)]
        self.assertEqual(road_utils.select_viz_frames(names), road_utils.select_viz_frames(names))

    def test_small_sample_counts_hit_the_10_frame_floor(self) -> None:
        # 90 frames / 30 = 3 -> the 10-frame floor applies.
        names = [f"frame_{i:04d}" for i in range(90)]
        self.assertEqual(len(road_utils.select_viz_frames(names)), 10)


class TestTorchKernelProperties(unittest.TestCase):
    """Analytic properties of the torch filtering kernels.

    No external implementation is normative for these kernels, so instead of
    diffing against one, each is pinned by properties with known-exact
    answers: constant preservation, impulse symmetry/normalisation, and
    hand-checkable neighbourhoods.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.img64 = torch.from_numpy(rng.random((64, 96)))[None, None]

    def test_gaussian_blur_preserves_constant_image(self) -> None:
        # A normalised kernel with reflect padding must return a constant
        # image unchanged, at any sigma (including kernels bigger than the
        # image, which are clamped to fit).
        const = torch.full((1, 1, 64, 96), 0.37, dtype=torch.float64)
        for sigma in (1.0, 4.0, 16.0, 21.0):
            with self.subTest(sigma=sigma):
                out = road_utils._gaussian_blur_t(const, sigma)
                np.testing.assert_allclose(out.numpy(), 0.37, atol=1e-12)

    def test_gaussian_blur_impulse_response_symmetric_and_normalised(self) -> None:
        impulse = torch.zeros((1, 1, 65, 65), dtype=torch.float64)
        impulse[0, 0, 32, 32] = 1.0
        out = road_utils._gaussian_blur_t(impulse, 3.0)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=10, msg="kernel must be normalised")
        np.testing.assert_allclose(out.numpy(), out.flip(-1).numpy(), atol=1e-15)
        np.testing.assert_allclose(out.numpy(), out.flip(-2).numpy(), atol=1e-15)

    def test_gaussian_blur_clamps_kernel_to_tiny_images(self) -> None:
        # sigma=16 auto-sizes to a 97-tap kernel; on a 20x30 image it must
        # clamp to the image instead of failing in torch's reflect pad.
        out = road_utils._gaussian_blur_t(torch.rand(1, 1, 20, 30, dtype=torch.float64), 16.0)
        self.assertEqual(out.shape, (1, 1, 20, 30))
        self.assertTrue(bool(out.isfinite().all()))

    def test_box_mean_equals_neighbourhood_average_in_interior(self) -> None:
        # Away from the border, the box mean is exactly the window average.
        out = road_utils._box_mean_t(self.img64, 5)[0, 0]
        img = self.img64[0, 0]
        for i, j in ((10, 10), (32, 48), (50, 90)):
            self.assertAlmostEqual(
                float(out[i, j]), float(img[i - 2 : i + 3, j - 2 : j + 3].mean()), places=12, msg=f"pixel ({i}, {j})"
            )

    def test_binary_open_removes_specks_keeps_blocks(self) -> None:
        mask = torch.zeros((32, 32), dtype=torch.bool)
        mask[5, 5] = True  # isolated pixel: must be removed
        mask[20:23, 20:23] = True  # 3x3 block: must survive intact
        opened = road_utils._binary_open_t(mask, 3)
        self.assertFalse(bool(opened[5, 5]))
        self.assertTrue(bool(opened[20:23, 20:23].all()))
        self.assertEqual(int(opened.sum()), 9)


@unittest.skipUnless(torch.cuda.is_available(), "CPU/CUDA parity requires a GPU")
class TestCpuCudaParity(_RoadUtilsBase):
    """Identical inputs must score identically (within float tolerance) on CPU and CUDA."""

    def test_compute_helpers_agree_across_devices(self) -> None:
        computes = (road_utils.compute_rlcf, road_utils.compute_lm, road_utils.compute_rmtf)
        for compute in computes:
            with self.subTest(helper=compute.__name__):
                cpu = compute(self.gt, self.rendered, self.valid_mask, device="cpu")
                cuda = compute(self.gt, self.rendered, self.valid_mask, device="cuda")
                self.assertEqual(set(cpu.keys()), set(cuda.keys()))
                for key, cpu_val in cpu.items():
                    self.assertAlmostEqual(cpu_val, cuda[key], places=5, msg=key)


if __name__ == "__main__":
    unittest.main()
