# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the four road-pixel quality BaseMetric subclasses."""

import math
import unittest

from unittest.mock import patch

import numpy as np
import torch

from nurec.metrics.impl.road_quality import LmMetric, RefMetric, RlcfMetric, RmtfMetric
from nurec.metrics.metric import MetricResult
from nurec.metrics.types import MetricType
from nurec.metrics.utils import AggregationMethod


def _make_image(height: int = 64, width: int = 96, seed: int = 42) -> np.ndarray:
    """Build a deterministic float32 RGB image in [0, 1] for tests."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0.1, 0.9, height), np.linspace(0.1, 0.9, width), indexing="ij")
    base = np.stack([yy, xx, (yy + xx) / 2.0], axis=-1).astype(np.float32)
    noise = np.asarray(rng.normal(0.0, 0.02, base.shape), dtype=np.float32)
    base += noise
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _make_mask(height: int = 64, width: int = 96) -> np.ndarray:
    """Boolean road mask: bottom half of the frame."""
    m = np.zeros((height, width), dtype=bool)
    m[height // 2 :, :] = True
    return m


def _hwc_to_chw_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, 3) float32 numpy -> (3, H, W) float32 tensor (CPU)."""
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


# (metric_cls, expected MetricType, headline key) covers every road sub-metric
# so per-class behaviour can be table-driven without duplicating cases.
_ALL_METRICS = (
    (RefMetric, MetricType.REF, "ref_chamfer"),
    (RlcfMetric, MetricType.RLCF, "rlcf_preservation"),
    (LmMetric, MetricType.LM, "lm_iou"),
    (RmtfMetric, MetricType.RMTF, "rmtf_aggregate"),
)


class TestRoadMetricsCommon(unittest.TestCase):
    """Behaviour identical across all four road metric classes."""

    def setUp(self) -> None:
        self.pred = _make_image()
        self.target = _make_image(seed=7)
        self.valid_mask = _make_mask()

    def test_type_enum_per_class(self) -> None:
        for cls, expected_type, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                self.assertEqual(cls(device="cpu").type(), expected_type)

    def test_semantic_class_attribute(self) -> None:
        # SEMANTIC_CLASS is the CLI-discovered output region: eval_rendering_metrics.py
        # reads it via getattr to route headline + diagnostic rows under the correct
        # semantic_class column value.
        for cls, _, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                self.assertEqual(cls.SEMANTIC_CLASS, "road")

    def test_emit_diagnostics_attribute(self) -> None:
        # EMIT_DIAGNOSTICS opts the class in to per-frame metadata export via
        # metric_group="diagnostic". Road classes opt in; metrics that don't
        # declare this attr keep the main-compat behaviour of dropping metadata.
        for cls, _, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                self.assertTrue(cls.EMIT_DIAGNOSTICS)

    def test_overlay_headline_opt_out(self) -> None:
        # Road metric headlines (ref_chamfer, lm_iou, ...) are not standard
        # scoreboard values, so they opt out of the comparison-image overlay
        # via the OVERLAY_HEADLINE class attr. The CLI reads it via getattr;
        # default True is implied for metrics that don't declare it.
        for cls, _, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                self.assertFalse(cls.OVERLAY_HEADLINE)

    def test_mask_kind_attribute(self) -> None:
        # MASK_KIND is the class-attr signal the CLI reads via getattr to pick
        # which mask (per-class road mask vs inverse camera mask) to pass to
        # each metric's _compute. All 4 road classes must declare "road".
        for cls, _, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                self.assertEqual(cls.MASK_KIND, "road")

    def test_compute_without_mask_fails_loudly(self) -> None:
        # The mask is required: the evaluation layer only invokes a road metric when
        # its mask exists, so a missing/None mask is a caller bug and must fail loudly.
        # (mask=None means "unmasked/full image" for psnr/ssim/lpips -- road metrics
        # must not silently reinterpret it.)
        for cls, _, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                metric = cls(device="cpu")
                with self.assertRaises(TypeError):
                    metric.compute(self.pred, self.target, None)

    def test_compute_returns_single_headline(self) -> None:
        for cls, _, headline in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                metric = cls(device="cpu")
                result = metric.compute(self.pred, self.target, self.valid_mask)
                self.assertIsInstance(result, MetricResult)
                self.assertEqual(
                    set(result.values.keys()),
                    {headline},
                    f"{cls.__name__} must put exactly one headline ({headline}) in values",
                )
                val = result.values[headline]
                self.assertIsInstance(val, torch.Tensor)
                self.assertEqual(val.numel(), 1)
                self.assertTrue(torch.isfinite(val).all())

    def test_diagnostics_land_in_metadata(self) -> None:
        # Per-frame diagnostics (precision/recall/f1/per-band ratios) must NOT
        # be promoted into `values` -- they belong in metadata so they don't
        # show up under static_camera_mask in rendering_metrics.yaml headlines.
        for cls, _, headline in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                metric = cls(device="cpu")
                result = metric.compute(self.pred, self.target, self.valid_mask)
                self.assertNotIn(headline, result.metadata)
                self.assertGreater(len(result.metadata), 0, f"{cls.__name__} must emit diagnostics in metadata")

    def test_compute_accepts_torch_tensors(self) -> None:
        for cls, _, headline in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                metric = cls(device="cpu")
                np_result = metric.compute(self.pred, self.target, self.valid_mask)
                t_result = metric.compute(
                    _hwc_to_chw_tensor(self.pred),
                    _hwc_to_chw_tensor(self.target),
                    torch.from_numpy(self.valid_mask),
                )
                self.assertAlmostEqual(
                    np_result.values[headline].item(),
                    t_result.values[headline].item(),
                    places=5,
                    msg=f"{cls.__name__} tensor path must match numpy path",
                )

    def test_metadata_method(self) -> None:
        for cls, _, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                meta = cls(device="cpu").metadata()
                self.assertIn("metric_name", meta)
                self.assertIn("sub_metric_params", meta)

    def test_compute_debug_images_returns_dict(self) -> None:
        for cls, _, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                metric = cls(device="cpu")
                # Pass by kwarg so the mask parameter name is part of the tested contract
                # (the CLI dispatcher passes mask=; a rename here would break integration).
                imgs = metric.compute_debug_images(gt=self.target, rendered=self.pred, mask=self.valid_mask)
                self.assertIsInstance(imgs, dict)
                self.assertGreater(len(imgs), 0, f"{cls.__name__} debug should yield at least one image")
                for name, arr in imgs.items():
                    self.assertIsInstance(arr, np.ndarray, f"{cls.__name__}/{name} must be an ndarray")


class TestRoadMetricsValidateInputs(unittest.TestCase):
    """validate_inputs is shared via _validate_road_inputs; one class exercises it."""

    def setUp(self) -> None:
        self.metric = RefMetric(device="cpu")
        self.pred = _make_image()
        self.target = _make_image(seed=7)
        self.valid_mask = _make_mask()

    def test_valid_numpy_inputs_pass(self) -> None:
        self.metric.validate_inputs(self.pred, self.target, self.valid_mask)

    def test_valid_torch_inputs_pass(self) -> None:
        self.metric.validate_inputs(
            _hwc_to_chw_tensor(self.pred),
            _hwc_to_chw_tensor(self.target),
            torch.from_numpy(self.valid_mask),
        )

    def test_pred_unsupported_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            self.metric.validate_inputs([1, 2, 3], self.target, self.valid_mask)  # type: ignore[arg-type]

    def test_wrong_ndim_raises(self) -> None:
        bad = np.zeros((64, 96), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.metric.validate_inputs(bad, self.target, self.valid_mask)

    def test_wrong_channels_raises(self) -> None:
        bad_pred = np.zeros((64, 96, 4), dtype=np.float32)
        bad_target = np.zeros((64, 96, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.metric.validate_inputs(bad_pred, bad_target, self.valid_mask)

    def test_shape_mismatch_raises(self) -> None:
        wrong_target = np.zeros((64, 80, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.metric.validate_inputs(self.pred, wrong_target, self.valid_mask)

    def test_mask_shape_mismatch_raises(self) -> None:
        bad_mask = np.zeros((32, 32), dtype=bool)
        with self.assertRaises(ValueError):
            self.metric.validate_inputs(self.pred, self.target, bad_mask)


class TestRefMetricDiagnostics(unittest.TestCase):
    """REF-specific behaviour beyond the shared shape tests."""

    def setUp(self) -> None:
        self.pred = _make_image()
        self.target = _make_image(seed=7)
        self.valid_mask = _make_mask()

    def test_ref_metadata_keys(self) -> None:
        metric = RefMetric(device="cpu")
        result = metric.compute(self.pred, self.target, self.valid_mask)
        for key in ("ref_precision", "ref_recall", "ref_f1"):
            self.assertIn(key, result.metadata, f"{key} should land in per-frame metadata")

    def test_sub_metric_params_propagates(self) -> None:
        metric = RefMetric(device="cpu", sub_metric_params={"tolerance_px": 2})
        result = metric.compute(self.pred, self.target, self.valid_mask)
        self.assertIn("ref_chamfer", result.values)

    def test_typoed_sub_metric_param_raises(self) -> None:
        # Helpers don't absorb unknown kwargs -- a typo surfaces as Python's
        # own TypeError, so no init-time registry/inspect validation needed.
        metric = RefMetric(device="cpu", sub_metric_params={"tolernce_px": 2})
        with self.assertRaises(TypeError) as ctx:
            metric.compute(self.pred, self.target, self.valid_mask)
        self.assertIn("tolernce_px", str(ctx.exception))


class TestRuntimeErrorHandling(unittest.TestCase):
    """Per-frame runtime errors must be swallowed; config errors must propagate."""

    def setUp(self) -> None:
        self.pred = _make_image()
        self.target = _make_image(seed=7)
        self.valid_mask = _make_mask()

    def test_helper_returns_nan_dict_produces_nan_headline(self) -> None:
        # When the helper catches a library error internally and returns its
        # NaN dict (the new in-helper failure path), the metric must produce
        # a NaN-headline MetricResult so MetricManager.collect_metric (which
        # rejects empty values) passes, and downstream NaN-aware aggregation
        # skips this frame. The NaN sub-scores ride along in metadata.
        nan_dict = {
            "ref_chamfer": float("nan"),
            "ref_precision": float("nan"),
            "ref_recall": float("nan"),
            "ref_f1": float("nan"),
        }

        metric = RefMetric(device="cpu")
        with patch("nurec.metrics.impl.road_quality.compute_ref", return_value=nan_dict):
            result = metric.compute(self.pred, self.target, self.valid_mask)
        self.assertIn(RefMetric.HEADLINE_KEY, result.values)
        self.assertTrue(torch.isnan(result.values[RefMetric.HEADLINE_KEY]).item())
        self.assertEqual(set(result.metadata.keys()), {"ref_precision", "ref_recall", "ref_f1"})
        for v in result.metadata.values():
            self.assertTrue(math.isnan(v))

    def test_config_error_propagates(self) -> None:
        # ValueError raised by the helper itself (out-of-range config) must
        # surface loudly, NOT get swallowed as a per-frame library error.
        metric = RmtfMetric(device="cpu", sub_metric_params={"n_scales": 99})
        with self.assertRaises(ValueError):
            metric.compute(self.pred, self.target, self.valid_mask)


class TestAggregate(unittest.TestCase):
    """Aggregation across stored MetricResults."""

    def setUp(self) -> None:
        self.pred = _make_image()
        self.target = _make_image(seed=7)
        self.valid_mask = _make_mask()

    def test_aggregate_empty_returns_empty_dict(self) -> None:
        for cls, _, _ in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                self.assertEqual(cls(device="cpu").aggregate(), {})

    def test_aggregate_mean_over_frames(self) -> None:
        for cls, _, headline in _ALL_METRICS:
            with self.subTest(metric=cls.__name__):
                metric = cls(device="cpu")
                for _ in range(3):
                    metric.append(metric.compute(self.pred, self.target, self.valid_mask))
                agg = metric.aggregate()
                self.assertIn(AggregationMethod.MEAN, agg)
                self.assertIn(headline, agg[AggregationMethod.MEAN].values)

    def test_aggregate_skips_empty_frames(self) -> None:
        # An empty-values MetricResult (helper swallowed a runtime error) must
        # not crash aggregation -- surviving frames still aggregate.
        metric = RefMetric(device="cpu")
        good = metric.compute(self.pred, self.target, self.valid_mask)
        metric.append(good)
        metric.append(MetricResult(values={}, metadata={}))
        metric.append(good)
        agg = metric.aggregate()
        self.assertIn("ref_chamfer", agg[AggregationMethod.MEAN].values)

    def test_aggregate_all_empty_returns_nan(self) -> None:
        # If every frame failed, aggregation must emit a NaN scalar rather
        # than raising -- downstream pipelines treat NaN as "skip this metric".
        metric = RefMetric(device="cpu")
        metric.append(MetricResult(values={}, metadata={}))
        metric.append(MetricResult(values={}, metadata={}))
        agg = metric.aggregate()
        self.assertTrue(torch.isnan(agg[AggregationMethod.MEAN].values["ref_chamfer"]).item())

    def test_clear_drops_stored_values(self) -> None:
        metric = RefMetric(device="cpu")
        metric.append(metric.compute(self.pred, self.target, self.valid_mask))
        self.assertEqual(len(metric._values), 1)
        metric.clear()
        self.assertEqual(metric._values, [])
        self.assertEqual(metric.aggregate(), {})

    def test_reset_is_a_noop(self) -> None:
        metric = RefMetric(device="cpu")
        metric.append(metric.compute(self.pred, self.target, self.valid_mask))
        metric.reset()
        self.assertEqual(len(metric._values), 1, "reset() must leave _values untouched (clear()'s job)")


if __name__ == "__main__":
    unittest.main()
