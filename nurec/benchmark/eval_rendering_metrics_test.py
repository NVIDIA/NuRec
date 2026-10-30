# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import tempfile
import unittest

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from PIL import Image, ImageFont

from nurec.benchmark.eval_rendering_metrics import (
    MetricsStore,
    compute_image_metrics,
    create_comparison_image,
    create_diff_image,
    create_image_comparison_metric_manager,
    draw_text,
    find_image,
    load_image_to_tensor,
    load_mask_to_tensor,
    load_timestamps_json,
    overlay_mask,
)


# All tests run on the CPU so they don't require a GPU on the CI runner.
CPU = torch.device("cpu")


def _write_png(path: str, array: np.ndarray, mode: str = "RGB") -> None:
    """Persist a uint8 numpy array as a PNG for the loader helpers to read back."""
    Image.fromarray(array, mode=mode).save(path)


class TestLoadTimestampsJson(unittest.TestCase):
    def test_maps_basename_to_int_timestamp(self):
        frame_list = [
            {"file_name": "100.png", "frame_end_timestamp_us": 100},
            {"file_name": "200.jpg", "frame_end_timestamp_us": "200"},  # string is coerced to int
        ]
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "timestamps.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(frame_list, f)
            timestamps = load_timestamps_json(json_path)

        self.assertEqual(timestamps, {"100": 100, "200": 200})
        self.assertIsInstance(timestamps["200"], int)


class TestFindImage(unittest.TestCase):
    def test_returns_path_of_existing_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            noext = os.path.join(tmp, "frame")
            _write_png(noext + ".png", np.zeros((2, 2, 3), dtype=np.uint8))
            self.assertEqual(find_image(noext), noext + ".png")

    def test_respects_extension_priority_order(self):
        # The default extension order lists .jpg before .png, so .jpg wins when both exist.
        with tempfile.TemporaryDirectory() as tmp:
            noext = os.path.join(tmp, "frame")
            _write_png(noext + ".jpg", np.zeros((2, 2, 3), dtype=np.uint8))
            _write_png(noext + ".png", np.zeros((2, 2, 3), dtype=np.uint8))
            self.assertEqual(find_image(noext), noext + ".jpg")

    def test_raises_when_no_image_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                find_image(os.path.join(tmp, "missing"))


class TestLoadImageToTensor(unittest.TestCase):
    def test_shape_dtype_and_normalization(self):
        # Solid red image: R=255, G=B=0.
        array = np.zeros((3, 4, 3), dtype=np.uint8)
        array[..., 0] = 255
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "red.png")
            _write_png(path, array)
            tensor = load_image_to_tensor(path, device=CPU)

        self.assertEqual(tensor.shape, (3, 3, 4))  # (C, H, W)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.all(tensor[0] == 1.0))  # red channel
        self.assertTrue(torch.all(tensor[1] == 0.0))
        self.assertTrue(torch.all(tensor[2] == 0.0))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)


class TestLoadMaskToTensor(unittest.TestCase):
    def _mask_png(self, tmp: str) -> str:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :] = 255  # top half set
        path = os.path.join(tmp, "mask.png")
        _write_png(path, mask, mode="L")
        return path

    def test_loads_boolean_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensor = load_mask_to_tensor(self._mask_png(tmp), device=CPU)
        self.assertEqual(tensor.dtype, torch.bool)
        self.assertTrue(torch.all(tensor[:2, :]))
        self.assertFalse(torch.any(tensor[2:, :]))

    def test_invert_flips_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._mask_png(tmp)
            normal = load_mask_to_tensor(path, device=CPU)
            inverted = load_mask_to_tensor(path, device=CPU, invert=True)
        self.assertTrue(torch.equal(inverted, ~normal))


class TestOverlayMask(unittest.TestCase):
    def test_masked_pixels_brightened_others_unchanged(self):
        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        result = overlay_mask(image, mask, alpha=0.5)
        # Brightened pixel: 0.5 * 100 + 0.5 * 255 = 177.5 -> 177 (uint8 truncation).
        self.assertEqual(result[0, 0, 0], 177)
        # Unmasked pixels are untouched.
        np.testing.assert_array_equal(result[1, 1], np.array([100, 100, 100], dtype=np.uint8))


class TestCreateDiffImage(unittest.TestCase):
    def test_constant_difference_with_default_factor(self):
        pred = np.full((2, 2, 3), 10, dtype=np.uint8)
        target = np.full((2, 2, 3), 4, dtype=np.uint8)
        diff, factor = create_diff_image(pred, target)
        self.assertEqual(factor, 1.0)
        self.assertTrue(np.all(diff == 6))
        self.assertEqual(diff.dtype, np.uint8)

    def test_explicit_factor_scales_difference(self):
        pred = np.full((2, 2, 3), 10, dtype=np.uint8)
        target = np.full((2, 2, 3), 4, dtype=np.uint8)
        diff, factor = create_diff_image(pred, target, factor=2.0)
        self.assertEqual(factor, 2.0)
        self.assertTrue(np.all(diff == 12))

    def test_auto_factor_normalizes_to_255(self):
        pred = np.full((2, 2, 3), 10, dtype=np.uint8)
        target = np.full((2, 2, 3), 4, dtype=np.uint8)
        diff, factor = create_diff_image(pred, target, factor=None)
        self.assertAlmostEqual(factor, 255.0 / 6.0)
        self.assertTrue(np.all(diff == 255))  # max difference stretched to full range

    def test_identical_images_auto_factor_avoids_divide_by_zero(self):
        img = np.full((2, 2, 3), 30, dtype=np.uint8)
        diff, factor = create_diff_image(img, img.copy(), factor=None)
        self.assertEqual(factor, 1.0)
        self.assertTrue(np.all(diff == 0))

    def test_mask_suppresses_pixels(self):
        pred = np.full((2, 2, 3), 10, dtype=np.uint8)
        target = np.full((2, 2, 3), 4, dtype=np.uint8)
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        diff, _ = create_diff_image(pred, target, mask=mask)
        self.assertTrue(np.all(diff[0, 0] == 0))  # masked pixel suppressed
        self.assertTrue(np.all(diff[1, 1] == 6))  # unmasked pixel keeps its diff

    def test_invert(self):
        pred = np.full((2, 2, 3), 10, dtype=np.uint8)
        target = np.full((2, 2, 3), 4, dtype=np.uint8)
        diff, _ = create_diff_image(pred, target, invert=True)
        self.assertTrue(np.all(diff == 255 - 6))


class TestDrawText(unittest.TestCase):
    def test_draws_in_place_and_returns_none(self):
        image = Image.new("RGB", (120, 40), color=(0, 0, 0))
        before = np.array(image).copy()
        font = ImageFont.load_default(size=20)
        result = draw_text(image, "hello", font=font)
        self.assertIsNone(result)
        self.assertFalse(np.array_equal(before, np.array(image)))  # pixels changed


class TestCreateComparisonImage(unittest.TestCase):
    def _pair(self, h: int = 8, w: int = 6):
        pred = torch.zeros((3, h, w), dtype=torch.float32)
        target = torch.ones((3, h, w), dtype=torch.float32)
        return pred, target

    def test_returns_2x2_stacked_image(self):
        pred, target = self._pair()
        out = create_comparison_image(pred, target)
        self.assertIsInstance(out, Image.Image)
        # PIL size is (width, height); stacked image is 2x the source in each axis.
        self.assertEqual(out.size, (6 * 2, 8 * 2))

    def test_accepts_mask_and_metrics_overlay(self):
        pred, target = self._pair()
        mask = torch.zeros((8, 6), dtype=torch.bool)
        mask[0, 0] = True
        out = create_comparison_image(pred, target, mask=mask, metrics={"psnr": 12.34})
        self.assertEqual(out.size, (12, 16))


class TestCreateImageComparisonMetricManager(unittest.TestCase):
    def test_registers_requested_metrics(self):
        manager = create_image_comparison_metric_manager(["psnr", "ssim"], device="cpu")
        self.assertEqual(sorted(manager.list_metrics()), ["psnr", "ssim"])

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            create_image_comparison_metric_manager(["not_a_metric"], device="cpu")


class TestComputeImageMetrics(unittest.TestCase):
    """Covers this module's dispatch/plumbing: batch-dim insertion, per-metric mask
    selection, and turning MetricResults into labeled MetricRecords + an overlay dict.
    The metric math itself is owned by the metrics' own tests (e.g. psnr_test.py).
    """

    def test_returns_labeled_records_and_overlay(self):
        # 16x16 so SSIM's default gaussian window fits.
        manager = create_image_comparison_metric_manager(["psnr", "ssim"], device="cpu")
        identical = torch.zeros((3, 16, 16), dtype=torch.float32)
        different = torch.full((3, 16, 16), 0.5, dtype=torch.float32)

        # Scheduling is explicit; omitting `masks` means every named metric runs maskless.
        records, overlay = compute_image_metrics(manager, identical, different, ["psnr", "ssim"])

        self.assertEqual({r.metric_name for r in records}, {"psnr", "ssim"})
        # psnr/ssim declare no MASK_KIND/SEMANTIC_CLASS -> default label, headline group.
        self.assertTrue(all(r.semantic_class == "static_camera_mask" for r in records))
        self.assertTrue(all(r.metric_group == "headline" for r in records))
        self.assertTrue(all(isinstance(r.metric_value, float) and math.isfinite(r.metric_value) for r in records))
        self.assertEqual(set(overlay), {"psnr", "ssim"})  # OVERLAY_HEADLINE defaults to True


# --- Equivalence harness: proves the extracted compute_image_metrics() is behavior-
# --- preserving vs the pre-refactor inline CLI block (byte-identical MetricsStore + overlay).


def _fake_result(values, metadata=None):
    """A stand-in MetricResult: .values maps name -> tensor scalar, plus .metadata."""
    return SimpleNamespace(
        values={k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()},
        metadata=metadata or {},
    )


class _FakeManager:
    """Minimal stand-in for the four MetricManager methods the dispatcher uses, so the
    road/camera/diagnostic branches can be exercised deterministically without GPU or
    real metric weights -- we compare dispatch logic here, not metric math."""

    def __init__(self, specs):
        # specs: list of (name, metric_with_attrs, preset_result), order preserved.
        self._order = [name for name, _, _ in specs]
        self._specs = {name: (metric, result) for name, metric, result in specs}

    def list_metrics(self):
        return list(self._order)

    def get_metric(self, name):
        return self._specs[name][0]

    def compute(self, name, pred, target, mask):  # no-op: result is preset
        pass

    def get_last(self, name):
        return self._specs[name][1]


def _inline_reference(manager, store, rendered_image, gt_image, *, inv_camera_mask, per_class_valid_mask):
    """Verbatim copy of the pre-refactor inline CLI block, writing into `store`.

    Retained here as the reference implementation so the equivalence test can prove the
    extracted compute_image_metrics() yields identical MetricsStore contents and overlay.
    """
    pred_batched = rendered_image.unsqueeze(0)
    target_batched = gt_image.unsqueeze(0)
    overlay = {}
    for metric_name in manager.list_metrics():
        metric = manager.get_metric(metric_name)
        mask_kind = getattr(metric, "MASK_KIND", None)
        mask = per_class_valid_mask if mask_kind == "road" else inv_camera_mask
        if mask is None and mask_kind == "road":
            continue
        manager.compute(metric_name, pred_batched, target_batched, mask)
        result = manager.get_last(metric_name)
        if result is None:
            continue
        mask_region = getattr(metric, "SEMANTIC_CLASS", "static_camera_mask")
        for value_key, value in result.values.items():
            scalar = float(value.item())
            store.add_metric(
                sequence_id="cam0",
                frame_id="f1",
                frame_timestamp_us=100,
                metric_name=value_key,
                metric_value=scalar,
                semantic_class=mask_region,
                metric_group="headline",
            )
            if getattr(metric, "OVERLAY_HEADLINE", True):
                overlay[value_key] = scalar
        if getattr(metric, "EMIT_DIAGNOSTICS", False):
            for diag_key, diag_val in result.metadata.items():
                if isinstance(diag_val, (int, float)) and not isinstance(diag_val, bool):
                    store.add_metric(
                        sequence_id="cam0",
                        frame_id="f1",
                        frame_timestamp_us=100,
                        metric_name=diag_key,
                        metric_value=float(diag_val),
                        semantic_class=mask_region,
                        metric_group="diagnostic",
                    )
    return overlay


def _rows(store):
    """Sortable snapshot of everything a MetricsStore holds, for equality comparison."""
    return sorted(
        (
            it.sequence_id,
            it.frame_id,
            it.frame_timestamp_us,
            it.metric_name,
            it.metric_value,
            it.semantic_class,
            it.metric_group,
        )
        for it in store._metrics
    )


class TestComputeImageMetricsEquivalence(unittest.TestCase):
    """Proves the extracted compute_image_metrics() reproduces the old inline behavior."""

    def _manager(self):
        # A camera metric (default label, overlay on) and a road metric (road mask, custom
        # label, overlay off, diagnostics on). Values are exact in float32 so equality is clean;
        # metadata mixes numeric + str + bool to exercise the diagnostic filter.
        camera = SimpleNamespace(OVERLAY_HEADLINE=True, EMIT_DIAGNOSTICS=False)
        road = SimpleNamespace(MASK_KIND="road", SEMANTIC_CLASS="road", OVERLAY_HEADLINE=False, EMIT_DIAGNOSTICS=True)
        return _FakeManager(
            [
                ("psnr", camera, _fake_result({"psnr": 30.0}, {"data_range": 1.0})),
                (
                    "ref",
                    road,
                    _fake_result(
                        {"ref": 0.5}, {"ref_precision": 0.25, "ref_recall": 0.75, "note": "ignore", "flag": True}
                    ),
                ),
            ]
        )

    def _run_both(self, *, camera_mask, road_mask):
        img = torch.zeros((3, 4, 4), dtype=torch.float32)
        store_ref = MetricsStore()
        overlay_ref = _inline_reference(
            self._manager(), store_ref, img, img, inv_camera_mask=camera_mask, per_class_valid_mask=road_mask
        )
        # Caller schedules metrics + assigns masks from MASK_KIND declarations, mirroring the
        # CLI dispatch: mask-based metrics are only scheduled when their mask is available.
        mgr = self._manager()
        names_to_run = []
        masks_per_metric = {}
        for name in mgr.list_metrics():
            metric = mgr.get_metric(name)
            kind = getattr(metric, "MASK_KIND", None)
            mask = road_mask if kind == "road" else camera_mask
            if kind is not None and mask is None:
                continue
            names_to_run.append(name)
            masks_per_metric[name] = mask
        records, overlay_new = compute_image_metrics(mgr, img, img, names_to_run, masks=masks_per_metric)
        store_new = MetricsStore()
        for r in records:
            store_new.add_metric(
                sequence_id="cam0",
                frame_id="f1",
                frame_timestamp_us=100,
                metric_name=r.metric_name,
                metric_value=r.metric_value,
                semantic_class=r.semantic_class,
                metric_group=r.metric_group,
            )
        return (store_ref, overlay_ref), (store_new, overlay_new)

    def test_identical_with_road_mask_present(self):
        cam = torch.ones((4, 4), dtype=torch.bool)
        road = torch.ones((4, 4), dtype=torch.bool)
        (store_ref, overlay_ref), (store_new, overlay_new) = self._run_both(camera_mask=cam, road_mask=road)
        # The refactor is behavior-preserving: identical store contents and overlay.
        self.assertEqual(_rows(store_ref), _rows(store_new))
        self.assertEqual(overlay_ref, overlay_new)
        # And the produced rows are exactly the expected labeled set (headline + diagnostics).
        self.assertEqual(
            _rows(store_new),
            sorted(
                [
                    ("cam0", "f1", 100, "psnr", 30.0, "static_camera_mask", "headline"),
                    ("cam0", "f1", 100, "ref", 0.5, "road", "headline"),
                    ("cam0", "f1", 100, "ref_precision", 0.25, "road", "diagnostic"),
                    ("cam0", "f1", 100, "ref_recall", 0.75, "road", "diagnostic"),
                ]
            ),
        )
        self.assertEqual(overlay_new, {"psnr": 30.0})  # ref sets OVERLAY_HEADLINE=False

    def test_identical_when_road_mask_absent(self):
        # road_mask=None -> the road metric is skipped identically by both code paths.
        cam = torch.ones((4, 4), dtype=torch.bool)
        (store_ref, overlay_ref), (store_new, overlay_new) = self._run_both(camera_mask=cam, road_mask=None)
        self.assertEqual(_rows(store_ref), _rows(store_new))
        self.assertEqual(overlay_ref, overlay_new)
        self.assertEqual({row[3] for row in _rows(store_new)}, {"psnr"})  # only the camera metric survived


class TestComputeImageMetricsWithRealRoadMetric(unittest.TestCase):
    """End-to-end unit coverage: a real road metric class runs through the real dispatch.

    Complements the fake-manager tests (which prove the dispatch logic) and the CLI
    bit-diff (which proves the full pipeline). This narrows the gap in between: a real
    RefMetric instance, real MetricManager, real compute_image_metrics -- with a
    deterministic small tensor + full road mask so the assertions are stable.
    """

    def test_ref_metric_emits_headline_plus_diagnostics_under_road_bucket(self):
        # Real RefMetric via the standard manager. 32x32 with a random pattern so
        # Canny actually finds edges (a flat image would produce empty edge sets).
        manager = create_image_comparison_metric_manager(["ref"], device="cpu")
        rng = np.random.default_rng(42)
        gt = torch.from_numpy(rng.random((3, 32, 32)).astype(np.float32))
        rendered = gt.clone()
        road_mask = torch.ones((32, 32), dtype=torch.bool)

        records, overlay = compute_image_metrics(manager, rendered, gt, ["ref"], masks={"ref": road_mask})

        # Real RefMetric emits one headline (ref_chamfer) + three diagnostic sub-scores.
        emitted = {r.metric_name for r in records}
        self.assertEqual(emitted, {"ref_chamfer", "ref_precision", "ref_recall", "ref_f1"})

        # Every record routes to the "road" semantic class (via RefMetric.SEMANTIC_CLASS).
        self.assertTrue(all(r.semantic_class == "road" for r in records), [r.semantic_class for r in records])

        # Headline vs diagnostic labelling matches the class declarations.
        by_name = {r.metric_name: r for r in records}
        self.assertEqual(by_name["ref_chamfer"].metric_group, "headline")
        for diag in ("ref_precision", "ref_recall", "ref_f1"):
            self.assertEqual(by_name[diag].metric_group, "diagnostic", f"{diag} should be diagnostic")

        # RefMetric sets OVERLAY_HEADLINE = False -> no overlay entry.
        self.assertEqual(overlay, {})

    def test_road_metric_skipped_when_not_scheduled(self):
        # A metric registered on the manager but not named in `metric_names` produces no
        # records -- matches the "road mask missing this frame -> not scheduled" pipeline
        # behaviour (the CLI omits the metric from the schedule when its mask is missing).
        manager = create_image_comparison_metric_manager(["ref"], device="cpu")
        gt = torch.zeros((3, 32, 32), dtype=torch.float32)

        records, overlay = compute_image_metrics(manager, gt, gt, [])

        self.assertEqual(records, [])
        self.assertEqual(overlay, {})


class TestMetricsStoreAddAndConvert(unittest.TestCase):
    def _store(self) -> MetricsStore:
        store = MetricsStore()
        # Two metrics share one row (same sequence/frame/timestamp/class).
        store.add_metric("cam0", "f1", "PSNR ", 30.0, semantic_class="road", frame_timestamp_us=100)
        store.add_metric("cam0", "f1", "ssim", 0.9, semantic_class="road", frame_timestamp_us=100)
        # A second frame has only psnr -> ssim column must be filled with None.
        store.add_metric("cam0", "f2", "psnr", 20.0, semantic_class="road", frame_timestamp_us=200)
        return store

    def test_add_metric_normalizes_name(self):
        store = MetricsStore()
        store.add_metric("cam0", "f1", "  PSNR  ", 30.0)
        tree = store.to_tree()
        self.assertIn("psnr", tree[None]["cam0"])  # lowercased and stripped

    def test_to_columns_groups_rows_and_fills_missing(self):
        columns = self._store().to_columns()
        for expected_col in ["image_sequence_id", "frame_id", "frame_timestamp", "semantic_class", "psnr", "ssim"]:
            self.assertIn(expected_col, columns)
        # Two distinct rows (f1 and f2). Row order is unspecified, so index by frame_id.
        self.assertEqual(len(columns["frame_id"]), 2)
        by_frame = {columns["frame_id"][i]: i for i in range(2)}
        i1, i2 = by_frame["f1"], by_frame["f2"]
        self.assertEqual(columns["psnr"][i1], 30.0)
        self.assertEqual(columns["ssim"][i1], 0.9)
        self.assertEqual(columns["psnr"][i2], 20.0)
        self.assertIsNone(columns["ssim"][i2])  # missing metric -> None

    def test_to_tree_structure(self):
        tree = self._store().to_tree()
        self.assertEqual(tree["road"]["cam0"]["psnr"], [30.0, 20.0])
        self.assertEqual(tree["road"]["cam0"]["ssim"], [0.9])

    def test_empty_store(self):
        store = MetricsStore()
        self.assertEqual(store.to_tree(), {})
        columns = store.to_columns()
        self.assertEqual(columns["frame_id"], [])


class TestMetricsStoreAggregation(unittest.TestCase):
    def _store(self) -> MetricsStore:
        store = MetricsStore()
        # cam0 psnr values: [30, 20]; cam1 psnr values: [10].
        store.add_metric("cam0", "f1", "psnr", 30.0, semantic_class="road")
        store.add_metric("cam0", "f2", "psnr", 20.0, semantic_class="road")
        store.add_metric("cam1", "f1", "psnr", 10.0, semantic_class="road")
        return store

    def test_per_sequence_statistics(self):
        per_seq = MetricsStore.calculate_aggregated_metrics(self._store().to_tree())
        cam0 = per_seq["road"]["cam0"]["psnr"]
        self.assertAlmostEqual(cam0["avg"], 25.0)
        self.assertEqual(cam0["min"], 20.0)
        self.assertEqual(cam0["max"], 30.0)
        self.assertAlmostEqual(cam0["std"], 5.0)  # population std of [30, 20]
        self.assertEqual(cam0["cnt"], 2)

    def test_non_finite_values_are_dropped_and_logged(self):
        # NaN/inf frames must be excluded from the aggregate (cnt counts finite only)
        # AND the drop must be visible in the log -- a silent exclusion would make
        # corrupted data look like a clean run with a quietly smaller cnt.
        store = MetricsStore()
        store.add_metric("cam0", "f1", "psnr", 30.0, semantic_class="road")
        store.add_metric("cam0", "f2", "psnr", float("nan"), semantic_class="road")
        store.add_metric("cam0", "f3", "psnr", float("inf"), semantic_class="road")

        with self.assertLogs("nurec.benchmark.eval_rendering_metrics", level="WARNING") as captured:
            per_seq = MetricsStore.calculate_aggregated_metrics(store.to_tree())

        cam0 = per_seq["road"]["cam0"]["psnr"]
        self.assertAlmostEqual(cam0["avg"], 30.0)  # only the finite value contributes
        self.assertEqual(cam0["cnt"], 1)
        self.assertTrue(any("dropped 2 non-finite of 3" in msg for msg in captured.output), captured.output)

    def test_cross_sequence_statistics(self):
        per_seq = MetricsStore.calculate_aggregated_metrics(self._store().to_tree())
        cross = MetricsStore.calculate_cross_sequence_metrics(per_seq)["road"]["psnr"]
        # avgs across sequences: [25 (cam0), 10 (cam1)].
        self.assertAlmostEqual(cross["avg"], 17.5)
        self.assertEqual(cross["min"], 10.0)  # min of per-sequence mins (20, 10)
        self.assertEqual(cross["max"], 30.0)  # max of per-sequence maxes (30, 10)
        self.assertAlmostEqual(cross["min_of_avg"], 10.0)
        self.assertAlmostEqual(cross["max_of_avg"], 25.0)
        self.assertAlmostEqual(cross["std"], 7.5)  # population std of [25, 10]
        self.assertEqual(cross["cnt"], 3)  # total frames across sequences


class TestMetricsStoreSerialization(unittest.TestCase):
    def _store(self) -> MetricsStore:
        store = MetricsStore()
        store.add_metric("cam0", "f1", "psnr", 30.0, semantic_class="road", frame_timestamp_us=100)
        store.add_metric("cam0", "f2", "psnr", 20.0, semantic_class="road", frame_timestamp_us=200)
        return store

    def test_save_yaml_statistics_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.yaml")
            self._store().save_yaml(path)
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        self.assertIn("per_sequence_metrics", data)
        self.assertIn("cross_sequence_metrics", data)
        self.assertNotIn("per_frame_metrics", data)  # dropped when statistics_only=True
        self.assertAlmostEqual(data["per_sequence_metrics"]["road"]["cam0"]["psnr"]["avg"], 25.0)

    def test_save_yaml_with_per_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.yaml")
            self._store().save_yaml(path, statistics_only=False)
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        self.assertIn("per_frame_metrics", data)
        self.assertEqual(data["per_frame_metrics"]["road"]["cam0"]["psnr"], [30.0, 20.0])

    def test_save_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.csv")
            self._store().save_csv(path)
            df = pd.read_csv(path)
        self.assertEqual(len(df), 2)
        self.assertIn("psnr", df.columns)
        self.assertEqual(set(df["frame_id"]), {"f1", "f2"})

    def test_save_parquet_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.parquet")
            self._store().save_parquet(path)
            df = pd.read_parquet(path)
        self.assertEqual(len(df), 2)
        self.assertIn("psnr", df.columns)


if __name__ == "__main__":
    unittest.main()


class TestRequireObjectMetrics(unittest.TestCase):
    """The object-level pass swallows missing videos and per-sequence errors; the flag makes that fatal."""

    def _invoke(self, scored: int, *extra: str):
        from unittest.mock import patch

        from click.testing import CliRunner

        from nurec.benchmark import eval_rendering_metrics as erm

        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "renders", "camera_front_wide_120fov"))
            open(os.path.join(root, "renders", "camera_front_wide_120fov", "video.mp4"), "w").close()
            dataset = os.path.join(root, "clip.json")
            open(dataset, "w").close()
            with patch.object(erm, "compute_object_level_metrics", return_value=scored):
                return CliRunner().invoke(
                    erm.eval_rendering_metrics,
                    [
                        "--render-dir",
                        os.path.join(root, "renders"),
                        "--output-dir",
                        os.path.join(root, "out"),
                        # Required by the CLI even though the object-level path does not read it.
                        "--gt-dir",
                        os.path.join(root, "unused_gt"),
                        "--object-level-metrics",
                        "--dataset-path",
                        dataset,
                        *extra,
                    ],
                )

    def test_no_crops_fails_when_required(self):
        result = self._invoke(0, "--require-object-metrics")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("no crops were scored", result.output)

    def test_no_crops_still_exits_zero_without_the_flag(self):
        result = self._invoke(0)
        self.assertEqual(result.exit_code, 0, result.output)

    def test_scored_crops_pass_when_required(self):
        result = self._invoke(28, "--require-object-metrics")
        self.assertEqual(result.exit_code, 0, result.output)
