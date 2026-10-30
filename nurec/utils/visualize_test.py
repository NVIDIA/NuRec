# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nurec/utils/visualize.py — sem2img, build_semantics_class_colors, create_semantics_legend."""

import unittest

import numpy as np

from nurec.utils.visualize import (
    SEMANTICS_COLOR_LUT_DEFAULT_SIZE,
    build_semantics_class_color_lut,
    build_semantics_class_colors,
    create_semantics_legend,
    sem2img,
)


class TestSem2img(unittest.TestCase):
    def test_basic_mapping(self):
        color_remap = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        labels = np.array([[0, 1], [2, 0]], dtype=np.int64)
        out = sem2img(labels, color_remap)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(out[0, 1], [0, 255, 0])
        np.testing.assert_array_equal(out[1, 0], [0, 0, 255])

    def test_output_dtype_is_uint8(self):
        remap = np.array([[128, 64, 32]], dtype=np.uint8)
        labels = np.zeros((4, 4), dtype=np.int64)
        out = sem2img(labels, remap)
        self.assertEqual(out.dtype, np.uint8)


class TestBuildSemanticsClassColorLut(unittest.TestCase):
    def test_zero_returns_empty(self):
        lut = build_semantics_class_color_lut(0)
        self.assertEqual(lut.shape, (0, 3))
        self.assertEqual(lut.dtype, np.uint8)

    def test_negative_returns_empty(self):
        lut = build_semantics_class_color_lut(-5)
        self.assertEqual(lut.shape, (0, 3))

    def test_shape_and_dtype(self):
        lut = build_semantics_class_color_lut(10)
        self.assertEqual(lut.shape, (10, 3))
        self.assertEqual(lut.dtype, np.uint8)

    def test_values_in_range(self):
        lut = build_semantics_class_color_lut(256)
        self.assertTrue((lut >= 0).all())
        self.assertTrue((lut <= 255).all())

    def test_deterministic(self):
        a = build_semantics_class_color_lut(50)
        b = build_semantics_class_color_lut(50)
        np.testing.assert_array_equal(a, b)

    def test_consecutive_classes_visually_distinct(self):
        """Consecutive class ids should have different colors (golden-ratio hue spacing)."""
        lut = build_semantics_class_color_lut(8)
        for i in range(len(lut) - 1):
            self.assertFalse(
                np.array_equal(lut[i], lut[i + 1]),
                f"Classes {i} and {i + 1} have the same color {lut[i]}",
            )

    def test_default_lut_size_constant(self):
        lut = build_semantics_class_color_lut(SEMANTICS_COLOR_LUT_DEFAULT_SIZE)
        self.assertEqual(lut.shape[0], SEMANTICS_COLOR_LUT_DEFAULT_SIZE)


class TestBuildSemanticsClassColors(unittest.TestCase):
    def test_no_artifact_colors_uses_generated_lut(self):
        lut = build_semantics_class_colors(4, None)
        expected = build_semantics_class_color_lut(4)
        np.testing.assert_array_equal(lut, expected)

    def test_empty_artifact_colors_uses_generated_lut(self):
        lut = build_semantics_class_colors(4, np.empty((0, 3), dtype=np.uint8))
        expected = build_semantics_class_color_lut(4)
        np.testing.assert_array_equal(lut, expected)

    def test_artifact_colors_fill_first_rows(self):
        artifact_colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        lut = build_semantics_class_colors(4, artifact_colors)
        self.assertEqual(lut.shape, (4, 3))
        np.testing.assert_array_equal(lut[0], [10, 20, 30])
        np.testing.assert_array_equal(lut[1], [40, 50, 60])
        # Remaining rows are generated, so they differ from artifact zeros.
        generated = build_semantics_class_color_lut(4)
        np.testing.assert_array_equal(lut[2], generated[2])

    def test_artifact_colors_longer_than_lut_clipped(self):
        artifact_colors = np.arange(15, dtype=np.uint8).reshape(5, 3)
        lut = build_semantics_class_colors(3, artifact_colors)
        self.assertEqual(lut.shape, (3, 3))
        np.testing.assert_array_equal(lut[0], artifact_colors[0])

    def test_invalid_shape_artifact_ignored(self):
        bad = np.array([255, 0, 0], dtype=np.uint8)  # (3,) not (N, 3)
        lut = build_semantics_class_colors(2, bad)
        expected = build_semantics_class_color_lut(2)
        np.testing.assert_array_equal(lut, expected)


class TestCreateSemanticsLegend(unittest.TestCase):
    def test_zero_classes_returns_placeholder(self):
        colors = np.empty((0, 3), dtype=np.uint8)
        img = create_semantics_legend(colors)
        self.assertEqual(img.shape, (1, 1, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_one_class_produces_image(self):
        colors = np.array([[255, 0, 0]], dtype=np.uint8)
        img = create_semantics_legend(colors)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)

    def test_many_classes_height_grows(self):
        c1 = create_semantics_legend(np.zeros((1, 3), dtype=np.uint8))
        c4 = create_semantics_legend(np.zeros((4, 3), dtype=np.uint8))
        self.assertGreater(c4.shape[0], c1.shape[0])

    def test_with_class_names(self):
        colors = np.array([[0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        img = create_semantics_legend(colors, class_names={0: "bg", 1: "car"})
        self.assertEqual(img.ndim, 3)
        # Image with names should be wider (text column added).
        img_no_names = create_semantics_legend(colors, class_names=None)
        self.assertGreaterEqual(img.shape[1], img_no_names.shape[1])

    def test_with_sparse_class_names(self):
        """Sparse id->name mapping: ids without a name fall back to numeric labels."""
        colors = np.array([[0, 255, 0], [128, 128, 128], [0, 0, 255]], dtype=np.uint8)
        img = create_semantics_legend(colors, class_names={0: "bg", 2: "car"})
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)

    def test_256_classes_no_crash(self):
        colors = build_semantics_class_color_lut(256)
        img = create_semantics_legend(colors)
        self.assertEqual(img.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
