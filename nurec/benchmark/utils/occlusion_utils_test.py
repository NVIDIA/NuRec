# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for occlusion utilities.

Covers the pure-numpy geometry primitives, the class-aware shape proxies, the
occlusion-ratio computation and the per-frame wrapper, plus the ``should_skip_object``
filtering contract. Also covers the ``camera_inside_object`` helper and the
per-object ``camera_inside`` flag emitted by the frame wrapper (a camera inside
an *occluder*, not the object itself), and that the GT and novel viewpoints can
diverge for the same object.

The sampling-based occlusion functions use ``np.random``; tests seed the RNG and
assert on physically-extreme geometries (fully occluded vs. not occluded) with a
tolerance band so they are robust to the sampling count.
"""

import unittest

import numpy as np

from nurec.benchmark.utils.occlusion_utils import (
    DEFAULT_PROXY_CONFIG,
    ProxyConfig,
    _proxy_kind,
    build_class_proxy,
    camera_inside_object,
    compute_occlusion_ratio,
    compute_occlusion_ratios_for_frame,
    obb_from_corners,
    ray_ellipsoid_entry,
    ray_obb_entry,
    should_skip_object,
)


def make_box_corners(
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    yaw: float = 0.0,
) -> np.ndarray:
    """Build the 8 corners of an oriented box (yaw about the world-up/z axis)."""
    hx, hy, hz = half_extents
    local = np.array(
        [[sx * hx, sy * hy, sz * hz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    if yaw:
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = local @ rot.T
    return local + np.asarray(center, dtype=np.float64)


class TestShouldSkipObject(unittest.TestCase):
    """The occlusion-threshold filtering contract (pure, deterministic)."""

    def test_below_threshold_not_skipped(self) -> None:
        skip, reason = should_skip_object(0.1, 0.2, occlusion_threshold=0.3)
        self.assertFalse(skip)
        self.assertEqual(reason, "")

    def test_gt_only_above_threshold(self) -> None:
        skip, reason = should_skip_object(0.9, 0.1, occlusion_threshold=0.3)
        self.assertTrue(skip)
        self.assertEqual(reason, "gt_occluded")

    def test_novel_only_above_threshold(self) -> None:
        skip, reason = should_skip_object(0.1, 0.9, occlusion_threshold=0.3)
        self.assertTrue(skip)
        self.assertEqual(reason, "novel_occluded")

    def test_both_above_threshold(self) -> None:
        skip, reason = should_skip_object(0.8, 0.9, occlusion_threshold=0.3)
        self.assertTrue(skip)
        self.assertEqual(reason, "both_occluded")

    def test_exactly_at_threshold_not_skipped(self) -> None:
        # Filtering uses a strict '>' so a ratio equal to the threshold is kept.
        skip, reason = should_skip_object(0.3, 0.3, occlusion_threshold=0.3)
        self.assertFalse(skip)
        self.assertEqual(reason, "")


class TestProxyClassification(unittest.TestCase):
    """Exact class-name -> proxy kind mapping (deterministic)."""

    def test_known_classes_map_to_proxies(self) -> None:
        self.assertEqual(_proxy_kind("automobile"), "car")
        self.assertEqual(_proxy_kind("bus"), "box")
        self.assertEqual(_proxy_kind("heavy_truck"), "box")
        self.assertEqual(_proxy_kind("rider"), "rider")
        self.assertEqual(_proxy_kind("person"), "person")

    def test_matching_is_case_insensitive(self) -> None:
        self.assertEqual(_proxy_kind("Automobile"), "car")
        self.assertEqual(_proxy_kind("  RIDER  "), "rider")

    def test_unknown_or_unmodeled_classes_map_to_bbox(self) -> None:
        # Classes present in the data but without a shape model use the raw bbox.
        self.assertEqual(_proxy_kind("stroller"), "bbox")
        self.assertEqual(_proxy_kind("protruding_object"), "bbox")
        # Unseen classes and None also fall back to the raw bbox.
        self.assertEqual(_proxy_kind("traffic_cone"), "bbox")
        self.assertEqual(_proxy_kind(None), "bbox")

    def test_matching_is_exact_not_substring(self) -> None:
        # Near-miss names must NOT match a proxy (no substring/keyword heuristics).
        self.assertEqual(_proxy_kind("sedan"), "bbox")
        self.assertEqual(_proxy_kind("cyclist"), "bbox")
        self.assertEqual(_proxy_kind("pedestrian"), "bbox")
        self.assertEqual(_proxy_kind("motorcycle"), "bbox")


class TestGeometryPrimitives(unittest.TestCase):
    """OBB recovery and ray-entry primitives (deterministic)."""

    def test_obb_from_corners_recovers_box(self) -> None:
        center = (1.0, 2.0, 3.0)
        half = (2.0, 1.0, 0.5)
        corners = make_box_corners(center, half)
        rec_center, rec_rot, rec_half = obb_from_corners(corners)
        np.testing.assert_allclose(rec_center, center, atol=1e-9)
        # Axis order is eigenvector-dependent -> compare as sets.
        np.testing.assert_allclose(np.sort(rec_half), np.sort(half), atol=1e-9)
        # Rotation must be orthonormal.
        np.testing.assert_allclose(rec_rot @ rec_rot.T, np.eye(3), atol=1e-9)

    def test_ray_obb_entry_hit_from_outside(self) -> None:
        center = np.array([10.0, 0.0, 0.0])
        rot = np.eye(3)
        half = np.array([1.0, 1.0, 1.0])
        origins = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        t_entry, hit = ray_obb_entry(origins, directions, center, rot, half)
        self.assertTrue(bool(hit[0]))
        self.assertAlmostEqual(float(t_entry[0]), 9.0, places=6)

    def test_ray_obb_entry_miss_when_pointing_away(self) -> None:
        center = np.array([10.0, 0.0, 0.0])
        rot = np.eye(3)
        half = np.array([1.0, 1.0, 1.0])
        origins = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[-1.0, 0.0, 0.0]])
        _, hit = ray_obb_entry(origins, directions, center, rot, half)
        self.assertFalse(bool(hit[0]))

    def test_ray_obb_entry_origin_inside_clamps_to_zero(self) -> None:
        center = np.array([10.0, 0.0, 0.0])
        rot = np.eye(3)
        half = np.array([1.0, 1.0, 1.0])
        origins = np.array([[10.0, 0.0, 0.0]])  # inside the box
        directions = np.array([[1.0, 0.0, 0.0]])
        t_entry, hit = ray_obb_entry(origins, directions, center, rot, half)
        self.assertTrue(bool(hit[0]))
        self.assertEqual(float(t_entry[0]), 0.0)

    def test_ray_ellipsoid_entry_hit_from_outside(self) -> None:
        center = np.array([10.0, 0.0, 0.0])
        rot = np.eye(3)
        semi = np.array([1.0, 1.0, 1.0])  # unit sphere
        origins = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        t_entry, hit = ray_ellipsoid_entry(origins, directions, center, rot, semi)
        self.assertTrue(bool(hit[0]))
        self.assertAlmostEqual(float(t_entry[0]), 9.0, places=6)

    def test_ray_ellipsoid_entry_origin_inside_clamps_to_zero(self) -> None:
        center = np.array([10.0, 0.0, 0.0])
        rot = np.eye(3)
        semi = np.array([1.0, 1.0, 1.0])
        origins = np.array([[10.0, 0.0, 0.0]])  # center of the sphere
        directions = np.array([[1.0, 0.0, 0.0]])
        t_entry, hit = ray_ellipsoid_entry(origins, directions, center, rot, semi)
        self.assertTrue(bool(hit[0]))
        self.assertEqual(float(t_entry[0]), 0.0)


class TestBuildClassProxy(unittest.TestCase):
    """Per-class proxy construction (deterministic)."""

    def setUp(self) -> None:
        # Distinct extents so vertical/longitudinal/lateral axes are unambiguous.
        self.half = (2.0, 1.0, 0.75)
        self.corners = make_box_corners((0.0, 0.0, 0.0), self.half)

    def test_bus_is_shrunk_box(self) -> None:
        proxy = build_class_proxy(self.corners, "bus")
        self.assertEqual(proxy["kind"], "box")
        expected = np.sort(np.array(self.half) * DEFAULT_PROXY_CONFIG.bus_margin)
        np.testing.assert_allclose(np.sort(proxy["half"]), expected, atol=1e-9)

    def test_car_is_ellipsoid_clipped_to_box(self) -> None:
        proxy = build_class_proxy(self.corners, "automobile")
        self.assertEqual(proxy["kind"], "car")
        self.assertIsNotNone(proxy["semi"])
        self.assertIsNotNone(proxy["half"])
        # Horizontals scaled by 1.5, vertical (0.75) by 1.0 -> {3.0, 1.5, 0.75}.
        expected = np.sort([2.0 * 1.5, 1.0 * 1.5, 0.75 * 1.0])
        np.testing.assert_allclose(np.sort(proxy["semi"]), expected, atol=1e-9)

    def test_rider_proxy_is_axis_aligned(self) -> None:
        proxy = build_class_proxy(self.corners, "rider")
        self.assertEqual(proxy["kind"], "ellipsoid")
        # Rider uses a world-axis-aligned proxy (rotation == identity).
        np.testing.assert_allclose(proxy["rotation"], np.eye(3), atol=1e-9)
        # Equal horizontal radii = width * rider_widen; vertical = height half.
        rh = 1.0 * DEFAULT_PROXY_CONFIG.rider_widen
        np.testing.assert_allclose(np.sort(proxy["semi"]), np.sort([rh, rh, 0.75]), atol=1e-9)

    def test_person_is_widened_ellipsoid(self) -> None:
        proxy = build_class_proxy(self.corners, "person")
        self.assertEqual(proxy["kind"], "ellipsoid")
        # Both horizontals scaled by 0.5 * person_widen; vertical kept at full half.
        horiz = 0.5 * DEFAULT_PROXY_CONFIG.person_widen
        expected = np.sort([2.0 * horiz, 1.0 * horiz, 0.75])
        np.testing.assert_allclose(np.sort(proxy["semi"]), expected, atol=1e-9)

    def test_unknown_class_uses_raw_bbox(self) -> None:
        # Unmodeled/unseen classes -> full oriented box, no shrink, no ellipsoid.
        for cls in ("traffic_cone", "stroller", "protruding_object", None):
            proxy = build_class_proxy(self.corners, cls)
            self.assertEqual(proxy["kind"], "box", msg=f"class={cls}")
            self.assertIsNone(proxy["semi"], msg=f"class={cls}")
            # Half-extents are the full box (unlike bus/heavy_truck which shrink).
            np.testing.assert_allclose(np.sort(proxy["half"]), np.sort(self.half), atol=1e-9)

    def test_custom_config_scales_car_proxy(self) -> None:
        default_proxy = build_class_proxy(self.corners, "automobile")
        big_proxy = build_class_proxy(self.corners, "automobile", ProxyConfig(car_lon_scale=3.0))
        self.assertGreater(float(np.max(big_proxy["semi"])), float(np.max(default_proxy["semi"])))


class TestComputeOcclusionRatio(unittest.TestCase):
    """Occlusion-ratio computation (seeded RNG + extreme geometries)."""

    def setUp(self) -> None:
        np.random.seed(0)
        self.camera = np.array([0.0, 0.0, 1.5])
        # Target far along +x; box class -> flat, predictable geometry.
        self.target = make_box_corners((20.0, 0.0, 1.0), (2.0, 1.0, 0.75))

    def test_no_other_objects_returns_zero(self) -> None:
        ratio = compute_occlusion_ratio(self.target, [], self.camera, target_class="truck")
        self.assertEqual(ratio, 0.0)

    def test_large_occluder_in_front_is_fully_occluded(self) -> None:
        occluder = make_box_corners((10.0, 0.0, 1.0), (1.0, 8.0, 8.0))
        ratio = compute_occlusion_ratio(
            self.target,
            [occluder],
            self.camera,
            target_class="truck",
            other_classes=["truck"],
        )
        self.assertGreaterEqual(ratio, 0.9)

    def test_occluder_behind_target_not_occluded(self) -> None:
        occluder = make_box_corners((24.0, 0.0, 1.0), (1.0, 8.0, 8.0))
        ratio = compute_occlusion_ratio(
            self.target,
            [occluder],
            self.camera,
            target_class="truck",
            other_classes=["truck"],
        )
        self.assertLessEqual(ratio, 0.1)

    def test_occluder_off_to_side_not_occluded(self) -> None:
        occluder = make_box_corners((10.0, 30.0, 1.0), (1.0, 2.0, 2.0))
        ratio = compute_occlusion_ratio(
            self.target,
            [occluder],
            self.camera,
            target_class="truck",
            other_classes=["truck"],
        )
        self.assertLessEqual(ratio, 0.1)


class TestComputeOcclusionRatiosForFrame(unittest.TestCase):
    """Per-frame wrapper: structure + occlusion ratios (camera_inside not asserted)."""

    def setUp(self) -> None:
        np.random.seed(0)
        self.camera = np.array([0.0, 0.0, 1.5])

    def test_returns_three_equal_length_lists(self) -> None:
        boxes = [
            make_box_corners((20.0, 0.0, 1.0), (2.0, 1.0, 0.75)),
            make_box_corners((10.0, 0.0, 1.0), (1.0, 8.0, 8.0)),
        ]
        result = compute_occlusion_ratios_for_frame(boxes, self.camera, self.camera, classes=["truck", "truck"])
        self.assertEqual(len(result), 3)
        occ_gt, occ_novel, camera_inside = result
        self.assertEqual(len(occ_gt), 2)
        self.assertEqual(len(occ_novel), 2)
        self.assertEqual(len(camera_inside), 2)

    def test_single_object_has_zero_occlusion(self) -> None:
        boxes = [make_box_corners((20.0, 0.0, 1.0), (2.0, 1.0, 0.75))]
        occ_gt, occ_novel, _ = compute_occlusion_ratios_for_frame(boxes, self.camera, self.camera, classes=["truck"])
        self.assertEqual(occ_gt, [0.0])
        self.assertEqual(occ_novel, [0.0])

    def test_front_object_occludes_back_object(self) -> None:
        # boxes[0] is the far target; boxes[1] is a large near occluder.
        boxes = [
            make_box_corners((20.0, 0.0, 1.0), (2.0, 1.0, 0.75)),
            make_box_corners((10.0, 0.0, 1.0), (1.0, 8.0, 8.0)),
        ]
        occ_gt, occ_novel, _ = compute_occlusion_ratios_for_frame(
            boxes, self.camera, self.camera, classes=["truck", "truck"]
        )
        # The far object is heavily occluded by the near one; the near one,
        # having nothing in front of it, is essentially unoccluded.
        self.assertGreaterEqual(occ_gt[0], 0.9)
        self.assertLessEqual(occ_gt[1], 0.1)
        self.assertGreaterEqual(occ_novel[0], 0.9)
        self.assertLessEqual(occ_novel[1], 0.1)

    def test_gt_and_novel_views_diverge(self) -> None:
        # Same object, two different cameras: the occluder blocks the target from
        # the GT camera but not from the novel one, so the two per-view ratios
        # must land on opposite ends of the range (guards that each camera arg is
        # threaded to its own computation).
        boxes = [
            make_box_corners((20.0, 0.0, 1.0), (2.0, 1.0, 0.75)),  # target
            make_box_corners((10.0, 0.0, 1.0), (1.0, 8.0, 8.0)),  # occluder between
        ]
        cam_gt = np.array([0.0, 0.0, 1.5])  # behind the occluder -> target hidden
        cam_novel = np.array([0.0, 40.0, 1.5])  # off to the side -> clear view
        occ_gt, occ_novel, _ = compute_occlusion_ratios_for_frame(boxes, cam_gt, cam_novel, classes=["truck", "truck"])
        self.assertGreaterEqual(occ_gt[0], 0.9)
        self.assertLessEqual(occ_novel[0], 0.1)


class TestCameraInsideObject(unittest.TestCase):
    """The camera-inside helper and the per-object 'inside an occluder' flag."""

    def setUp(self) -> None:
        np.random.seed(0)

    def test_inside_box_proxy_is_true(self) -> None:
        box = make_box_corners((0.0, 0.0, 1.5), (3.0, 3.0, 3.0))
        self.assertTrue(camera_inside_object(box, np.array([0.0, 0.0, 1.5]), "truck"))

    def test_outside_proxy_is_false(self) -> None:
        box = make_box_corners((0.0, 0.0, 1.5), (3.0, 3.0, 3.0))
        self.assertFalse(camera_inside_object(box, np.array([50.0, 0.0, 1.5]), "truck"))

    def test_respects_class_proxy(self) -> None:
        # A point inside the raw box but outside the shrunk person ellipsoid:
        # box half = (2, 1, 0.75); person horizontal semi = half * 0.5 * 1.25.
        corners = make_box_corners((0.0, 0.0, 0.0), (2.0, 1.0, 0.75))
        point = np.array([1.8, 0.0, 0.0])
        self.assertTrue(camera_inside_object(corners, point, "truck"))
        self.assertFalse(camera_inside_object(corners, point, "person"))

    def test_frame_flag_marks_occluder_not_self(self) -> None:
        # Camera sits inside obj 1 (the occluder). Obj 0 must be flagged (its
        # occlusion is driven by the ambiguous camera position); obj 1 must NOT
        # be flagged on the basis of containing the camera itself.
        boxes = [
            make_box_corners((20.0, 0.0, 1.0), (2.0, 1.0, 0.75)),  # obj 0
            make_box_corners((0.0, 0.0, 1.5), (3.0, 3.0, 3.0)),  # obj 1 holds camera
        ]
        camera = np.array([0.0, 0.0, 1.5])
        _, _, camera_inside = compute_occlusion_ratios_for_frame(boxes, camera, camera, classes=["truck", "truck"])
        self.assertTrue(camera_inside[0])
        self.assertFalse(camera_inside[1])

    def test_frame_self_containment_not_flagged(self) -> None:
        # A single object containing the camera has no occluders, so it is not flagged.
        boxes = [make_box_corners((0.0, 0.0, 1.5), (3.0, 3.0, 3.0))]
        camera = np.array([0.0, 0.0, 1.5])
        _, _, camera_inside = compute_occlusion_ratios_for_frame(boxes, camera, camera, classes=["truck"])
        self.assertEqual(camera_inside, [False])

    def test_frame_no_camera_inside_all_false(self) -> None:
        # Camera outside every object -> no object is flagged.
        boxes = [
            make_box_corners((20.0, 0.0, 1.0), (2.0, 1.0, 0.75)),
            make_box_corners((-20.0, 0.0, 1.0), (2.0, 1.0, 0.75)),
        ]
        camera = np.array([0.0, 0.0, 1.5])
        _, _, camera_inside = compute_occlusion_ratios_for_frame(boxes, camera, camera, classes=["truck", "truck"])
        self.assertEqual(camera_inside, [False, False])


if __name__ == "__main__":
    unittest.main()
