# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for overlay_bounding_boxes (nurec/render/bbox_overlay.py).

These tests mock the ncore CameraModel projection to exercise the drawing
logic deterministically, without requiring a trained model or real camera data.
"""

import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nurec.render.actors import ActorsSnapshot
from nurec.render.bbox_overlay import overlay_bounding_boxes


_BOX_COLOR = (0, 255, 0)
_IMG_H, _IMG_W = 200, 200


def _blank_image() -> np.ndarray:
    return np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)


def _one_actor_snapshot() -> ActorsSnapshot:
    """Single 2×2×2 actor at world position (0, 0, 5), identity rotation."""
    actor_poses = torch.tensor(
        [[[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    cuboid_dims = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)
    return ActorsSnapshot(actor_ids=["track_0"], actor_poses=actor_poses, cuboid_dims=cuboid_dims)


def _mock_proj_result(image_points: torch.Tensor, valid_indices=None) -> MagicMock:
    """Build a mock world_points_to_image_points_shutter_pose result."""
    result = MagicMock()
    result.image_points = image_points
    result.valid_indices = valid_indices
    result.valid_flag = None  # must be explicitly None; a truthy MagicMock would break the check
    return result


class TestOverlayBoundingBoxes(unittest.TestCase):
    """Tests for the overlay_bounding_boxes standalone library function."""

    @patch("nurec.render.bbox_overlay.camera_model_from_parameters")
    def test_no_actors_returns_input_unchanged(self, mock_cm_class):
        """With zero actors the image is returned unchanged and CameraModel is never called."""
        img = _blank_image()
        img[50:60, 50:60] = (255, 0, 0)

        result = overlay_bounding_boxes(
            img,
            MagicMock(),  # camera_model_parameters — unused for empty snapshot
            torch.eye(4),
            ActorsSnapshot.empty(),
        )

        np.testing.assert_array_equal(result, img)
        mock_cm_class.assert_not_called()

    @patch("nurec.render.bbox_overlay.camera_model_from_parameters")
    def test_visible_actor_draws_only_green_pixels(self, mock_cm_class):
        """With one actor, all changed pixels must be the box color."""
        # edge_segments=1: 12 edges × 2 sample points = 24 points per actor.
        # All 24 points → two clusters at (80,80) and (120,120) to produce visible green lines.
        n_pts = 12 * 2
        pts = torch.zeros((n_pts, 2), dtype=torch.float32)
        pts[: n_pts // 2] = torch.tensor([80.0, 80.0])
        pts[n_pts // 2 :] = torch.tensor([120.0, 120.0])

        mock_model = MagicMock()
        mock_model.world_points_to_image_points_shutter_pose.return_value = _mock_proj_result(pts)
        mock_cm_class.return_value = mock_model

        img = _blank_image()
        result = overlay_bounding_boxes(
            img,
            MagicMock(),
            torch.eye(4),
            _one_actor_snapshot(),
            box_color_rgb=_BOX_COLOR,
            edge_segments=1,
        )

        changed_mask = np.any(result != img, axis=2)
        n_changed = int(changed_mask.sum())
        self.assertGreater(n_changed, 0, "Expected at least one pixel to change")

        changed_pixels = result[changed_mask]
        not_green = ~np.all(changed_pixels == np.array(_BOX_COLOR, dtype=np.uint8)[None], axis=1)
        self.assertEqual(
            int(not_green.sum()),
            0,
            f"All changed pixels must be {_BOX_COLOR}; got unexpected: "
            f"{np.unique(changed_pixels[not_green], axis=0).tolist()}",
        )

    @patch("nurec.render.bbox_overlay.camera_model_from_parameters")
    def test_all_invalid_projections_no_change(self, mock_cm_class):
        """When no projected points are valid no lines are drawn and the image is unchanged."""
        n_pts = 12 * 2  # edge_segments=1: 12 edges × 2 sample points
        pts = torch.full((n_pts, 2), 5000.0)  # far outside image bounds
        empty_valid = torch.zeros((0,), dtype=torch.long)

        mock_model = MagicMock()
        mock_model.world_points_to_image_points_shutter_pose.return_value = _mock_proj_result(
            pts, valid_indices=empty_valid
        )
        mock_cm_class.return_value = mock_model

        img = _blank_image()
        result = overlay_bounding_boxes(
            img,
            MagicMock(),
            torch.eye(4),
            _one_actor_snapshot(),
            edge_segments=1,
        )

        np.testing.assert_array_equal(result, img)

    @patch("nurec.render.bbox_overlay.camera_model_from_parameters")
    def test_out_of_bounds_corners_no_crash(self, mock_cm_class):
        """Sample points that project far outside the image must not crash and leave the image unchanged."""
        n_pts = 12 * 2  # edge_segments=1
        pts = torch.full((n_pts, 2), 5000.0)  # entirely outside [0,199]×[0,199]

        mock_model = MagicMock()
        mock_model.world_points_to_image_points_shutter_pose.return_value = _mock_proj_result(pts)
        mock_cm_class.return_value = mock_model

        img = _blank_image()
        result = overlay_bounding_boxes(
            img,
            MagicMock(),
            torch.eye(4),
            _one_actor_snapshot(),
            edge_segments=1,
        )

        self.assertEqual(result.shape, img.shape)
        np.testing.assert_array_equal(result, img)

    @patch("nurec.render.bbox_overlay.camera_model_from_parameters")
    def test_default_edge_segments_draws_only_box_color(self, mock_cm_class):
        """Default edge_segments=16 (segmented path): changed pixels must all be box color."""
        n_segs = 16
        n_pts = 1 * 12 * (n_segs + 1)  # 204 sample points for 1 actor
        # Spread points along diagonal so consecutive segment endpoints differ → lines are drawn.
        pts = torch.zeros((n_pts, 2), dtype=torch.float32)
        for k in range(n_pts):
            v = 20.0 + (k / max(n_pts - 1, 1)) * 150.0  # [20, 170]
            pts[k, 0] = v
            pts[k, 1] = v

        mock_model = MagicMock()
        mock_model.world_points_to_image_points_shutter_pose.return_value = _mock_proj_result(pts)
        mock_cm_class.return_value = mock_model

        img = _blank_image()
        result = overlay_bounding_boxes(
            img,
            MagicMock(),
            torch.eye(4),
            _one_actor_snapshot(),
            box_color_rgb=_BOX_COLOR,
            # edge_segments defaults to 16
        )

        changed_mask = np.any(result != img, axis=2)
        self.assertGreater(int(changed_mask.sum()), 0, "Expected at least one pixel to change")
        changed_pixels = result[changed_mask]
        not_box_color = ~np.all(changed_pixels == np.array(_BOX_COLOR, dtype=np.uint8)[None], axis=1)
        self.assertEqual(
            int(not_box_color.sum()),
            0,
            f"All changed pixels must be {_BOX_COLOR}; unexpected: "
            f"{np.unique(changed_pixels[not_box_color], axis=0).tolist()}",
        )

    @patch("nurec.render.bbox_overlay.camera_model_from_parameters")
    def test_multi_actor_all_boxes_drawn(self, mock_cm_class):
        """Two actors at distinct projected positions both produce changed pixels."""
        n_actors = 2
        n_pts = n_actors * 12 * 2  # edge_segments=1
        pts = torch.zeros((n_pts, 2), dtype=torch.float32)
        half = n_pts // 2
        pts[:half] = torch.tensor([40.0, 40.0])  # actor 0 at upper-left cluster
        pts[half:] = torch.tensor([160.0, 160.0])  # actor 1 at lower-right cluster

        mock_model = MagicMock()
        mock_model.world_points_to_image_points_shutter_pose.return_value = _mock_proj_result(pts)
        mock_cm_class.return_value = mock_model

        actor_poses = torch.tensor(
            [
                [[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0]],
                [[5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        snap = ActorsSnapshot(
            actor_ids=["a", "b"],
            actor_poses=actor_poses,
            cuboid_dims=torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
        )

        img = _blank_image()
        result = overlay_bounding_boxes(img, MagicMock(), torch.eye(4), snap, box_color_rgb=_BOX_COLOR, edge_segments=1)

        changed_mask = np.any(result != img, axis=2)
        self.assertGreater(int(changed_mask.sum()), 0, "Expected pixels to change for two actors")
        changed_pixels = result[changed_mask]
        not_box_color = ~np.all(changed_pixels == np.array(_BOX_COLOR, dtype=np.uint8)[None], axis=1)
        self.assertEqual(int(not_box_color.sum()), 0, "All changed pixels must be box color")

    @patch("nurec.render.bbox_overlay._draw_label")
    @patch("nurec.render.bbox_overlay.camera_model_from_parameters")
    def test_label_skipped_when_center_projects_out_of_frame(self, mock_cm_class, mock_draw):
        """An actor whose projected center is outside the image gets NO label: relocating
        its tag into frame would put its id over some other object."""
        from nurec.render.bbox_overlay import overlay_labeled_boxes

        n_pts = 12 * 2
        edge_pts = torch.full((n_pts, 2), 100.0)
        in_frame = _mock_proj_result(edge_pts)
        out_center = _mock_proj_result(torch.tensor([[-50.0, 100.0]]))  # valid, but off-frame
        mock_model = MagicMock()
        mock_model.world_points_to_image_points_shutter_pose.side_effect = [in_frame, out_center]
        mock_cm_class.return_value = mock_model

        overlay_labeled_boxes(
            _blank_image(), MagicMock(), torch.eye(4), _one_actor_snapshot(), labels=["7"], edge_segments=1
        )
        mock_draw.assert_not_called()

    @patch("nurec.render.bbox_overlay._draw_label")
    @patch("nurec.render.bbox_overlay.camera_model_from_parameters")
    def test_label_drawn_when_center_projects_in_frame(self, mock_cm_class, mock_draw):
        from nurec.render.bbox_overlay import overlay_labeled_boxes

        n_pts = 12 * 2
        edge_pts = torch.full((n_pts, 2), 100.0)
        mock_model = MagicMock()
        mock_model.world_points_to_image_points_shutter_pose.side_effect = [
            _mock_proj_result(edge_pts),
            _mock_proj_result(torch.tensor([[100.0, 100.0]])),
        ]
        mock_cm_class.return_value = mock_model

        overlay_labeled_boxes(
            _blank_image(), MagicMock(), torch.eye(4), _one_actor_snapshot(), labels=["7"], edge_segments=1
        )
        mock_draw.assert_called_once()
        self.assertEqual(mock_draw.call_args.args[1], "7")

    def test_invalid_edge_segments_raises(self):
        """edge_segments < 1 must raise ValueError."""
        with self.assertRaises(ValueError):
            overlay_bounding_boxes(
                _blank_image(),
                MagicMock(),
                torch.eye(4),
                _one_actor_snapshot(),
                edge_segments=0,
            )

    def test_missing_cuboid_dims_raises(self):
        """overlay_bounding_boxes must raise ValueError when cuboid_dims is None and num_actors > 0."""
        snap = ActorsSnapshot(
            actor_ids=["a"],
            actor_poses=torch.zeros((1, 2, 7)),
            cuboid_dims=None,
        )
        with self.assertRaises(ValueError, msg="cuboid_dims is required"):
            overlay_bounding_boxes(
                np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8),
                MagicMock(),
                torch.eye(4),
                snap,
            )


if __name__ == "__main__":
    unittest.main()
