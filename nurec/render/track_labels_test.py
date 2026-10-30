# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrackLabeler (nurec/render/track_labels.py).

CPU-only, hand-built fixtures: real CuboidTracks (built via Factory.from_numpy on the CPU),
a real distortion-free OpenCV pinhole camera, and a fake GaussiansComposite-like model
(SimpleNamespace). The pose-interpolation / tquat / label-mapping / intrinsics tests patch
``overlay_labeled_boxes`` to capture what the labeler would draw; the projection smoke tests
run the full real drawing path through ncore's camera model.

The only piece stubbed out is ``CuboidTracks.Ops.subset_from_indices``: the real op is backed
by a CUDA-only packed kernel (``linstep_interleave_cu``), so ``_labeler_from`` swaps in a
functionally equivalent CPU reimplementation for the duration of ``from_model``.
"""

import unittest

from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import torch

from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.render.track_labels import LabeledTrack, TrackLabeler
from nurec.utils.geometry import se3_matrix_to_tquat, tquat_to_se3_matrix
from nurec.utils.types import FrameConversion


_LAYER = "dynamic_rigids"
_CPU = torch.device("cpu")

# Camera natively calibrated at 200x200 with a centered principal point.
_CAM_W, _CAM_H = 200, 200
_FOCAL = 100.0
_CX = (_CAM_W - 1) / 2.0  # 99.5
_CY = (_CAM_H - 1) / 2.0  # 99.5

_TS0, _TS1 = 1_000_000, 2_000_000  # keyframe timestamps (us) used by most fixtures


def _pose(translation: Sequence[float], yaw_deg: float = 0.0) -> np.ndarray:
    """(4, 4) float32 SE3 with a rotation of ``yaw_deg`` about +z and the given translation."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    m[:3, 3] = np.asarray(translation, dtype=np.float32)
    return m


def _make_cuboid_tracks(
    track_ids: List[str],
    label_classes: List[str],
    poses: List[np.ndarray],  # one (n_i, 4, 4) float32 array per track
    timestamps: List[np.ndarray],  # one (n_i,) int64 array per track
    dims: List[np.ndarray],  # one (3,) array per track
) -> CuboidTracks:
    """Real CPU CuboidTracks from per-track keyframe poses (track-to-world 4x4 matrices)."""
    return CuboidTracks.Factory.from_numpy(
        tracks_id=track_ids,
        tracks_poses=poses,
        tracks_timestamps_us=timestamps,
        tracks_label_class=label_classes,
        tracks_flags=[TrackFlags.NONE] * len(track_ids),
        cuboids_dims=[np.asarray(d, dtype=np.float32) for d in dims],
        device=_CPU,
    )


def _static_track_poses(center: Sequence[float], yaw_deg: float = 0.0) -> np.ndarray:
    """(2, 4, 4) pair of identical keyframe poses (a static object over [_TS0, _TS1])."""
    return np.stack([_pose(center, yaw_deg), _pose(center, yaw_deg)])


def _default_ts() -> np.ndarray:
    return np.array([_TS0, _TS1], dtype=np.int64)


def _single_track_model(
    center: Sequence[float] = (0.0, 0.0, 5.0),
    yaw_deg: float = 0.0,
    poses: Optional[np.ndarray] = None,
    dims: Sequence[float] = (2.0, 2.0, 2.0),
    layer_id: str = _LAYER,
) -> SimpleNamespace:
    """Fake GaussiansComposite-like model with a single-track layer of cuboid tracks."""
    track_poses = poses if poses is not None else _static_track_poses(center, yaw_deg)
    ct = _make_cuboid_tracks(
        track_ids=["track-a"],
        label_classes=["automobile"],
        poses=[track_poses],
        timestamps=[_default_ts()],
        dims=[np.asarray(dims, dtype=np.float32)],
    )
    return _model_with_layer(layer_id, ct)


def _model_with_layer(layer_id: str, cuboid_tracks: CuboidTracks) -> SimpleNamespace:
    return SimpleNamespace(gaussians_nodes={layer_id: SimpleNamespace(cuboid_tracks=cuboid_tracks)})


def _cpu_subset_from_indices(cuboid_tracks: CuboidTracks, indices: Sequence[int]) -> CuboidTracks:
    """CPU stand-in for CuboidTracks.Ops.subset_from_indices (the real op needs a CUDA kernel).

    Rebuilds the selected tracks from the packed representation via Factory.from_numpy,
    producing the same single-track CuboidTracks the real op would.
    """
    idx = [int(i) for i in indices]
    poses = tquat_to_se3_matrix(cuboid_tracks.tracks_poses.data.detach().cpu(), unbatch=False).numpy()
    timestamps = cuboid_tracks.tracks_timestamps_us.detach().cpu().numpy()
    packinfo = cuboid_tracks.tracks_packinfo.detach().cpu().numpy()
    starts_counts = [(int(packinfo[i, 0]), int(packinfo[i, 1])) for i in idx]
    return CuboidTracks.Factory.from_numpy(
        tracks_id=[cuboid_tracks.tracks_id[i] for i in idx],
        tracks_poses=[poses[s : s + n].astype(np.float32) for s, n in starts_counts],
        tracks_timestamps_us=[timestamps[s : s + n].astype(np.int64) for s, n in starts_counts],
        tracks_label_class=[cuboid_tracks.tracks_label_class[i] for i in idx],
        tracks_flags=[TrackFlags(int(cuboid_tracks.tracks_flags[i].item())) for i in idx],
        cuboids_dims=[cuboid_tracks.cuboids_dims[i].detach().cpu().numpy() for i in idx],
        device=_CPU,
    )


def _labeler_from(model: object, **kwargs: object) -> TrackLabeler:
    kwargs.setdefault("layer_id", _LAYER)
    with patch.object(CuboidTracks.Ops, "subset_from_indices", side_effect=_cpu_subset_from_indices):
        labeler = TrackLabeler.from_model(model, **kwargs)  # type: ignore[arg-type]
    assert labeler is not None, "from_model unexpectedly returned None for a populated layer"
    return labeler


def _pinhole_params(width: int = _CAM_W, height: int = _CAM_H) -> OpenCVPinholeCameraModelParameters:
    """Distortion-free global-shutter OpenCV pinhole camera (equivalent to an ideal pinhole)."""
    return OpenCVPinholeCameraModelParameters(
        resolution=np.array([width, height], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32),
        focal_length=np.array([_FOCAL, _FOCAL], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )


def _identity_cam_to_world() -> torch.Tensor:
    """(2, 7) start/end camera-to-world tquat: camera at the world origin, no rotation."""
    return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float32)


def _blank_image(width: int = _CAM_W, height: int = _CAM_H) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _overlay_drawn(labeler: TrackLabeler, frame_start_us: int, frame_end_us: int, image_size: int = _CAM_W) -> bool:
    """True when labeler.overlay actually drew boxes for this frame."""
    image = _blank_image(image_size, image_size)
    with patch("nurec.render.track_labels.overlay_labeled_boxes") as mock_overlay:
        mock_overlay.return_value = image
        labeler.overlay(image, _pinhole_params(), _identity_cam_to_world(), frame_start_us, frame_end_us)
    return mock_overlay.call_count > 0


def _overlay_capture(labeler: TrackLabeler, frame_start_us: int, frame_end_us: int, image_size: int = _CAM_W) -> dict:
    """Run labeler.overlay with overlay_labeled_boxes patched; return the kwargs it was called with."""
    image = _blank_image(image_size, image_size)
    with patch("nurec.render.track_labels.overlay_labeled_boxes") as mock_overlay:
        mock_overlay.return_value = image
        labeler.overlay(image, _pinhole_params(), _identity_cam_to_world(), frame_start_us, frame_end_us)
    mock_overlay.assert_called_once()
    return dict(mock_overlay.call_args.kwargs)


class TestFromModelLayerResolution(unittest.TestCase):
    """Layer lookup and id mapping in TrackLabeler.from_model."""

    def test_missing_layer_returns_none(self) -> None:
        model = SimpleNamespace(gaussians_nodes={})
        self.assertIsNone(TrackLabeler.from_model(model, layer_id="dynamic_rigids"))

    def test_model_without_gaussians_nodes_returns_none(self) -> None:
        self.assertIsNone(TrackLabeler.from_model(object(), layer_id=_LAYER))

    def test_layer_with_no_tracks_returns_none(self) -> None:
        model = _model_with_layer(_LAYER, CuboidTracks.Factory.empty(device=_CPU))
        self.assertIsNone(TrackLabeler.from_model(model, layer_id=_LAYER))

    def test_all_tracks_selected_without_class_filter(self) -> None:
        ct = _make_cuboid_tracks(
            track_ids=["veh-1", "veh-2"],
            label_classes=["automobile", "truck"],
            poses=[_static_track_poses((1.0, 2.0, 3.0)), _static_track_poses((4.0, 5.0, 6.0))],
            timestamps=[_default_ts(), _default_ts()],
            dims=[np.array([2.0, 2.0, 2.0]), np.array([4.0, 2.0, 1.5])],
        )
        labeler = _labeler_from(_model_with_layer("my_layer", ct), layer_id="my_layer")

        self.assertEqual(labeler.layer_id, "my_layer")
        tracks = labeler.list_tracks()
        self.assertEqual([t.nurec_track_id for t in tracks], [0, 1])
        self.assertEqual([t.track_id_str for t in tracks], ["veh-1", "veh-2"])
        self.assertEqual([t.label_class for t in tracks], ["automobile", "truck"])
        np.testing.assert_allclose(tracks[0].world_centroid, (1.0, 2.0, 3.0), atol=1e-5)
        np.testing.assert_allclose(tracks[1].world_centroid, (4.0, 5.0, 6.0), atol=1e-5)
        np.testing.assert_allclose(tracks[1].dims, (4.0, 2.0, 1.5), atol=1e-6)

    def test_world_to_nurec_conversion_maps_centroids_back_to_world(self) -> None:
        """Tracks stored in the NuRec frame are converted via world_to_nurec.inverse()."""
        origin = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        world_to_nurec = FrameConversion.from_origin_scale_axis(
            target_origin=origin, target_scale=1.0, target_axis=[0, 1, 2]
        )
        # Track at the NuRec-frame origin => world centroid must equal the NuRec frame's world origin.
        labeler = _labeler_from(_single_track_model(center=(0.0, 0.0, 0.0)), world_to_nurec=world_to_nurec)
        np.testing.assert_allclose(labeler.list_tracks()[0].world_centroid, origin, atol=1e-4)

    def test_list_tracks_sorted_by_nurec_track_id(self) -> None:
        track_b = LabeledTrack(5, "b", "truck", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        track_a = LabeledTrack(2, "a", "automobile", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        labeler = TrackLabeler([], [track_b, track_a], "layer")
        self.assertEqual([t.nurec_track_id for t in labeler.list_tracks()], [2, 5])


class TestOverlayPoseInterpolation(unittest.TestCase):
    """Per-frame pose sampling: exact at keyframes, blended between, clamped outside."""

    def _moving_labeler(self) -> TrackLabeler:
        """One track translating (0,0,0) -> (10,0,0) between _TS0 and _TS1."""
        poses = np.stack([_pose((0.0, 0.0, 0.0)), _pose((10.0, 0.0, 0.0))])
        return _labeler_from(_single_track_model(poses=poses))

    def _translation(self, frame_start_us: int, frame_end_us: int) -> torch.Tensor:
        snapshot = _overlay_capture(self._moving_labeler(), frame_start_us, frame_end_us)["actors_snapshot"]
        self.assertEqual(tuple(snapshot.actor_poses.shape), (1, 2, 7))
        return snapshot.actor_poses[0, :, :3]  # (2, 3) start/end translations

    def test_pose_exact_at_keyframes(self) -> None:
        translations = self._translation(_TS0, _TS1)
        torch.testing.assert_close(translations[0], torch.tensor([0.0, 0.0, 0.0]))
        torch.testing.assert_close(translations[1], torch.tensor([10.0, 0.0, 0.0]))

    def test_pose_blended_between_keyframes(self) -> None:
        midpoint = (_TS0 + _TS1) // 2
        translations = self._translation(midpoint, midpoint)
        torch.testing.assert_close(translations[0], torch.tensor([5.0, 0.0, 0.0]))
        torch.testing.assert_close(translations[1], torch.tensor([5.0, 0.0, 0.0]))

    def test_track_not_labeled_before_its_first_keyframe(self) -> None:
        """A frame entirely before the track's span draws nothing.

        The renderer culls the track's gaussians there, so a label would sit over
        unrelated content. Clamping the pose into the span and drawing anyway is exactly
        the mis-binding this overlay exists to prevent.
        """
        self.assertFalse(_overlay_drawn(self._moving_labeler(), 0, _TS0 - 1))

    def test_track_not_labeled_after_its_last_keyframe(self) -> None:
        """A frame entirely after the track's span draws nothing."""
        self.assertFalse(_overlay_drawn(self._moving_labeler(), _TS1 + 1_000_000, _TS1 + 2_000_000))

    def test_track_labeled_when_the_frame_partially_overlaps_its_span(self) -> None:
        """A frame straddling the first keyframe still labels, with the pose clamped in."""
        translations = self._translation(_TS0 - 1_000_000, _TS0)
        torch.testing.assert_close(translations, torch.zeros((2, 3)))

    def test_rotation_slerp_midpoint(self) -> None:
        """SLERP between 0 and 90 degrees of yaw lands at exactly 45 degrees at the midpoint."""
        poses = np.stack([_pose((0.0, 0.0, 0.0), yaw_deg=0.0), _pose((0.0, 0.0, 0.0), yaw_deg=90.0)])
        labeler = _labeler_from(_single_track_model(poses=poses))
        midpoint = (_TS0 + _TS1) // 2

        snapshot = _overlay_capture(labeler, midpoint, midpoint)["actors_snapshot"]
        got = tquat_to_se3_matrix(snapshot.actor_poses[0, 0], unbatch=True)
        expected = torch.from_numpy(_pose((0.0, 0.0, 0.0), yaw_deg=45.0))
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=0.0)


class TestOverlayIntrinsicsRescaling(unittest.TestCase):
    """Native-resolution intrinsics are rescaled to the resolution of the rendered image."""

    def test_intrinsics_rescaled_to_smaller_render_resolution(self) -> None:
        labeler = _labeler_from(_single_track_model())
        cam = _overlay_capture(labeler, _TS0, _TS1, image_size=100)["camera_model_parameters"]

        self.assertEqual(cam.resolution.astype(int).tolist(), [100, 100])
        np.testing.assert_allclose(cam.focal_length, [_FOCAL / 2.0, _FOCAL / 2.0], rtol=1e-6)
        # Pinhole-family transform scales the principal point directly: 99.5 * 0.5 = 49.75
        np.testing.assert_allclose(cam.principal_point, [_CX / 2.0, _CY / 2.0], atol=1e-5)

    def test_intrinsics_unchanged_at_native_resolution(self) -> None:
        labeler = _labeler_from(_single_track_model())
        cam = _overlay_capture(labeler, _TS0, _TS1, image_size=_CAM_W)["camera_model_parameters"]

        self.assertEqual(cam.resolution.astype(int).tolist(), [_CAM_W, _CAM_H])
        np.testing.assert_allclose(cam.focal_length, [_FOCAL, _FOCAL], rtol=1e-6)
        np.testing.assert_allclose(cam.principal_point, [_CX, _CY], atol=1e-5)


class TestSe3TquatConversion(unittest.TestCase):
    """SE3 matrix <-> tquat conversion used to hand poses to the box overlay."""

    def test_se3_to_tquat_round_trip_batch(self) -> None:
        poses = np.stack(
            [
                _pose((0.0, 0.0, 0.0), yaw_deg=0.0),
                _pose((1.0, 2.0, 3.0), yaw_deg=90.0),
                _pose((-5.0, 0.5, 10.0), yaw_deg=137.0),
            ]
        )
        tquat = se3_matrix_to_tquat(torch.from_numpy(poses), unbatch=False)
        self.assertEqual(tuple(tquat.shape), (3, 7))
        # Quaternions are unit-norm, translations pass through untouched.
        torch.testing.assert_close(tquat[:, 3:].norm(dim=1), torch.ones(3))
        torch.testing.assert_close(tquat[:, :3], torch.from_numpy(poses[:, :3, 3]))

        round_tripped = tquat_to_se3_matrix(tquat, unbatch=False)
        torch.testing.assert_close(round_tripped, torch.from_numpy(poses), atol=1e-6, rtol=0.0)

    def test_overlay_poses_are_tquat_of_keyframe_pose(self) -> None:
        """The snapshot handed to overlay_labeled_boxes carries the tquat of the sampled SE3 pose."""
        center: Tuple[float, float, float] = (1.0, 2.0, 3.0)
        labeler = _labeler_from(_single_track_model(center=center, yaw_deg=90.0))

        snapshot = _overlay_capture(labeler, _TS0, _TS0)["actors_snapshot"]
        expected = se3_matrix_to_tquat(torch.from_numpy(_pose(center, yaw_deg=90.0)), unbatch=True)
        for row in range(2):  # static track: identical start/end pose
            got = snapshot.actor_poses[0, row]
            torch.testing.assert_close(got[:3], expected[:3], atol=1e-5, rtol=0.0)
            # q and -q encode the same rotation; compare via the rotation matrix.
            torch.testing.assert_close(
                tquat_to_se3_matrix(got, unbatch=True),
                tquat_to_se3_matrix(expected, unbatch=True),
                atol=1e-5,
                rtol=0.0,
            )


class TestLabelIdMapping(unittest.TestCase):
    """The drawn label is the layer-local nurec_track_id of the right track."""

    def _three_track_model(self) -> SimpleNamespace:
        ct = _make_cuboid_tracks(
            track_ids=["veh-1", "tl-1", "ped-1"],
            label_classes=["automobile", "traffic_light", "pedestrian"],
            poses=[_static_track_poses((float(i), 0.0, 5.0)) for i in range(3)],
            timestamps=[_default_ts()] * 3,
            dims=[np.array([1.0 + i, 2.0, 3.0]) for i in range(3)],
        )
        return _model_with_layer(_LAYER, ct)

    def test_every_track_is_labeled_with_its_layer_local_id(self) -> None:
        labeler = _labeler_from(self._three_track_model())
        kwargs = _overlay_capture(labeler, _TS0, _TS1)

        self.assertEqual(kwargs["labels"], ["0", "1", "2"])  # layer-local indices
        snapshot = kwargs["actors_snapshot"]
        self.assertEqual(snapshot.actor_ids, ["0", "1", "2"])
        # Row 1 carries the data of track index 1 (dims disambiguate the tracks).
        torch.testing.assert_close(snapshot.cuboid_dims[1], torch.tensor([2.0, 2.0, 3.0]))
        torch.testing.assert_close(snapshot.actor_poses[1, 0, :3], torch.tensor([1.0, 0.0, 5.0]))

    def test_all_tracks_labeled_in_layer_order(self) -> None:
        labeler = _labeler_from(self._three_track_model())
        kwargs = _overlay_capture(labeler, _TS0, _TS1)

        self.assertEqual(kwargs["labels"], ["0", "1", "2"])
        snapshot = kwargs["actors_snapshot"]
        self.assertEqual(snapshot.num_actors(), 3)
        torch.testing.assert_close(
            snapshot.cuboid_dims, torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 3.0]])
        )

    def test_empty_labeler_returns_untouched_copy_without_drawing(self) -> None:
        labeler = TrackLabeler([], [], "empty_layer")
        image = _blank_image()
        image[10:20, 10:20] = (7, 8, 9)

        with patch("nurec.render.track_labels.overlay_labeled_boxes") as mock_overlay:
            result = labeler.overlay(image, _pinhole_params(), _identity_cam_to_world(), _TS0, _TS1)

        mock_overlay.assert_not_called()
        self.assertIsNot(result, image)
        np.testing.assert_array_equal(result, image)


class TestProjectionSmoke(unittest.TestCase):
    """Full real pipeline: pinhole projection, box drawing, and label anchor selection.

    Camera at the world origin looking along +z (identity camera-to-world; ncore camera
    convention x-right / y-down / z-forward), 200x200 image, f=100, principal point (99.5, 99.5).
    A 2m cube at (0, 0, 5) projects its corners to a square roughly spanning pixels 74..125.
    """

    def _render(self, center: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        labeler = _labeler_from(_single_track_model(center=center))
        image = _blank_image()
        result = labeler.overlay(image, _pinhole_params(), _identity_cam_to_world(), _TS0, _TS1)
        return image, result

    def test_cuboid_in_front_of_camera_draws_box_and_label_inside_image(self) -> None:
        image, result = self._render((0.0, 0.0, 5.0))
        changed = np.any(result != image, axis=2)
        self.assertGreater(int(changed.sum()), 0, "Expected the box/label to change pixels")

        rows, cols = np.nonzero(changed)
        # Everything drawn stays in the neighbourhood of the projected cuboid (image center).
        self.assertGreaterEqual(rows.min(), 30)
        self.assertLessEqual(rows.max(), 140)
        self.assertGreaterEqual(cols.min(), 45)
        self.assertLessEqual(cols.max(), 155)
        # Box edges land in the projected square (top edge at y ~ 74.5).
        self.assertTrue(np.any(rows >= 73), "Expected box-edge pixels at/below the projected box top")
        # The id tag is anchored ABOVE the projected box top, still inside the image.
        self.assertTrue(np.any(rows < 73), "Expected label-tag pixels above the projected box top")
        # Box lines are drawn without anti-aliasing: the exact default box color must appear.
        yellow = np.all(result == np.array([255, 255, 0], dtype=np.uint8)[None, None], axis=2)
        self.assertGreater(int(yellow.sum()), 0, "Expected pure box-color pixels from the wireframe")

    def test_cuboid_behind_camera_draws_nothing(self) -> None:
        image, result = self._render((0.0, 0.0, -5.0))
        np.testing.assert_array_equal(result, image)

    def test_off_center_cuboid_draws_on_the_correct_side(self) -> None:
        """A cuboid at +x must draw only right of the principal point (catches sign flips)."""
        image, result = self._render((1.0, 0.0, 5.0))
        changed = np.any(result != image, axis=2)
        self.assertGreater(int(changed.sum()), 0)
        _, cols = np.nonzero(changed)
        self.assertGreaterEqual(cols.min(), 95, "All drawn pixels must sit right of the image center")


if __name__ == "__main__":
    unittest.main()
