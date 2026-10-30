# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Asset-edits track label overlay: draw a labeled box around each cuboid track of a
# chosen Gaussian layer (e.g. the actors you want to bind an asset edit to) so a
# human (or an AI tool) can watch the rendered video and read off which on-screen object
# corresponds to which ``nurec_track_id``. The same ids can be listed from the container
# (see ``--list-tracks`` in the render CLI) and then referenced in an explicit
# per-edit asset_edits.yaml document.

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from ncore.data import ConcreteCameraModelParametersUnion
from ncore.impl.common.transformations import PoseInterpolator
from nurec.datasets.tracks import CuboidTracks
from nurec.render.actors import ActorsSnapshot
from nurec.render.bbox_overlay import overlay_labeled_boxes
from nurec.render.utils import transform_intrinsics_to_resolution
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.types import FrameConversion


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabeledTrack:
    """A single labeled cuboid track of a Gaussian layer."""

    nurec_track_id: int  # layer-local cuboid index; matches gaussian_cuboid_ids
    track_id_str: str  # original annotation track id string
    label_class: str  # semantic class string from the annotation track
    world_centroid: Tuple[float, float, float]  # cuboid center (m) at the first keyframe
    dims: Tuple[float, float, float]  # full cuboid edge lengths (m)


@dataclass
class _TrackPose:
    """Runtime per-track data used to project a labeled box on each frame."""

    nurec_track_id: int
    interpolator: PoseInterpolator
    dims: torch.Tensor  # (3,) full cuboid edge lengths
    ts_min_us: int
    ts_max_us: int


class TrackLabeler:
    """Builds per-track cuboid boxes for a Gaussian layer and overlays labeled boxes.

    The label drawn on each box is the track's ``nurec_track_id`` (the layer-local cuboid
    index, identical to the id stored in ``gaussian_cuboid_ids``), so a user can match a box
    in the video to an id and author an explicit asset-edits document. Works for any layer
    of cuboid tracks, static or dynamic (the per-track pose is interpolated to each
    frame's timestamp, so a moving car's box rides the car).

    A track is labeled only on frames overlapping its keyframe span (the renderer drops
    a track's gaussians outside that span, so a clamped-pose label would sit over
    unrelated content); within the span the pose is keyframe-clamped, and tracks outside
    the camera frustum are culled by the projection.
    """

    def __init__(
        self,
        track_poses: List[_TrackPose],
        tracks: List[LabeledTrack],
        layer_id: str,
    ) -> None:
        """Hold the prepared per-track pose data; build via :meth:`from_model`."""
        self._track_poses = track_poses
        self._tracks = tracks
        self._layer_id = layer_id

    @classmethod
    def from_model(
        cls,
        model: object,
        layer_id: str,
        world_to_nurec: Optional[FrameConversion] = None,
    ) -> Optional["TrackLabeler"]:
        """Construct a labeler from a GaussiansComposite-like model, or ``None`` if unavailable.

        Returns ``None`` (with a warning) when the model has no ``layer_id`` layer or that
        layer has no cuboid tracks, so callers can degrade gracefully.

        Args:
            model: A GaussiansComposite-like model exposing ``gaussians_nodes``.
            world_to_nurec: The model's world-to-NuRec frame conversion. The per-layer cuboid tracks
                live in the model's NuRec (recentered) frame; this is required to convert them back
                to the world frame so the boxes align with the rendered ``camera_to_world`` poses.
                If ``None``, boxes are left in the NuRec frame (and will likely be misaligned).
            layer_id: Gaussian layer whose cuboid tracks to label (e.g. ``traffic_light`` or the
                dynamic-actor layer for vehicles). Every track in the layer is labeled.
        """
        gaussians_nodes = getattr(model, "gaussians_nodes", None)
        if gaussians_nodes is None or layer_id not in gaussians_nodes:
            available = list(gaussians_nodes.keys()) if gaussians_nodes is not None else []
            log.warning(
                "TrackLabeler: model has no '%s' layer; nothing to label (available layers: %s).",
                layer_id,
                available,
            )
            return None
        node = gaussians_nodes[layer_id]
        cuboid_tracks = getattr(node, "cuboid_tracks", None)
        if cuboid_tracks is None or cuboid_tracks.n_tracks == 0:
            log.warning("TrackLabeler: '%s' layer has no cuboid tracks; nothing to label.", layer_id)
            return None

        # The layer's cuboid tracks are stored in the model's NuRec (recentered) frame. Convert them
        # to the world frame (same conversion the renderer applies via get_actor_tracks) so the
        # projected boxes land on the rendered actors.
        if world_to_nurec is not None:
            cuboid_tracks = CuboidTracks.Ops.transform_with_frame_conversion(
                cuboid_tracks, world_to_nurec.inverse(), None
            )
        else:
            log.warning(
                "TrackLabeler: world_to_nurec is None; labeled boxes are in the NuRec "
                "frame and may not align with the rendered image."
            )

        track_ids = list(cuboid_tracks.tracks_id)
        label_classes = list(cuboid_tracks.tracks_label_class)
        n_tracks = len(track_ids)

        track_poses: List[_TrackPose] = []
        tracks: List[LabeledTrack] = []
        for i in range(n_tracks):
            sub = CuboidTracks.Ops.subset_from_indices(cuboid_tracks, [i])
            poses = sub.tracks_poses.matrix().detach().cpu()  # (n_poses, 4, 4) track-to-world
            timestamps = sub.tracks_timestamps_us.detach().cpu()  # (n_poses,)
            dims = sub.cuboids_dims[0].detach().cpu()  # (3,)
            interpolator = PoseInterpolator(poses=poses, timestamps=timestamps)
            ts_min = int(timestamps.min().item())
            ts_max = int(timestamps.max().item())
            track_poses.append(
                _TrackPose(
                    nurec_track_id=i,
                    interpolator=interpolator,
                    dims=dims.to(torch.float32),
                    ts_min_us=ts_min,
                    ts_max_us=ts_max,
                )
            )
            centroid = poses[0, :3, 3].numpy()
            dims_np = dims.numpy()
            tracks.append(
                LabeledTrack(
                    nurec_track_id=i,
                    track_id_str=str(track_ids[i]),
                    label_class=str(label_classes[i]),
                    world_centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                    dims=(float(dims_np[0]), float(dims_np[1]), float(dims_np[2])),
                )
            )

        log.info(
            "TrackLabeler: %d track(s) selected in layer '%s' (of %d cuboid tracks).",
            len(tracks),
            layer_id,
            n_tracks,
        )
        return cls(track_poses, tracks, layer_id)

    @property
    def layer_id(self) -> str:
        """The Gaussian layer whose tracks this labeler draws."""
        return self._layer_id

    def list_tracks(self) -> List[LabeledTrack]:
        """All labeled tracks, sorted by ``nurec_track_id``."""
        return sorted(self._tracks, key=lambda t: t.nurec_track_id)

    def overlay(
        self,
        image: np.ndarray,
        camera_model_parameters: ConcreteCameraModelParametersUnion,
        camera_to_world: torch.Tensor,
        frame_start_us: int,
        frame_end_us: int,
        box_color_rgb: Tuple[int, int, int] = (255, 255, 0),
        line_width: int = 2,
    ) -> np.ndarray:
        """Draw a labeled cuboid box for every track present in this frame.

        Tracks whose keyframe span does not overlap the frame are skipped (the renderer
        drops their gaussians, so a label would sit over unrelated content); the rest are
        projected, with off-frame/behind-camera boxes culled by the projection. Poses are
        sampled at the frame timestamp clamped to the track's keyframe range.

        Args:
            image: (H, W, 3) uint8 RGB rendered frame; copied and drawn on.
            camera_model_parameters: Camera intrinsics at the camera's native resolution; they
                are scaled internally to the resolution of ``image``.
            camera_to_world: (2, 7) start/end camera-to-world tquat for the frame.
            frame_start_us, frame_end_us: Frame capture window (microseconds).
            box_color_rgb: RGB color 0-255 for the box edges and the id tag background.
            line_width: Box edge width in pixels.

        Returns:
            (H, W, 3) uint8 image with labeled boxes drawn (copy of input if no tracks).
        """
        if not self._track_poses:
            return np.asarray(image, dtype=np.uint8).copy()

        poses_list: List[torch.Tensor] = []
        dims_list: List[torch.Tensor] = []
        labels: List[str] = []
        for ft in self._track_poses:
            # Label only tracks present in this frame. The renderer drops a track's
            # gaussians once the frame leaves its keyframe span, so drawing a clamped pose
            # anyway would put this track's id over unrelated content.
            if frame_end_us < ft.ts_min_us or frame_start_us > ft.ts_max_us:
                continue
            ts_start = min(max(frame_start_us, ft.ts_min_us), ft.ts_max_us)
            ts_end = min(max(frame_end_us, ft.ts_min_us), ft.ts_max_us)
            world_poses = ft.interpolator.interpolate_to_timestamps([ts_start, ts_end])  # (2, 4, 4)
            world_poses_tquat = se3_matrix_to_tquat(torch.as_tensor(world_poses, dtype=torch.float32))  # (2, 7)
            poses_list.append(world_poses_tquat)
            dims_list.append(ft.dims)
            labels.append(str(ft.nurec_track_id))

        if not poses_list:
            # No track overlaps this frame; torch.stack would raise on an empty list.
            return np.asarray(image, dtype=np.uint8).copy()

        snapshot = ActorsSnapshot(
            actor_ids=labels,
            actor_poses=torch.stack(poses_list, dim=0),
            cuboid_dims=torch.stack(dims_list, dim=0),
        )

        height, width = image.shape[0], image.shape[1]
        cam_params = transform_intrinsics_to_resolution(camera_model_parameters, (width, height))
        return overlay_labeled_boxes(
            image=image,
            camera_model_parameters=cam_params,
            camera_to_world=camera_to_world,
            actors_snapshot=snapshot,
            labels=labels,
            box_color_rgb=box_color_rgb,
            line_width=line_width,
        )
