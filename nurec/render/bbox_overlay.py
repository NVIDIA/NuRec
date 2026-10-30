# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bounding box overlay: project 3D cuboids from ActorsSnapshot onto an image
# and draw the box edges. Implemented as a separate library function in nurec.render so that
# the CLI, gRPC serve, and any other caller can use it without duplicating logic.

from __future__ import annotations

import functools
import logging

from typing import List, Optional, Tuple

import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont

from ncore.data import ConcreteCameraModelParametersUnion
from ncore.sensors import camera_model_from_parameters
from nurec.render.actors import ActorsSnapshot
from nurec.utils.geometry import se3_matrix_inverse, tquat_to_se3_matrix


log = logging.getLogger(__name__)


# Box edges: each tuple is (corner_i, corner_j) for the 12 edges of a cuboid.
# Corners 0..7 are (±hx, ±hy, ±hz) in order: 0=(+,+,+), 1=(+,+,-), 2=(+,-,+), 3=(+,-,-), 4=(-,+,+), 5=(-,+,-), 6=(-,-,+), 7=(-,-,-)
_CUBOID_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _clip_line_segment_to_image(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Clip a segment to pixel bounds [0, width-1] x [0, height-1]; None if it does not intersect."""
    if width <= 0 or height <= 0:
        return None
    xmin, ymin = 0.0, 0.0
    xmax, ymax = float(width - 1), float(height - 1)
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q, strict=True):
        if abs(pi) < 1e-15:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 > u2:
            return None
    xa, ya = x0 + u1 * dx, y0 + u1 * dy
    xb, yb = x0 + u2 * dx, y0 + u2 * dy
    return (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb)))


def _build_world_corners_from_snapshot(
    actors_snapshot: ActorsSnapshot,
) -> torch.Tensor:
    """Build (8*num_actors, 3) world-space corners from snapshot poses and cuboid dimensions."""
    num_actors = actors_snapshot.num_actors()
    if num_actors == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=actors_snapshot.actor_poses.device)
    cuboid_dims = actors_snapshot.cuboid_dims
    assert cuboid_dims is not None

    device = actors_snapshot.actor_poses.device
    poses = actors_snapshot.actor_poses  # (num_actors, 2, 7)
    # Static-pose approximation: use frame-start pose for each actor.
    # Averaging quaternions is not a valid SE(3) operation and can flip sign for fast-rotating actors.
    pose_start = poses[:, 0, :]  # (num_actors, 7)
    T_local_to_world = tquat_to_se3_matrix(pose_start, unbatch=False)  # (num_actors, 4, 4)

    # Local corners: (±hx, ±hy, ±hz); track dims are full lengths, so half-extents = 0.5 * dim
    h = 0.5 * cuboid_dims.to(device=device, dtype=poses.dtype)
    # (num_actors, 8, 3): for each box, 8 corners
    signs = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=h.dtype,
        device=device,
    )  # (8, 3)
    local_corners = h.unsqueeze(1) * signs.unsqueeze(0)  # (num_actors, 8, 3)
    local_corners_h = torch.nn.functional.pad(local_corners, (0, 1), value=1.0)  # (num_actors, 8, 4)
    # Transform to world: p_world = T @ p_local
    world_corners = (T_local_to_world.unsqueeze(1) @ local_corners_h.unsqueeze(-1)).squeeze(-1)[
        :, :, :3
    ]  # (num_actors, 8, 3)
    return world_corners.reshape(-1, 3)  # (8*num_actors, 3)


def _build_world_edge_sample_points(
    world_corners_flat: torch.Tensor,
    num_actors: int,
    segments_per_edge: int,
) -> torch.Tensor:
    """Sample points along each of the 12 cuboid edges in world space (for distortion + partial FOV).

    ``world_corners_flat``: (8 * num_actors, 3). Returns (num_actors * 12 * (segments_per_edge + 1), 3)
    ordered as: for each actor, for each edge in ``_CUBOID_EDGES``, points along the edge from corner i→j.
    """
    if segments_per_edge < 1:
        raise ValueError("segments_per_edge must be >= 1")
    device = world_corners_flat.device
    dtype = world_corners_flat.dtype
    corners = world_corners_flat.view(num_actors, 8, 3)
    n = segments_per_edge + 1
    t = torch.linspace(0, 1, n, device=device, dtype=dtype)
    edge_chunks: list[torch.Tensor] = []
    for ci, cj in _CUBOID_EDGES:
        p0 = corners[:, ci, :]
        p1 = corners[:, cj, :]
        seg = p0.unsqueeze(1) * (1.0 - t).view(1, n, 1) + p1.unsqueeze(1) * t.view(1, n, 1)
        edge_chunks.append(seg)
    stacked = torch.stack(edge_chunks, dim=1)
    return stacked.reshape(-1, 3)


def _valid_mask_from_projection_result(
    num_points: int,
    valid_indices: Optional[torch.Tensor],
    valid_flag: Optional[torch.Tensor],
) -> np.ndarray:
    """(num_points,) bool — True where the projection is valid (e.g. in front of camera)."""
    if valid_flag is not None:
        return valid_flag.cpu().numpy().astype(bool, copy=False)
    if valid_indices is None:
        return np.ones((num_points,), dtype=bool)
    out = np.zeros((num_points,), dtype=bool)
    out[valid_indices.cpu().numpy()] = True
    return out


def _resolve_world_to_cam_matrix(camera_to_world: torch.Tensor) -> torch.Tensor:
    """Resolve a (4,4)/(2,7)/(7,) camera-to-world pose into a single (4,4) world-to-camera SE3.

    Uses the start pose for (2,7) / (7,) inputs. Static-pose approximation: averaging SE3
    matrices is not valid; sign-flipped quaternions give a meaningless mean rotation for
    fast-rolling-shutter motion, so we always project against the frame-start pose.
    """
    if camera_to_world.shape not in {torch.Size([4, 4]), torch.Size([2, 7]), torch.Size([7])}:
        raise ValueError(f"Unsupported camera_to_world shape {camera_to_world.shape}; expected (4, 4), (2, 7), or (7,)")
    if camera_to_world.dim() == 2 and camera_to_world.shape == (4, 4):
        T_cam_to_world = camera_to_world
    else:
        start_tquat = camera_to_world[0] if camera_to_world.dim() == 2 else camera_to_world
        T_cam_to_world = tquat_to_se3_matrix(start_tquat, unbatch=True)  # (4, 4)
    return se3_matrix_inverse(T_cam_to_world)


def project_world_points_to_image(
    camera_model_parameters: ConcreteCameraModelParametersUnion,
    camera_to_world: torch.Tensor,
    world_points: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project ``world_points`` (N, 3) into image pixels via the exact camera model.

    Shares the same projection path as the cuboid overlay (fisheye/distortion aware), so the
    returned pixels land on the same locations the renderer rasterizes to.

    Args:
        camera_model_parameters: Camera intrinsics, already transformed to the target output
            resolution (see ``transform_intrinsics_to_resolution``).
        camera_to_world: (2, 7) start/end tquat, (7,) tquat, or (4, 4) SE3 camera-to-world pose.
        world_points: (N, 3) float tensor of points in world frame.

    Returns:
        (pixels, valid): ``pixels`` is (N, 2) float32 image coordinates; ``valid`` is (N,) bool,
        True where the point projects in front of the camera with a defined projection.
    """
    world_points_t = world_points.detach().cpu().to(torch.float32)
    if world_points_t.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    camera_model = camera_model_from_parameters(camera_model_parameters, device="cpu")
    T_world_to_cam_np = _resolve_world_to_cam_matrix(camera_to_world).cpu().numpy()
    # Use shutter_pose API with same 4x4 twice to avoid Pose object (ncore's to_torch doesn't handle Pose).
    result = camera_model.world_points_to_image_points_shutter_pose(
        world_points_t,
        T_world_to_cam_np,
        T_world_to_cam_np,
        return_valid_indices=True,
        return_all_projections=True,
    )
    valid_np = _valid_mask_from_projection_result(
        result.image_points.shape[0],
        getattr(result, "valid_indices", None),
        getattr(result, "valid_flag", None),
    )
    return result.image_points.cpu().numpy().astype(np.float32, copy=False), valid_np


@functools.lru_cache(maxsize=8)
def _load_label_font_sized(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Load (and cache) a scalable font at ``size``; the parse is too slow to repeat per frame.

    A label the reader cannot resolve defeats the whole overlay, so the fallback chain
    keeps the REQUESTED SIZE rather than the font family: system DejaVu/Arial first, then
    Pillow's own bundled scalable font via ``load_default(size)``. Only if that is also
    unavailable (Pillow < 10.1) do we drop to the unscaled ~11 px bitmap default, and then
    we say so loudly — on a 2160-row render that is an unreadable tag.
    """
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        log.warning(
            "no scalable font available (no system DejaVu/Arial TTF, and this Pillow's "
            "load_default() takes no size); track labels fall back to the ~11px bitmap font "
            "and will be unreadable at high resolution. Install fonts-dejavu-core."
        )
        return ImageFont.load_default()
    log.warning(
        "no system TTF found (tried DejaVu Sans/Arial); track labels use Pillow's bundled "
        "font at the requested %dpx. Install fonts-dejavu-core for the intended typeface.",
        size,
    )
    return font


def _load_label_font(image_height: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Pick a legible font size relative to image height, falling back to PIL's bitmap font."""
    return _load_label_font_sized(max(14, image_height // 55))


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor_xy: Tuple[float, float],
    width: int,
    height: int,
    box_color: Tuple[int, int, int],
    font: "ImageFont.FreeTypeFont | ImageFont.ImageFont",
    margin: int = 3,
) -> None:
    """Draw ``text`` in a filled tag just ABOVE ``anchor_xy``, nudged to stay inside the image.

    The tag is horizontally centered on the anchor and lifted so its bottom sits just
    above it — anchor with the box-top point so the tag never covers the object.

    ``anchor_xy`` must already be inside the image (:func:`overlay_labeled_boxes`
    rejects out-of-frame anchors before calling this — see the rationale there).
    """
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = int(right - left), int(bottom - top)
    pad = 2
    tag_w = tw + 2 * pad
    tag_h = th + 2 * pad
    x = int(anchor_xy[0] - tag_w / 2)
    y = int(anchor_xy[1] - tag_h - margin)
    if y < 0:
        # No room above the box top (near the frame top): drop the tag just below the anchor.
        y = int(anchor_xy[1] + margin)
    # Nudge the TAG back inside the frame — never the anchor.
    x = min(max(x, 0), max(width - tag_w - 1, 0))
    y = min(max(y, 0), max(height - tag_h - 1, 0))
    draw.rectangle([x, y, x + tag_w, y + tag_h], fill=box_color)
    # Black or white text depending on tag luminance, for contrast against the box color.
    lum = 0.299 * box_color[0] + 0.587 * box_color[1] + 0.114 * box_color[2]
    text_color = (0, 0, 0) if lum > 140 else (255, 255, 255)
    draw.text((x + pad, y + pad), text, fill=text_color, font=font)


def overlay_labeled_boxes(
    image: np.ndarray,
    camera_model_parameters: ConcreteCameraModelParametersUnion,
    camera_to_world: torch.Tensor,
    actors_snapshot: ActorsSnapshot,
    labels: Optional[List[str]] = None,
    box_color_rgb: Tuple[int, int, int] = (0, 255, 0),
    line_width: int = 1,
    edge_segments: int = 16,
) -> np.ndarray:
    """Overlay 3D cuboid wireframes onto an image, optionally with a per-actor text label.

    Standalone library function: any image in, camera intrinsics/pose in, box color in,
    ActorsSnapshot with cuboid poses in, overlaid image out. Does not call RenderableModel.

    Args:
        image: (H, W, 3) uint8 RGB image; will be copied and drawn on.
        camera_model_parameters: Camera intrinsics (ncore ConcreteCameraModelParametersUnion),
            already transformed to the resolution of ``image``.
        camera_to_world: (2, 7) tquat [tx,ty,tz,qx,qy,qz,qw] start/end camera-to-world pose,
            or (4, 4) single pose. Used to compute world-to-camera for projection.
        actors_snapshot: Poses of each cuboid (world frame) and per-actor cuboid dimensions.
        labels: Optional list of length ``num_actors`` of text drawn in a tag ABOVE each cuboid
            (e.g. the track id), centered over the box to avoid covering the object (near the
            frame top the tag falls back to just below the anchor). An actor whose projected
            center falls outside the image is NOT labeled — relocating its tag into frame
            would put its id on some other object. ``None`` draws boxes only.
        box_color_rgb: RGB color 0-255 for box edges (and the label tag background).
        line_width: Line width in pixels for drawing.
        edge_segments: Number of sub-segments per 3D edge (must be >= 1). Each edge is sampled at
            ``edge_segments + 1`` points in world space, projected, then drawn as short 2D segments.
            This follows lens distortion better than a single corner-to-corner chord and correctly
            renders partially-visible edges where one corner is behind the camera.

    Returns:
        (H, W, 3) uint8 image with boxes (and labels) drawn (copy of input if no actors).
    """
    out = np.asarray(image, dtype=np.uint8).copy()
    num_actors = actors_snapshot.num_actors()
    if num_actors == 0:
        return out
    if actors_snapshot.cuboid_dims is None:
        raise ValueError("actors_snapshot.cuboid_dims is required for overlay_labeled_boxes when num_actors > 0")
    if edge_segments < 1:
        raise ValueError(f"edge_segments must be >= 1, got {edge_segments}")
    if labels is not None and len(labels) != num_actors:
        raise ValueError(f"labels has length {len(labels)} but snapshot has {num_actors} actors")

    world_corners = _build_world_corners_from_snapshot(actors_snapshot)
    height, width = out.shape[0], out.shape[1]
    pil_image = Image.fromarray(out, mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    lw = max(1, line_width)

    world_points = _build_world_edge_sample_points(world_corners, num_actors, edge_segments)
    samples_per_edge = edge_segments + 1

    pts, valid_np = project_world_points_to_image(camera_model_parameters, camera_to_world, world_points)

    pts_per_actor = len(_CUBOID_EDGES) * samples_per_edge

    for actor_idx in range(num_actors):
        base = actor_idx * pts_per_actor
        for edge_idx in range(len(_CUBOID_EDGES)):
            edge_base = base + edge_idx * samples_per_edge
            for s in range(edge_segments):
                ia = edge_base + s
                ib = edge_base + s + 1
                if not (valid_np[ia] and valid_np[ib]):
                    continue
                pi, pj = pts[ia], pts[ib]
                clipped = _clip_line_segment_to_image(
                    float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1]), width, height
                )
                if clipped is not None:
                    c1, c2 = clipped
                    draw.line([c1[0], c1[1], c2[0], c2[1]], fill=box_color_rgb, width=lw)

    if labels is not None:
        # Project cuboid centers (pose translation) once, used for the visibility test and to
        # horizontally center each tag over its box.
        centers_world = actors_snapshot.actor_poses[:, 0, :3].detach().cpu().to(torch.float32)
        center_pts, center_valid = project_world_points_to_image(
            camera_model_parameters, camera_to_world, centers_world
        )
        font = _load_label_font(height)
        for actor_idx in range(num_actors):
            if not center_valid[actor_idx]:
                continue
            cx, cy = float(center_pts[actor_idx, 0]), float(center_pts[actor_idx, 1])
            # The anchor must be IN FRAME. This tool exists so a human reads an id off a
            # video and binds it in asset_edits.yaml, so a tag drawn anywhere other than
            # over its own object is worse than no tag: an actor off to the side would get
            # its id painted at the frame edge, on top of a different object. Skipping is
            # correct — that actor genuinely is not in this picture.
            if not (0.0 <= cx < width and 0.0 <= cy < height):
                continue
            # Anchor the tag at the TOP of the projected cuboid (topmost valid edge sample),
            # so it sits ABOVE the box and never covers the lamp. Only IN-FRAME samples
            # count: near a fisheye projection singularity a valid sample can land at a
            # large negative pixel row, which would pin every such tag to the frame top.
            base = actor_idx * pts_per_actor
            actor_pts = pts[base : base + pts_per_actor]
            actor_valid = valid_np[base : base + pts_per_actor] & (
                (actor_pts[:, 0] >= 0) & (actor_pts[:, 0] < width) & (actor_pts[:, 1] >= 0) & (actor_pts[:, 1] < height)
            )
            top_y = float(actor_pts[actor_valid, 1].min()) if actor_valid.any() else cy
            _draw_label(
                draw,
                labels[actor_idx],
                (cx, top_y),
                width,
                height,
                box_color_rgb,
                font,
            )

    return np.asarray(pil_image)


def overlay_bounding_boxes(
    image: np.ndarray,
    camera_model_parameters: ConcreteCameraModelParametersUnion,
    camera_to_world: torch.Tensor,
    actors_snapshot: ActorsSnapshot,
    box_color_rgb: Tuple[int, int, int] = (0, 255, 0),
    line_width: int = 1,
    edge_segments: int = 16,
) -> np.ndarray:
    """Overlay 3D bounding boxes (cuboids) onto an image (no labels).

    Thin wrapper around :func:`overlay_labeled_boxes` with ``labels=None``; kept as the stable
    public entry point used by the render CLI and gRPC serve. See that function for arg docs.
    """
    return overlay_labeled_boxes(
        image=image,
        camera_model_parameters=camera_model_parameters,
        camera_to_world=camera_to_world,
        actors_snapshot=actors_snapshot,
        labels=None,
        box_color_rgb=box_color_rgb,
        line_width=line_width,
        edge_segments=edge_segments,
    )
