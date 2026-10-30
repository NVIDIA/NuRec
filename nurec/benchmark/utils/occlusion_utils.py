# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Occlusion utilities for object-level metrics evaluation.

This module computes how much each object is occluded by closer objects, from a
given camera. Rays are cast from the camera to points
sampled on the target's camera-facing surface and tested against each closer
object, using a class-aware 3D *shape proxy* (``build_class_proxy``) instead of
the loose cuboid -- automobiles = an oriented ellipsoid enlarged in length+width
and clipped to the box, buses/heavy_trucks = the (shrunk) oriented box, riders =
a compact world-axis-aligned ellipsoid, persons = a widened ellipsoid. This
removes the empty-corner overlap that makes non-overlapping objects (staggered
cyclists, adjacent cars) read as occluded. Any class outside this known set
(e.g. stroller, protruding_object, or an unseen class) falls back to the raw
oriented bounding box, so unmodeled objects are handled conservatively.

The computation is pure ``numpy`` geometry with no dependency on the CUDA
``vren`` extension or ``ncore``, so the module is importable and unit-testable
in isolation.
"""

import logging

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


log = logging.getLogger(__name__)

# Divide-by-zero guard for the geometry below (unitless).
_EPS = 1e-9

# Depth tolerance in meters: an occluder must be entered this much closer to the
# camera than the target's sampled surface point to block that ray. Keeps proxies
# that merely touch or graze the target from reading as occlusion.
_DEPTH_TOLERANCE_M = 1e-3


# --- Per-class surface proxies ----------------------------------------------
@dataclass(frozen=True)
class ProxyConfig:
    """Tunable geometry parameters for the class-aware occlusion proxies.

    Larger scale / widen / margin values enlarge a proxy, so more rays are
    blocked and occlusion ratios rise (and vice-versa).

    Attributes:
        car_lon_scale: Car ellipsoid semi-axis scale along length (vs box).
        car_lat_scale: Car ellipsoid semi-axis scale along width (vs box).
        car_vert_scale: Car ellipsoid semi-axis scale along height (vs box).
        car_clip_box: If True, clip the enlarged car ellipsoid to its box;
            otherwise use a pure ellipsoid.
        bus_margin: Box shrink factor for bus / heavy_truck proxies
            (0.9 = 90% of the annotation box).
        rider_widen: Horizontal-radius widen factor for the rider ellipsoid.
        person_widen: Length/width widen factor for the person ellipsoid.
    """

    car_lon_scale: float = 1.5
    car_lat_scale: float = 1.5
    car_vert_scale: float = 1.0
    car_clip_box: bool = True
    bus_margin: float = 0.9
    rider_widen: float = 1.2
    person_widen: float = 1.25


# Default configuration used when a caller does not supply one.
DEFAULT_PROXY_CONFIG = ProxyConfig()

# Exact class name -> proxy kind. Matching is case-insensitive but exact, with no
# substring or keyword heuristics.
_CLASS_PROXY_KIND = {
    "automobile": "car",
    "bus": "box",
    "heavy_truck": "box",
    "rider": "rider",
    "person": "person",
}


def _proxy_kind(class_name: Optional[str]) -> str:
    """Map a class name to its proxy kind, or ``"bbox"`` when it is not a known class."""
    return _CLASS_PROXY_KIND.get((class_name or "").strip().lower(), "bbox")


def ray_obb_entry(
    origins: np.ndarray,
    directions: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    eps: float = _EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect rays with an oriented bounding box, returning entry distance.

    Keeps only the near hit of :func:`ray_obb_interval` (slab method in the box's
    local frame), clamped to 0 for origins already inside the box.

    Args:
        origins: Ray origins in world coordinates, shape (N, 3).
        directions: Ray directions in world coordinates, shape (N, 3) (need not
            be normalized).
        center: Box center in world coordinates, shape (3,).
        rotation: Box rotation matrix (box-to-world), shape (3, 3).
        half_extents: Box half sizes along its local axes, shape (3,).
        eps: Small value to avoid division by zero for axis-parallel rays.

    Returns:
        Tuple of (t_entry, hit):
            t_entry: Distance along the (input-scaled) ray to the entry point,
                shape (N,). Meaningless where ``hit`` is False.
            hit: Boolean array of shape (N,), True where the ray hits the box.
    """
    t_near, _, hit = ray_obb_interval(origins, directions, center, rotation, half_extents, eps)
    t_entry = np.where(t_near < 0, 0.0, t_near)
    return t_entry, hit


def obb_from_corners(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover an oriented bounding box from its 8 (oriented) corner points.

    The box axes are recovered as the eigenvectors of the corner covariance
    (order/sign-independent), and the half-extents as the maximum absolute
    projection of the corners onto each axis.

    Args:
        corners: Corner points of shape (8, 3) (oriented box corners).

    Returns:
        Tuple of (center (3,), rotation (3, 3) box-to-world, half_extents (3,)).
    """
    corners = np.asarray(corners, dtype=np.float64)
    center = corners.mean(axis=0)
    centered = corners - center
    cov = centered.T @ centered
    _, eigvecs = np.linalg.eigh(cov)
    rotation = eigvecs  # columns are the box axes in world coordinates
    local = centered @ rotation
    half_extents = np.abs(local).max(axis=0)
    return center, rotation, half_extents


def sample_obb_surface_points(
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    camera_origin: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """Sample points on the camera-facing surface of an oriented box.

    Only the up-to-three faces whose outward normal points toward the camera are
    sampled, with points allocated across faces proportionally to their
    camera-projected area. This represents what the camera can actually see of
    the object, instead of its (mostly empty) axis-aligned volume.

    Args:
        center: Box center (3,).
        rotation: Box rotation (3, 3), columns are world-space axes.
        half_extents: Box half sizes along its axes (3,).
        camera_origin: Camera position in world coordinates (3,).
        num_points: Number of surface points to sample.

    Returns:
        Array of sampled surface points, shape (M, 3), or an empty (0, 3) array
        when no face is camera-facing. Otherwise ``M`` is at most ``num_points``,
        and can come out a point short: each face's share is rounded independently
        and floored at one point, and any surplus over the budget is trimmed from the end.
    """
    faces = []  # (axis, sign, weight, other_axes)
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = sign * rotation[:, axis]
            face_center = center + sign * half_extents[axis] * rotation[:, axis]
            view = camera_origin - face_center
            cos = float(normal @ view)
            if cos <= 0:
                continue
            other = [k for k in range(3) if k != axis]
            area = (2.0 * half_extents[other[0]]) * (2.0 * half_extents[other[1]])
            view_norm = float(np.linalg.norm(view))
            weight = area * (cos / view_norm if view_norm > _EPS else 0.0)
            faces.append((axis, sign, weight, other))

    if not faces:
        return np.empty((0, 3), dtype=np.float64)

    total_weight = sum(f[2] for f in faces) or 1.0
    points: List[np.ndarray] = []
    for axis, sign, weight, other in faces:
        count = max(1, int(round(num_points * weight / total_weight)))
        face_center = center + sign * half_extents[axis] * rotation[:, axis]
        a = (np.random.rand(count) * 2.0 - 1.0) * half_extents[other[0]]
        b = (np.random.rand(count) * 2.0 - 1.0) * half_extents[other[1]]
        pts = face_center + a[:, None] * rotation[:, other[0]] + b[:, None] * rotation[:, other[1]]
        points.append(pts)

    # Per-face counts are rounded independently and floored at one point, so they
    # can add up to slightly more than the budget; drop the surplus.
    return np.concatenate(points, axis=0)[:num_points]


def ray_ellipsoid_entry(
    origins: np.ndarray,
    directions: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    semi_axes: np.ndarray,
    eps: float = _EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect rays with an oriented ellipsoid, returning the entry distance.

    Keeps only the near hit of :func:`ray_ellipsoid_interval`, clamped to 0 for
    origins already inside. Because the ellipsoid axes are orthonormal and the
    directions are unit, ``t_entry`` is a Euclidean distance.

    Args:
        origins: Ray origins (N, 3).
        directions: Unit ray directions (N, 3).
        center: Ellipsoid center (3,).
        rotation: Box-to-world rotation (3, 3), columns are axes.
        semi_axes: Ellipsoid semi-axes along its axes (3,).
        eps: Small value to avoid division by zero.

    Returns:
        Tuple of (t_entry, hit): entry distance (N,) and boolean hit mask (N,).
    """
    t_near, _, hit = ray_ellipsoid_interval(origins, directions, center, rotation, semi_axes, eps)
    t_entry = np.where(t_near < 0, 0.0, t_near)
    return t_entry, hit


def sample_ellipsoid_surface_points(
    center: np.ndarray,
    rotation: np.ndarray,
    semi_axes: np.ndarray,
    camera_origin: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """Sample points on the camera-facing surface of an oriented ellipsoid.

    Args:
        center: Ellipsoid center (3,).
        rotation: Box-to-world rotation (3, 3), columns are axes.
        semi_axes: Ellipsoid semi-axes (3,).
        camera_origin: Camera position in world coordinates (3,).
        num_points: Target number of surface points.

    Returns:
        Array of camera-facing surface points, shape (M, 3) with M <= num_points.
    """
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    semi = np.maximum(np.asarray(semi_axes, dtype=np.float64), _EPS)
    camera_origin = np.asarray(camera_origin, dtype=np.float64)

    n = max(num_points * 2, 64)
    unit = np.random.randn(n, 3)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), _EPS)
    world = (unit * semi) @ rotation.T + center
    world_normal = (unit / semi) @ rotation.T  # ellipsoid gradient -> outward normal
    facing = np.sum(world_normal * (camera_origin - world), axis=1) > 0
    pts = world[facing]
    return pts[:num_points] if pts.shape[0] > num_points else pts


def _box_axes(rotation: np.ndarray, half_extents: np.ndarray) -> Tuple[int, int, int]:
    """Return (vertical, longitudinal, lateral) box-axis indices.

    Vertical = axis most aligned with world up; of the two horizontal axes the
    one with the larger half-extent is longitudinal (length), the other lateral
    (width).
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    half_extents = np.asarray(half_extents, dtype=np.float64)
    v_ax = int(np.argmax(np.abs(rotation[2, :])))
    horiz = [k for k in range(3) if k != v_ax]
    if half_extents[horiz[0]] >= half_extents[horiz[1]]:
        lon_ax, lat_ax = horiz[0], horiz[1]
    else:
        lon_ax, lat_ax = horiz[1], horiz[0]
    return v_ax, lon_ax, lat_ax


def ray_ellipsoid_interval(
    origins: np.ndarray,
    directions: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    semi_axes: np.ndarray,
    eps: float = _EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray/oriented-ellipsoid entry+exit distances (near, far, hit)."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    semi = np.maximum(np.asarray(semi_axes, dtype=np.float64), eps)

    local_o = (origins - center) @ rotation
    local_d = directions @ rotation
    ou = local_o / semi
    du = local_d / semi

    a = np.sum(du * du, axis=1)
    b = 2.0 * np.sum(ou * du, axis=1)
    c = np.sum(ou * ou, axis=1) - 1.0
    disc = b * b - 4.0 * a * c
    a_safe = np.where(np.abs(a) < eps, eps, a)
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = (-b - sq) / (2.0 * a_safe)
    t1 = (-b + sq) / (2.0 * a_safe)
    t_near = np.minimum(t0, t1)
    t_far = np.maximum(t0, t1)
    hit = (disc >= 0) & (t_far >= 0)
    return t_near, t_far, hit


def ray_obb_interval(
    origins: np.ndarray,
    directions: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    eps: float = _EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray/oriented-box entry+exit distances (near, far, hit) via slabs."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    rot_inv = np.asarray(rotation, dtype=np.float64).T
    local_o = (origins - center) @ rot_inv.T
    local_d = directions @ rot_inv.T
    safe_d = np.where(np.abs(local_d) < eps, eps, local_d)
    t1 = (-half_extents - local_o) / safe_d
    t2 = (half_extents - local_o) / safe_d
    t_near = np.max(np.minimum(t1, t2), axis=1)
    t_far = np.min(np.maximum(t1, t2), axis=1)
    hit = (t_far >= t_near) & (t_far >= 0)
    return t_near, t_far, hit


def build_class_proxy(
    corners: np.ndarray,
    class_name: Optional[str],
    config: ProxyConfig = DEFAULT_PROXY_CONFIG,
) -> dict:
    """Build the per-class 3D surface proxy.

    Known classes get a shape proxy (automobile -> car, bus/heavy_truck -> shrunk
    box, rider/person -> ellipsoid). Any other class (stroller, protruding_object
    or an unseen class) uses the raw oriented bounding box as-is (``"box"`` kind
    with the full, un-shrunk half-extents).

    Args:
        corners: Object box corners [8, 3].
        class_name: Object class name (selects the proxy kind).
        config: Tunable proxy geometry parameters.

    Returns a descriptor dict with keys:
      ``kind``     -- "ellipsoid" | "box" | "car" (ellipsoid clipped to box)
      ``center``   -- proxy center (3,)
      ``rotation`` -- proxy axes (3, 3) columns (identity for axis-aligned rider)
      ``semi``     -- ellipsoid semi-axes (3,) (ellipsoid/car kinds)
      ``half``     -- box half-extents (3,) (box/car kinds)
    """
    center, rot, half = obb_from_corners(corners)
    kind = _proxy_kind(class_name)

    if kind == "bbox":  # unknown / unmodeled class -> raw oriented box (no shrink)
        return {"kind": "box", "center": center, "rotation": rot, "half": half, "semi": None}

    if kind == "box":  # bus / heavy_truck -> shrunk oriented box
        return {"kind": "box", "center": center, "rotation": rot, "half": half * config.bus_margin, "semi": None}

    if kind == "car":  # automobile -> enlarged ellipsoid clipped to box
        v_ax, lon_ax, lat_ax = _box_axes(rot, half)
        semi = np.array(half, dtype=np.float64)
        semi[lon_ax] *= config.car_lon_scale
        semi[lat_ax] *= config.car_lat_scale
        semi[v_ax] *= config.car_vert_scale
        return {
            "kind": ("car" if config.car_clip_box else "ellipsoid"),
            "center": center,
            "rotation": rot,
            "semi": semi,
            "half": half,
        }

    if kind == "rider":  # compact world-axis-aligned ellipsoid based on width
        v_ax, lon_ax, lat_ax = _box_axes(rot, half)
        rh = half[lat_ax] * config.rider_widen
        rv = half[v_ax]
        semi = np.array([rh, rh, rv], dtype=np.float64)
        return {"kind": "ellipsoid", "center": center, "rotation": np.eye(3), "semi": semi, "half": half}

    # person -> compact ellipsoid: thin in both horizontals, full height, then
    # widened by person_widen (vertical is kept at the full box half-extent).
    v_ax, lon_ax, lat_ax = _box_axes(rot, half)
    person_semi_frac = 0.5  # horizontal semi-axis as a fraction of the box half-extent
    semi = np.array(half, dtype=np.float64)
    semi[lon_ax] *= person_semi_frac * config.person_widen
    semi[lat_ax] *= person_semi_frac * config.person_widen
    return {"kind": "ellipsoid", "center": center, "rotation": rot, "semi": semi, "half": half}


def sample_proxy_surface_points(proxy: dict, camera_origin: np.ndarray, num_points: int) -> np.ndarray:
    """Sample camera-facing surface points on a per-class proxy."""
    if proxy["kind"] == "box":
        return sample_obb_surface_points(proxy["center"], proxy["rotation"], proxy["half"], camera_origin, num_points)
    if proxy["kind"] == "ellipsoid":
        return sample_ellipsoid_surface_points(
            proxy["center"], proxy["rotation"], proxy["semi"], camera_origin, num_points
        )

    # car: ellipsoid clipped to box (radial min of the two supports)
    center = np.asarray(proxy["center"], dtype=np.float64)
    rot = np.asarray(proxy["rotation"], dtype=np.float64)
    semi = np.maximum(np.asarray(proxy["semi"], dtype=np.float64), _EPS)
    half = np.maximum(np.asarray(proxy["half"], dtype=np.float64), _EPS)
    camera_origin = np.asarray(camera_origin, dtype=np.float64)

    n = max(num_points * 2, 64)
    dirs = np.random.randn(n, 3)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), _EPS)
    local = dirs @ rot  # world dir -> box frame
    t_ell = 1.0 / np.maximum(np.sqrt(np.sum((local / semi) ** 2, axis=1)), _EPS)
    t_box = 1.0 / np.maximum(np.max(np.abs(local) / half, axis=1), _EPS)
    t = np.minimum(t_ell, t_box)
    world = center + (t[:, None] * dirs)
    facing = np.sum((world - center) * (camera_origin - world), axis=1) > 0
    pts = world[facing]
    return pts[:num_points] if pts.shape[0] > num_points else pts


def ray_proxy_entry(proxy: dict, origins: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ray entry distance + hit mask against a per-class proxy."""
    if proxy["kind"] == "box":
        return ray_obb_entry(origins, directions, proxy["center"], proxy["rotation"], proxy["half"])
    if proxy["kind"] == "ellipsoid":
        return ray_ellipsoid_entry(origins, directions, proxy["center"], proxy["rotation"], proxy["semi"])

    # car: intersection of the ellipsoid and the box
    ne, fe, he = ray_ellipsoid_interval(origins, directions, proxy["center"], proxy["rotation"], proxy["semi"])
    nb, fb, hb = ray_obb_interval(origins, directions, proxy["center"], proxy["rotation"], proxy["half"])
    t_near = np.maximum(ne, nb)
    t_far = np.minimum(fe, fb)
    hit = he & hb & (t_far >= t_near) & (t_far >= 0)
    t_entry = np.where(t_near < 0, 0.0, t_near)
    return t_entry, hit


def compute_occlusion_ratio(
    target_bbox_corners: Optional[np.ndarray],
    other_bboxes: Optional[List[np.ndarray]],
    camera_origin: np.ndarray,
    num_sample_rays: int = 128,
    target_class: Optional[str] = None,
    other_classes: Optional[Sequence[str]] = None,
    config: ProxyConfig = DEFAULT_PROXY_CONFIG,
    target_proxy: Optional[dict] = None,
    other_proxies: Optional[Sequence[dict]] = None,
) -> float:
    """Fraction of the target occluded by closer objects.

    Rays are cast from the camera to points sampled on the target's camera-facing
    surface and tested against each other object. A ray counts as occluded only
    if another object is entered strictly in front of the target sample point.
    All geometry is the class-aware proxy from :func:`build_class_proxy`, never
    the raw cuboid.

    Occluders whose volume contains the camera origin are NOT skipped: their ray
    entry clamps to 0, so they block every ray and the target reads as fully
    occluded (~1.0). :func:`compute_occlusion_ratios_for_frame` flags that
    degenerate camera position rather than dropping the value.

    Args:
        target_bbox_corners: 8 corner points of the target box [8, 3]. May be
            ``None`` when ``target_proxy`` is supplied.
        other_bboxes: Corner arrays for the other objects, each [8, 3]. May be
            ``None`` when ``other_proxies`` is supplied.
        camera_origin: Camera position in world coordinates [3,].
        num_sample_rays: Number of surface rays to sample.
        target_class: Object class of the target (for the shape proxy).
        other_classes: Object classes of ``other_bboxes`` (for the shape proxy),
            aligned by index.
        config: Tunable proxy geometry parameters.
        target_proxy: Optional pre-built target proxy (from
            :func:`build_class_proxy`). When provided, ``target_bbox_corners`` /
            ``target_class`` are not used. Building proxies once per frame and
            reusing them here avoids the O(N^2) proxy rebuild.
        other_proxies: Optional pre-built occluder proxies. When provided,
            ``other_bboxes`` / ``other_classes`` are not used.

    Returns:
        Occlusion ratio in [0, 1]. 0 = fully visible, 1 = fully occluded.
    """
    # Resolve occluder proxies (pre-built or constructed from corners).
    if other_proxies is not None:
        occluder_proxies: List[dict] = list(other_proxies)
    elif other_bboxes is not None:
        occluder_proxies = [
            build_class_proxy(
                occ_corners,
                other_classes[j] if other_classes is not None and j < len(other_classes) else None,
                config,
            )
            for j, occ_corners in enumerate(other_bboxes)
        ]
    else:
        occluder_proxies = []

    if len(occluder_proxies) == 0:
        return 0.0

    camera_origin = np.asarray(camera_origin, dtype=np.float64)
    if target_proxy is None:
        if target_bbox_corners is None:
            raise ValueError("Either target_proxy or target_bbox_corners must be provided.")
        target_proxy = build_class_proxy(target_bbox_corners, target_class, config)
    sample_points = sample_proxy_surface_points(target_proxy, camera_origin, num_sample_rays)
    if sample_points.shape[0] == 0:
        return 0.0

    origins = np.broadcast_to(camera_origin, sample_points.shape).copy()
    directions = sample_points - origins
    sample_dist = np.linalg.norm(directions, axis=1)
    directions = directions / np.maximum(sample_dist[:, None], _EPS)

    occluded = np.zeros(sample_points.shape[0], dtype=bool)
    for occ_proxy in occluder_proxies:
        t_entry, hit = ray_proxy_entry(occ_proxy, origins, directions)
        # Occluder must be entered strictly before reaching the target surface.
        occluded |= hit & (t_entry < sample_dist - _DEPTH_TOLERANCE_M)

    return float(occluded.mean())


def camera_inside_object(
    bbox_corners: Optional[np.ndarray],
    camera_origin: np.ndarray,
    class_name: Optional[str] = None,
    config: ProxyConfig = DEFAULT_PROXY_CONFIG,
    proxy: Optional[dict] = None,
) -> bool:
    """Whether a camera origin lies inside an object's class-aware proxy.

    Uses the same per-class proxy as the occlusion computation, so a camera
    reported inside here is one that reads as a full occluder there.

    Args:
        bbox_corners: Object box corners [8, 3]. May be ``None`` when ``proxy``
            is supplied.
        camera_origin: Camera position in world coordinates [3,].
        class_name: Object class name (selects the proxy kind).
        config: Tunable proxy geometry parameters.
        proxy: Optional pre-built proxy (from :func:`build_class_proxy`) to reuse
            instead of rebuilding from ``bbox_corners`` / ``class_name``.

    Returns:
        True if the camera origin is inside the object's proxy volume.
    """
    if proxy is None:
        if bbox_corners is None:
            raise ValueError("Either proxy or bbox_corners must be provided.")
        proxy = build_class_proxy(bbox_corners, class_name, config)
    inside, _ = _point_inside_built_proxy(proxy, np.asarray(camera_origin, dtype=np.float64))
    return bool(inside)


def compute_occlusion_ratios_for_frame(
    all_bbox_corners: List[np.ndarray],
    camera_origin_gt: np.ndarray,
    camera_origin_novel: np.ndarray,
    num_sample_rays: int = 128,
    classes: Optional[Sequence[str]] = None,
    config: ProxyConfig = DEFAULT_PROXY_CONFIG,
    target_mask: Optional[Sequence[bool]] = None,
    occluder_mask: Optional[Sequence[bool]] = None,
) -> Tuple[List[float], List[float], List[bool]]:
    """Compute occlusion ratios for all objects in a frame.

    For each object, computes the occlusion ratio from both the GT and novel
    camera viewpoints via :func:`compute_occlusion_ratio`.

    Also reports, per object, whether either camera origin lies inside one of
    *that object's occluders*, which marks the occlusion readings that stem from
    an ambiguous camera position. See ``Returns`` for the exact flag semantics.

    ``target_mask`` and ``occluder_mask`` allow skipping objects the caller knows
    are irrelevant to the visible frame (e.g. objects outside the camera
    frustum), which avoids the O(N^2) cost of computing occlusion over a full
    360-degree / all-sensor label set. Both default to ``None``, treating every
    object as both a target and an occluder.

    Args:
        all_bbox_corners: List of bbox corners [8, 3] for all objects.
        camera_origin_gt: GT camera position in world coordinates [3,].
        camera_origin_novel: Novel view camera position [3,].
        num_sample_rays: Number of rays to sample per object.
        classes: Optional per-object class names (aligned with ``all_bbox_corners``)
            used to select each shape proxy.
        config: Tunable proxy geometry parameters.
        target_mask: Optional per-object flags (aligned with ``all_bbox_corners``).
            When provided, occlusion is computed only for objects whose flag is
            True; skipped objects get an occlusion ratio of ``0.0`` (their values
            are not meant to be consumed downstream). ``None`` treats every object
            as a target.
        occluder_mask: Optional per-object flags. When provided, only objects
            whose flag is True can occlude another object. ``None`` treats every
            object as a potential occluder.

    Returns:
        Tuple of (occlusion_ratios_gt, occlusion_ratios_novel,
        camera_inside_flags), each a list with one entry per object.
        ``camera_inside_flags[i]`` is True when the GT or novel camera origin is
        inside any *other* object's proxy (an occluder of object ``i``), and
        False otherwise -- including when only object ``i`` itself contains a
        camera origin.
    """
    num_objects = len(all_bbox_corners)
    is_target = [True] * num_objects if target_mask is None else [bool(m) for m in target_mask]
    is_occluder = [True] * num_objects if occluder_mask is None else [bool(m) for m in occluder_mask]

    # Build each object's class proxy once per frame, skipping objects that are
    # neither a target nor a potential occluder, and reuse it for both the ray
    # tests and the camera-inside checks. Proxy construction draws no RNG, so
    # ordering it before the surface sampling leaves the sample stream unchanged.
    proxies: List[Optional[dict]] = [None] * num_objects
    for i in range(num_objects):
        if is_target[i] or is_occluder[i]:
            cls_i = classes[i] if classes is not None and i < len(classes) else None
            proxies[i] = build_class_proxy(all_bbox_corners[i], cls_i, config)

    occlusion_ratios_gt: List[float] = []
    occlusion_ratios_novel: List[float] = []
    # Whether either camera origin lies inside each object's *own* proxy. Only an
    # intermediate: the flag emitted below is "some *other* object contains a
    # camera origin", since a camera inside an object leaves that object's own
    # occlusion ~0.
    camera_inside_self: List[bool] = []

    for obj_idx in range(num_objects):
        target_proxy = proxies[obj_idx]

        # Containment is needed for any object that can be an occluder (it drives
        # OTHER objects' ambiguity flag) or that is itself a target.
        if target_proxy is not None:
            camera_inside_self.append(
                camera_inside_object(None, camera_origin_gt, proxy=target_proxy)
                or camera_inside_object(None, camera_origin_novel, proxy=target_proxy)
            )
        else:
            camera_inside_self.append(False)

        # Skip occlusion for non-target objects (e.g. objects not visible in the
        # frame). Their ratios are never read downstream, so 0.0 is a safe filler.
        if not is_target[obj_idx]:
            occlusion_ratios_gt.append(0.0)
            occlusion_ratios_novel.append(0.0)
            continue

        # By construction proxies[i] is non-None whenever is_occluder[i] is True
        # (built above); the explicit None check narrows the element type to dict.
        other_proxies = [
            proxy for i, proxy in enumerate(proxies) if i != obj_idx and is_occluder[i] and proxy is not None
        ]

        occ_gt = compute_occlusion_ratio(
            None,
            None,
            camera_origin_gt,
            num_sample_rays=num_sample_rays,
            config=config,
            target_proxy=target_proxy,
            other_proxies=other_proxies,
        )
        occlusion_ratios_gt.append(occ_gt)

        occ_novel = compute_occlusion_ratio(
            None,
            None,
            camera_origin_novel,
            num_sample_rays=num_sample_rays,
            config=config,
            target_proxy=target_proxy,
            other_proxies=other_proxies,
        )
        occlusion_ratios_novel.append(occ_novel)

    # Per-object flag: some *other* occluder contains a camera origin. Subtracting
    # the object's own containment keeps it from flagging itself.
    total_inside = sum(1 for j in range(num_objects) if is_occluder[j] and camera_inside_self[j])
    camera_inside_flags: List[bool] = [
        (total_inside - (1 if (is_occluder[obj_idx] and camera_inside_self[obj_idx]) else 0)) > 0
        for obj_idx in range(num_objects)
    ]

    return occlusion_ratios_gt, occlusion_ratios_novel, camera_inside_flags


def _point_inside_built_proxy(proxy: dict, point: np.ndarray) -> Tuple[bool, float]:
    """Test whether a world point lies inside an already-built class proxy.

    Uses the same box/ellipsoid frame convention as the occlusion ray tests, so a
    point reported as inside here is exactly one whose ray entry clamps to 0 in
    :func:`ray_proxy_entry`. Reached via :func:`camera_inside_object`.

    Args:
        proxy: Proxy descriptor from :func:`build_class_proxy`.
        point: World point (e.g., camera origin), shape (3,).

    Returns:
        Tuple of (inside, depth). ``depth`` is a normalized distance from the
        proxy center in proxy units: ``<= 1`` means the point is inside, and
        smaller values are deeper inside (0 == exact center). For the "car"
        (ellipsoid clipped to box) proxy the point must be inside both, so the
        max of the two normalized distances is returned.
    """
    center = np.asarray(proxy["center"], dtype=np.float64)
    rot = np.asarray(proxy["rotation"], dtype=np.float64)
    local = (np.asarray(point, dtype=np.float64) - center) @ rot

    if proxy["kind"] == "box":
        half = np.maximum(np.asarray(proxy["half"], dtype=np.float64), _EPS)
        depth = float(np.max(np.abs(local) / half))
        return depth <= 1.0, depth
    if proxy["kind"] == "ellipsoid":
        semi = np.maximum(np.asarray(proxy["semi"], dtype=np.float64), _EPS)
        depth = float(np.sqrt(np.sum((local / semi) ** 2)))
        return depth <= 1.0, depth

    # car: ellipsoid clipped to box -> inside iff inside both.
    semi = np.maximum(np.asarray(proxy["semi"], dtype=np.float64), _EPS)
    half = np.maximum(np.asarray(proxy["half"], dtype=np.float64), _EPS)
    d_ell = float(np.sqrt(np.sum((local / semi) ** 2)))
    d_box = float(np.max(np.abs(local) / half))
    depth = max(d_ell, d_box)
    return depth <= 1.0, depth


def should_skip_object(
    occlusion_gt: float,
    occlusion_ren: float,
    occlusion_threshold: float = 0.3,
) -> Tuple[bool, str]:
    """Determine if an object should be filtered out based on occlusion.

    An object is filtered whenever its occlusion ratio exceeds
    ``occlusion_threshold`` in either the GT or the novel view. Filtered objects
    are excluded from per-track/per-class aggregation (but still recorded in the
    detailed metrics).

    Args:
        occlusion_gt: Occlusion ratio in GT view [0, 1].
        occlusion_ren: Occlusion ratio in novel view [0, 1].
        occlusion_threshold: Maximum allowed occlusion ratio in either view.

    Returns:
        Tuple of (should_skip, reason). reason is empty string if not skipped.
    """
    gt_occluded = occlusion_gt > occlusion_threshold
    novel_occluded = occlusion_ren > occlusion_threshold

    if gt_occluded and novel_occluded:
        return True, "both_occluded"
    elif gt_occluded:
        return True, "gt_occluded"
    elif novel_occluded:
        return True, "novel_occluded"

    return False, ""


class OcclusionStatistics:
    """Tracks occlusion statistics for reporting."""

    def __init__(self) -> None:
        """Initialize occlusion statistics tracker."""
        self.total_objects: int = 0
        self.skipped_gt_occluded: int = 0
        self.skipped_novel_occluded: int = 0
        self.skipped_both_occluded: int = 0
        self.evaluated_objects: int = 0
        self.visibility_gt_sum: float = 0.0
        self.visibility_novel_sum: float = 0.0

    def record_object(
        self,
        occlusion_gt: float,
        occlusion_ren: float,
        was_skipped: bool,
        skip_reason: str,
    ) -> None:
        """Record statistics for one object.

        Args:
            occlusion_gt: Occlusion ratio in GT view.
            occlusion_ren: Occlusion ratio in novel view.
            was_skipped: Whether the object was skipped.
            skip_reason: Reason for skipping (if applicable).
        """
        self.total_objects += 1
        self.visibility_gt_sum += 1.0 - occlusion_gt
        self.visibility_novel_sum += 1.0 - occlusion_ren

        if was_skipped:
            if skip_reason == "both_occluded":
                self.skipped_both_occluded += 1
            elif skip_reason == "gt_occluded":
                self.skipped_gt_occluded += 1
            elif skip_reason == "novel_occluded":
                self.skipped_novel_occluded += 1
        else:
            self.evaluated_objects += 1

    def to_dict(self) -> dict:
        """Convert statistics to dictionary for YAML output.

        Returns:
            Dictionary with occlusion statistics.
        """
        return {
            "total_objects": self.total_objects,
            "skipped_gt_occluded": self.skipped_gt_occluded,
            "skipped_novel_occluded": self.skipped_novel_occluded,
            "skipped_both_occluded": self.skipped_both_occluded,
            "evaluated_objects": self.evaluated_objects,
            "avg_visibility_gt": (self.visibility_gt_sum / self.total_objects if self.total_objects > 0 else 0.0),
            "avg_visibility_novel": (self.visibility_novel_sum / self.total_objects if self.total_objects > 0 else 0.0),
        }

    def log_summary(self) -> None:
        """Log summary of occlusion statistics."""
        if self.total_objects == 0:
            log.info("No objects processed for occlusion statistics")
            return

        log.info("Occlusion Statistics:")
        log.info("  Total objects: %d", self.total_objects)
        log.info(
            "  Evaluated: %d (%.1f%%)", self.evaluated_objects, 100.0 * self.evaluated_objects / self.total_objects
        )
        log.info("  Skipped (GT occluded): %d", self.skipped_gt_occluded)
        log.info("  Skipped (novel occluded): %d", self.skipped_novel_occluded)
        log.info("  Skipped (both occluded): %d", self.skipped_both_occluded)
        log.info("  Avg visibility GT: %.3f", self.visibility_gt_sum / self.total_objects)
        log.info("  Avg visibility novel: %.3f", self.visibility_novel_sum / self.total_objects)
