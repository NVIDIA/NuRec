#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""thin_client.py -- host-side Python gRPC client for `nurec serve-grpc`.

Bundled NuRec RGB client. Talks to a warm
serve-grpc server over its gRPC port directly, with no per-call
client-container cold-start (no per-call `docker run` boot, Python
interpreter startup, or CUDA context init in a client container).
See `README.md` next to this script for the full picture.

Request flow:

  1. At startup, query the warm server once for:
       - get_available_scenes        -> pick scene_id (or use --scene-id)
       - get_available_cameras       -> camera intrinsics + camera_to_rig
       - get_available_trajectories  -> ego trajectory (poses + timestamps)
  2. For each recorded ego pose in the trajectory, build an
     RGBRenderRequest (single camera, --mode single) or
     BatchRGBRenderRequest (multiple cameras, --mode batch) and send it.
  3. Write the returned JPEG/PNG bytes to <output-dir>/000000.jpeg ...
     (single-cam) or <output-dir>/<camera>/000000.jpeg ... (batch).
  4. Write timestamps.json and results.json (per-frame stats + summary).

What this client deliberately does NOT do (kept for the bundled CLI
`nurec render-grpc` to handle, or punted to a future iteration):

  - Rolling shutter modeling: start_pose == end_pose for each frame;
    the server gets a [t, t+1) instant-shutter window.
  - Dynamic obstacles / actor edits / asset edits.
  - Lidar rendering.

What this client DOES do (the common "shift / rotate the rig and
re-render" interaction for novel-view generation):

  - Rig translation + rotation offset, via --rig-translation-offset and
    --rig-rotation-offset. Composition matches `nurec render` (and the
    source-built `render-grpc` CLI) so the rendered novel view is byte-
    comparable with the per-turn-CLI baseline.
  - Runtime scene loading via --scene-url + --scene-id. Sent as
    `x-nre-scene-url` / `x-nre-scene-id` gRPC metadata; the server's
    SceneDownloadInterceptor registers (file://...) or downloads
    (https://...) on first request, then caches under its
    --cache-size LRU. Lets a long-running session add a new HF/S3
    clip to a warm server without restarting it.
  - Camera-intrinsics override from a rig JSON, via --rig-file. The
    server's `get_available_cameras()` reports the camera intrinsics
    baked into the USDZ at recording / reconstruction time. Pointing
    --rig-file at a different rig (e.g. an `augmented_rig.json` that
    adds bivariate-windshield distortion on top of the recorded base
    rig) overrides the relevant fields
    on the per-call CameraSpec: F-theta lens (`pixeldist_to_angle_poly`
    + cx/cy + linear-cde) and, when present in the rig file,
    `bivariate_windshield_model_param` (the windshield's forward +
    inverse 2D polys). Cameras not listed in the rig file fall
    through to the server's recorded intrinsics, so partial overrides
    work. Lets the warm-thin path render an "augmented rig" view
    without dropping to the per-turn CLI's
    `export-custom-rig-trajectory` flow.

Proto stub resolution order:
  1. NUREC_RENDER_CLIENT_PROTO_DIR env var.
  2. ~/.cache/nurec-render-client/proto/  (the default that scripts/setup_protos.sh
     writes to).
  3. Hard error with instructions to run scripts/setup_protos.sh.

Run:
    python3 thin_client.py --cameras camera_front_wide_120fov \\
        --output-dir /path/to/outputs/turn-001/

    python3 thin_client.py --cameras cam_a,cam_b,cam_c,cam_d \\
        --mode batch --output-dir /path/to/outputs/turn-001-multicam/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from pathlib import Path


# ---------------------------------------------------------------------------
# Proto stub resolution (see docstring for resolution order)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve().parent


def _find_proto_dir() -> Path:
    candidates = []
    if env := os.environ.get("NUREC_RENDER_CLIENT_PROTO_DIR"):
        candidates.append(Path(env))
    candidates.append(Path(os.path.expanduser("~/.cache/nurec-render-client/proto")))
    for c in candidates:
        if (c / "nurec" / "grpc" / "protos" / "sensorsim_pb2.py").is_file():
            return c
    print(
        "STOP [thin_client]: NuRec proto stubs not found.\n"
        "  Searched (in order):\n"
        + "".join(f"    - {c}/nurec/grpc/protos/sensorsim_pb2.py\n" for c in candidates)
        + "\n"
        "  Generate them once via:\n"
        f"      bash {_THIS}/setup_protos.sh\n"
        "\n"
        "  Or set NUREC_RENDER_CLIENT_PROTO_DIR to a dir already containing them.",
        file=sys.stderr,
    )
    sys.exit(2)


_PROTO_DIR = _find_proto_dir()
sys.path.insert(0, str(_PROTO_DIR))

import grpc  # noqa: E402
import numpy as np  # noqa: E402

from scipy.spatial.transform import Rotation as R  # noqa: E402
from scipy.spatial.transform import Slerp  # noqa: E402

from nurec.grpc.protos import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc  # noqa: E402


# ---------------------------------------------------------------------------
# SE3 <-> gRPC Pose conversion
# ---------------------------------------------------------------------------


def grpc_pose_to_se3(pose: common_pb2.Pose) -> np.ndarray:
    """Convert a gRPC Pose (Vec3 + Quat) into a 4x4 SE3 matrix (float32)."""
    se3 = np.eye(4, dtype=np.float32)
    # scipy quat is [x, y, z, w]
    se3[:3, :3] = R.from_quat([pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w]).as_matrix()
    se3[:3, 3] = [pose.vec.x, pose.vec.y, pose.vec.z]
    return se3


def se3_to_grpc_pose(se3: np.ndarray) -> common_pb2.Pose:
    """Convert a 4x4 SE3 matrix into a gRPC Pose."""
    quat = R.from_matrix(se3[:3, :3]).as_quat(canonical=False)  # [x, y, z, w]
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=float(se3[0, 3]), y=float(se3[1, 3]), z=float(se3[2, 3])),
        quat=common_pb2.Quat(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])),
    )


_FORMAT_TO_PROTO = {
    "jpeg": sensorsim_pb2.ImageFormat.JPEG,
    "png": sensorsim_pb2.ImageFormat.PNG,
}
_FORMAT_TO_EXT = {"jpeg": "jpeg", "png": "png"}


# ---------------------------------------------------------------------------
# Rig translation/rotation offset (novel-view rendering)
#
# Lifted from nurec.utils.geometry.pose_offsets_to_se3 so the client can
# build its own rig-offset SE3 without importing the (unavailable on the
# host) NuRec Python module. NCore's `euler_2_so3` is replaced with scipy's
# Rotation -- equivalent for an "xyz" euler sequence in degrees.
#
# The composition `T_rig_world @ rig_offset @ T_camera_rig` is what
# `nurec.utils.io.export.render_grpc.py` does; bytewise-comparable result.
# ---------------------------------------------------------------------------
def pose_offsets_to_se3(
    translation: tuple,
    rotation_deg: tuple,
) -> np.ndarray:
    """4x4 SE3 transforming all sensors in the rig frame by translation +
    rotation offsets, matching NuRec's `nurec.utils.geometry.pose_offsets_to_se3`.

    Args:
        translation: (X, Y, Z) translation in meters in rig frame.
        rotation_deg: (yaw, -roll, -pitch) in degrees -- axis convention
            is the val-mode permutation hack from NuRec's geometry module
            (replicated below). Passing (0, 0, 0) is a safe identity.

    Returns:
        4x4 float32 SE3 to left-multiply onto a camera-to-rig pose
        before composing with rig-to-world.
    """
    delta_rot = R.from_euler("xyz", rotation_deg, degrees=True).as_matrix().astype(np.float32)
    # NuRec's val mode permutes rig axes (x,y,z) -> (z,-x,-y) before
    # applying the euler rotation, then permutes back. Net effect: the
    # input "(yaw, -roll, -pitch)" tuple actually rotates around the
    # original (x,y,z) rig axes. Replicate the hack so input semantics
    # match `nurec render`.
    T_PERM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    delta_rot = T_PERM.T @ delta_rot @ T_PERM

    se3 = np.eye(4, dtype=np.float32)
    se3[:3, :3] = delta_rot
    se3[:3, 3] = np.asarray(translation, dtype=np.float32)
    return se3


# ---------------------------------------------------------------------------
# Rig-file intrinsics override
#
# Hyperion-style rig JSONs (e.g. the bundled ../rig-json/rig.json +
# augmented_rig.json) list per-camera optical parameters under
# sensor entries with `protocol == "camera.virtual"`:
#
#   "name":   "camera:cross:left:120fov"          # ':'-separated
#   "properties": {
#       "Model":           "ftheta",
#       "cx": "<float>",   "cy": "<float>",
#       "polynomial":      "<space-separated coeffs>",
#       "polynomial-type": "pixeldistance-to-angle" | "angle-to-pixeldistance",
#       "linear-c": "1", "linear-d": "0", "linear-e": "0",   # optional affine
#       "windshield-horizontal-polynomial":         "<...>",  # optional
#       "windshield-vertical-polynomial":           "<...>",
#       "windshield-horizontal-polynomial-approx-inverse": "<...>",
#       "windshield-vertical-polynomial-approx-inverse":   "<...>",
#   }
#
# We parse those fields once, normalize the camera name from
# `camera:cross:left:120fov` to the gRPC logical_id form
# `camera_cross_left_120fov`, and stash a per-camera override dict.
# At request-build time, that override is merged onto the server-side
# CameraSpec (which ships the recorded USDZ intrinsics): F-theta sub-
# fields are overwritten and, if all four windshield polys are present,
# bivariate_windshield_model_param is populated (which the server uses
# at render time as an additional external distortion on top of F-theta).
# Cameras not listed in the rig file fall through to the recorded
# intrinsics unchanged.
# ---------------------------------------------------------------------------


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split()]


def parse_rig_file(rig_path: Path) -> dict[str, dict]:
    """Parse a Hyperion-style rig JSON.

    Returns a dict mapping `logical_id` (gRPC-style, underscores) to a
    `{"ftheta": ftheta_dict, "windshield": windshield_dict | None,
    "rig_width": int | None, "rig_height": int | None}` record. Cameras
    whose Model is not `ftheta` are skipped (NuRec only supports the
    F-theta lens model on the gRPC override path).
    Cameras missing any of the four windshield polys get
    `windshield=None` and only the F-theta block is overridden.
    `rig_width` / `rig_height` are the calibration resolution of the
    rig's intrinsics (used by `apply_intrinsics_override` to scale
    cx/cy/poly down to the USDZ-baked CameraSpec resolution rather
    than blow it up to render at the rig's native resolution).
    """
    raw = json.loads(Path(rig_path).read_text())
    sensors = raw.get("rig", {}).get("sensors", [])
    out: dict[str, dict] = {}
    for sensor in sensors:
        if sensor.get("protocol") != "camera.virtual":
            continue
        props = sensor.get("properties", {}) or {}
        if props.get("Model") != "ftheta":
            continue
        try:
            poly = _parse_float_list(props["polynomial"])
            cx = float(props["cx"])
            cy = float(props["cy"])
        except (KeyError, ValueError):
            continue
        ftheta = {
            "principal_point_x": cx,
            "principal_point_y": cy,
            "pixeldist_to_angle_poly": poly,
            "polynomial_type": props.get("polynomial-type", "pixeldistance-to-angle"),
        }
        for k_src, k_dst in (
            ("linear-c", "linear_c"),
            ("linear-d", "linear_d"),
            ("linear-e", "linear_e"),
        ):
            if k_src in props:
                try:
                    ftheta[k_dst] = float(props[k_src])
                except ValueError:
                    pass

        rig_width: int | None = None
        rig_height: int | None = None
        try:
            if "width" in props:
                rig_width = int(float(props["width"]))
            if "height" in props:
                rig_height = int(float(props["height"]))
        except ValueError:
            rig_width, rig_height = None, None

        ws = None
        h_fwd = props.get("windshield-horizontal-polynomial")
        v_fwd = props.get("windshield-vertical-polynomial")
        h_inv = props.get("windshield-horizontal-polynomial-approx-inverse")
        v_inv = props.get("windshield-vertical-polynomial-approx-inverse")
        if all(p is not None for p in (h_fwd, v_fwd, h_inv, v_inv)):
            try:
                ws = {
                    "horizontal_poly": _parse_float_list(h_fwd),
                    "vertical_poly": _parse_float_list(v_fwd),
                    "horizontal_poly_inverse": _parse_float_list(h_inv),
                    "vertical_poly_inverse": _parse_float_list(v_inv),
                }
            except ValueError:
                ws = None

        logical_id = sensor["name"].replace(":", "_")
        out[logical_id] = {
            "ftheta": ftheta,
            "windshield": ws,
            "rig_width": rig_width,
            "rig_height": rig_height,
        }
    return out


def _scale_pixeldist_poly(coeffs: list[float], scale: float) -> list[float]:
    """Scale a pixel-distance-domain polynomial to a different pixel scale.

    The F-theta forward poly converts pixel-distance from principal
    point into angle (or vice-versa). When the rig's calibration
    resolution differs from the resolution the CameraSpec's render
    canvas uses, the *same physical ray* falls at a pixel-distance
    that is `scale = render_res / rig_res` times the rig's. To
    preserve the angle/pixeldist mapping at the new scale:

        angle = sum_i c_rig_i * d_rig^i
              = sum_i c_rig_i * (d_render / scale)^i
              = sum_i (c_rig_i / scale^i) * d_render^i

    so the scaled coefficients are `c_rig_i / scale^i`. This applies
    to the `pixeldist_to_angle` direction. For `angle_to_pixeldist`
    the relationship inverts: `c_render_i = c_rig_i * scale`.
    """
    return [c / (scale**i) for i, c in enumerate(coeffs)]


def _scale_angle_poly(coeffs: list[float], scale: float) -> list[float]:
    """Scale an angle-domain polynomial whose output is pixel-distance."""
    return [c * scale for c in coeffs]


def _eval_poly(coeffs: list[float], x: float) -> float:
    return sum(c * (x**i) for i, c in enumerate(coeffs))


def apply_intrinsics_override(
    base_intrinsics: sensorsim_pb2.CameraSpec,
    override: dict | None,
) -> sensorsim_pb2.CameraSpec:
    """Return a CameraSpec copy with rig-file fields overlaid.

    `base_intrinsics` is the per-camera CameraSpec the server reports
    in `get_available_cameras()`. `override` is the per-camera dict
    produced by `parse_rig_file`. Pass `override=None` to no-op.

    The CameraSpec's `resolution_w` / `resolution_h` are deliberately
    preserved from the recorded base (i.e. the USDZ-baked resolution).
    Rig JSONs are typically calibrated at the camera's native capture
    resolution (e.g. 4K for the PAI NuRec sample set), but the USDZ
    is exported at half-res so the warm server's cheaper render
    resolution dominates per-frame cost. To stay on that perf path
    *and* use the rig's intrinsics we scale `cx`, `cy`, and the
    F-theta polynomial coefficients by `render_res / rig_res`, then
    recompute `max_angle` against the new image diagonal. Windshield
    polynomials are post-lens (operate in normalized image-plane
    coordinates) and are resolution-invariant, so they pass through
    verbatim. To render at the rig's native resolution instead, the
    caller would have to override `resolution_w/h` here too -- not
    done by default because that's a 4x per-frame cost on the PAI
    sample set with no demo-perceptible quality gain at the viewer's
    1080p.
    """
    if override is None:
        return base_intrinsics
    spec = sensorsim_pb2.CameraSpec()
    spec.CopyFrom(base_intrinsics)
    ft = spec.ftheta_param
    ftheta = override["ftheta"]

    rig_w = override.get("rig_width")
    rig_h = override.get("rig_height")
    base_w = spec.resolution_w
    base_h = spec.resolution_h
    if rig_w and rig_h and base_w and base_h:
        scale_x = base_w / float(rig_w)
        scale_y = base_h / float(rig_h)
        if abs(scale_x - scale_y) > 1e-3:
            print(
                f"[thin] warning: rig calibration aspect "
                f"({rig_w}x{rig_h}) does not match recorded camera "
                f"resolution ({base_w}x{base_h}); using horizontal "
                f"scale {scale_x:.6f} for both axes",
                file=sys.stderr,
            )
        scale = scale_x
    else:
        if rig_w or rig_h:
            print(
                "[thin] warning: rig file has only one of width/height; "
                "falling back to scale=1.0 (no resolution rescale)",
                file=sys.stderr,
            )
        scale = 1.0

    cx_scaled = ftheta["principal_point_x"] * scale
    cy_scaled = ftheta["principal_point_y"] * scale
    ft.principal_point_x = cx_scaled
    ft.principal_point_y = cy_scaled

    poly_type = ftheta["polynomial_type"].strip().lower()
    if poly_type.startswith("angle"):
        scaled_poly = _scale_angle_poly(ftheta["pixeldist_to_angle_poly"], scale)
        ft.reference_poly = sensorsim_pb2.FthetaCameraParam.ANGLE_TO_PIXELDIST
        del ft.angle_to_pixeldist_poly[:]
        ft.angle_to_pixeldist_poly.extend(scaled_poly)
        del ft.pixeldist_to_angle_poly[:]
    else:
        scaled_poly = _scale_pixeldist_poly(ftheta["pixeldist_to_angle_poly"], scale)
        ft.reference_poly = sensorsim_pb2.FthetaCameraParam.PIXELDIST_TO_ANGLE
        del ft.pixeldist_to_angle_poly[:]
        ft.pixeldist_to_angle_poly.extend(scaled_poly)
        del ft.angle_to_pixeldist_poly[:]

        if base_w and base_h:
            dx = max(cx_scaled, base_w - cx_scaled)
            dy = max(cy_scaled, base_h - cy_scaled)
            max_pixel_dist = float(np.hypot(dx, dy))
            ft.max_angle = float(_eval_poly(scaled_poly, max_pixel_dist))

    if "linear_c" in ftheta or "linear_d" in ftheta or "linear_e" in ftheta:
        had_cde = ft.HasField("linear_cde")
        cde = ft.linear_cde
        cde.linear_c = ftheta.get("linear_c", cde.linear_c if had_cde else 1.0)
        cde.linear_d = ftheta.get("linear_d", cde.linear_d if had_cde else 0.0)
        cde.linear_e = ftheta.get("linear_e", cde.linear_e if had_cde else 0.0)

    ws = override.get("windshield")
    if ws is not None:
        bw = spec.bivariate_windshield_model_param
        bw.reference_poly = sensorsim_pb2.BivariateWindshieldModelParameters.FORWARD
        for src_key, dst_field in (
            ("horizontal_poly", bw.horizontal_poly),
            ("vertical_poly", bw.vertical_poly),
            ("horizontal_poly_inverse", bw.horizontal_poly_inverse),
            ("vertical_poly_inverse", bw.vertical_poly_inverse),
        ):
            del dst_field[:]
            dst_field.extend(ws[src_key])
    return spec


# ---------------------------------------------------------------------------
# Setup RPCs
#
# All stub calls take a `metadata` kwarg (list of (key, value) pairs).
# When --scene-url is set, that metadata carries the
# x-nre-scene-url / x-nre-scene-id headers so the server-side
# SceneDownloadInterceptor (nurec/grpc/serve.py) registers or downloads
# the scene on the *first* RPC, before continuation. After that, the
# scene is in the server's LRU cache for the rest of the session;
# attaching the metadata on every subsequent call is a no-op (the
# interceptor short-circuits when the scene is already loaded).
# ---------------------------------------------------------------------------

# Server-side scene_id validator regex from
# nurec.grpc.downloader.check_safe_scene_id: letters, numbers, _ and -.
_SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def discover_scene_id(stub, metadata) -> str:
    resp = stub.get_available_scenes(common_pb2.Empty(), metadata=metadata)
    if not resp.scene_ids:
        raise RuntimeError("server reports no available scenes")
    if len(resp.scene_ids) > 1:
        print(
            f"[thin] note: multiple scenes available; picking first: {resp.scene_ids[0]}",
            file=sys.stderr,
        )
    return resp.scene_ids[0]


def get_cameras_by_id(stub, scene_id, names, metadata):
    """Return [AvailableCamera] in the order requested by `names`."""
    resp = stub.get_available_cameras(
        sensorsim_pb2.AvailableCamerasRequest(scene_id=scene_id),
        metadata=metadata,
    )
    by_id = {c.logical_id: c for c in resp.available_cameras}
    missing = [n for n in names if n not in by_id]
    if missing:
        raise RuntimeError(f"cameras not found on server: {missing}; available: {sorted(by_id)}")
    return [by_id[n] for n in names]


def get_trajectory(stub, scene_id, metadata):
    """Return the ego trajectory as [(timestamp_us, T_rig_world_4x4), ...]."""
    resp = stub.get_available_trajectories(
        sensorsim_pb2.AvailableTrajectoriesRequest(scene_id=scene_id),
        metadata=metadata,
    )
    if not resp.available_trajectories:
        raise RuntimeError("server returned no trajectories")
    traj = resp.available_trajectories[0].trajectory
    return [(int(p.timestamp_us), grpc_pose_to_se3(p.pose)) for p in traj.poses]


# ---------------------------------------------------------------------------
# Trajectory pose interpolation
#
# The gRPC API exposes only the recorded ego trajectory keyframes (typically
# ~10 Hz from the odometry/lidar channel), not the per-camera frame
# timestamps (typically ~30 Hz). To match `nurec render --frame-step 1`'s
# per-camera-frame cadence (and the 30 fps MP4s most downstream
# encoders expect) the client synthesizes timestamps at a chosen
# `target_fps` spanning the trajectory and interpolates ego poses at
# each.
#
# Interpolation pattern (slerp rotation + lerp translation) lifted from
# `alpasim_grpc.utils.trajectory.Trajectory.interpolate_to_timestamps` --
# the same pattern pacsim uses to drive its per-tick simulation.
# ---------------------------------------------------------------------------


def interpolate_trajectory_at_fps(trajectory, target_fps: float):
    """Resample `trajectory` (list of (ts_us, T_rig_world)) at `target_fps`.

    Returns a new list of (ts_us, T_rig_world) with timestamps from
    trajectory[0].ts to trajectory[-1].ts spaced at 1/target_fps seconds,
    with poses interpolated via scipy Slerp (rotations) + np.interp
    (translations). target_fps <= 0 returns the trajectory unchanged.
    """
    if target_fps is None or target_fps <= 0 or len(trajectory) < 2:
        return trajectory

    ts_keyframes = np.array([t for t, _ in trajectory], dtype=np.uint64)
    T_keyframes = np.stack([T for _, T in trajectory], axis=0)
    quats_keyframes = np.stack(
        [R.from_matrix(T[:3, :3]).as_quat(canonical=False) for T in T_keyframes],
        axis=0,
    )  # [N, 4] in [x, y, z, w]
    trans_keyframes = T_keyframes[:, :3, 3]

    step_us = max(1, int(round(1_000_000 / target_fps)))
    start_us = int(ts_keyframes[0])
    end_us = int(ts_keyframes[-1])
    ts_target = np.arange(start_us, end_us + 1, step_us, dtype=np.uint64)

    rotations_keyframes = R.from_quat(quats_keyframes)
    quats_target = Slerp(ts_keyframes, rotations_keyframes)(ts_target).as_quat()
    trans_target = np.stack(
        [np.interp(ts_target, ts_keyframes, trans_keyframes[:, axis]) for axis in range(3)],
        axis=1,
    )

    out = []
    for i, ts in enumerate(ts_target):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_quat(quats_target[i]).as_matrix()
        T[:3, 3] = trans_target[i]
        out.append((int(ts), T))
    return out


# ---------------------------------------------------------------------------
# RGBRenderRequest builder
# ---------------------------------------------------------------------------


def build_rgb_request(
    *,
    scene_id: str,
    camera,  # AvailableCamerasReturn.AvailableCamera
    T_rig_world: np.ndarray,
    rig_offset_se3: np.ndarray,
    timestamp_us: int,
    height: int,
    image_format: str,
    image_quality: int,
    intrinsics_override: dict | None = None,
) -> sensorsim_pb2.RGBRenderRequest:
    """Build a single RGBRenderRequest for one camera at one ego pose.

    Pose composition matches `nurec.utils.io.export.render_grpc.py`:
        T_camera_world = T_rig_world @ rig_offset_se3 @ T_camera_rig
    Pass identity for `rig_offset_se3` to render the recorded view.
    Pass `intrinsics_override=None` to use the server-side recorded
    CameraSpec verbatim; pass a parsed-rig dict (from
    `parse_rig_file`) to overlay F-theta / windshield fields.
    """
    # Server side, the field is named `rig_to_camera` but it actually carries
    # se3_to_grpc_pose(camera.T_camera_rig). Treat as camera-to-rig.
    T_camera_rig = grpc_pose_to_se3(camera.rig_to_camera)
    T_camera_world = T_rig_world @ rig_offset_se3 @ T_camera_rig
    pose_grpc = se3_to_grpc_pose(T_camera_world)
    intrinsics = apply_intrinsics_override(camera.intrinsics, intrinsics_override)
    return sensorsim_pb2.RGBRenderRequest(
        scene_id=scene_id,
        resolution_h=height,
        # Server convention: w=1 -> width is computed from h * intrinsic
        # aspect ratio. Matches `nurec.utils.io.export.render_grpc.generate_request`.
        resolution_w=1,
        camera_intrinsics=intrinsics,
        # Half-closed interval; +1 for instant shutter.
        frame_start_us=timestamp_us,
        frame_end_us=timestamp_us + 1,
        sensor_pose=sensorsim_pb2.PosePair(start_pose=pose_grpc, end_pose=pose_grpc),
        image_format=_FORMAT_TO_PROTO[image_format],
        image_quality=float(image_quality),
    )


# ---------------------------------------------------------------------------
# Render loops
# ---------------------------------------------------------------------------


def _percentile(sv, p):
    if not sv:
        return 0.0
    k = max(0, min(len(sv) - 1, int(round((p / 100.0) * (len(sv) - 1)))))
    return sv[k]


def _stats(per_pose_ms):
    if not per_pose_ms:
        return {"n": 0}
    sv = sorted(per_pose_ms)
    return {
        "n": len(sv),
        "min_ms": round(sv[0], 3),
        "median_ms": round(sv[len(sv) // 2], 3),
        "avg_ms": round(sum(sv) / len(sv), 3),
        "p95_ms": round(_percentile(sv, 95.0), 3),
        "max_ms": round(sv[-1], 3),
        "sum_s": round(sum(sv) / 1000.0, 3),
    }


def _run_single(
    stub, scene_id, camera, cam_name, trajectory, rig_offset_se3, metadata, out_dir, args, intrinsics_override=None
):
    """One render_rgb call per pose. Single camera."""
    del cam_name  # unused; kept in signature for symmetry with _run_batch
    ext = _FORMAT_TO_EXT[args.image_format]
    per_pose_ms = []
    file_records = []
    for i, (ts_us, T_rig_world) in enumerate(trajectory):
        req = build_rgb_request(
            scene_id=scene_id,
            camera=camera,
            T_rig_world=T_rig_world,
            rig_offset_se3=rig_offset_se3,
            timestamp_us=ts_us,
            height=args.height,
            image_format=args.image_format,
            image_quality=args.image_quality,
            intrinsics_override=intrinsics_override,
        )
        t0 = time.perf_counter()
        resp = stub.render_rgb(req, metadata=metadata)
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000.0
        per_pose_ms.append(wall_ms)
        fname = f"{i:06d}.{ext}"
        (out_dir / fname).write_bytes(resp.image_bytes)
        file_records.append(
            {
                "i": i,
                "file": fname,
                "timestamp_us": ts_us,
                "wall_ms": round(wall_ms, 3),
            }
        )
        if i % 50 == 0 or i + 1 == len(trajectory):
            print(
                f"[thin]   [{i + 1}/{len(trajectory)}] last_call_ms={wall_ms:.1f}",
                file=sys.stderr,
            )
    # Counts are part of the contract so run() can report and exit honestly.
    # This path writes one image per successful RPC and raises otherwise,
    # so every recorded pose is a rendered image.
    return per_pose_ms, file_records, len(per_pose_ms), 0


def _run_batch(
    stub, scene_id, cams, cam_names, trajectory, rig_offset_se3, metadata, out_dir, args, intrinsics_overrides=None
):
    """One batch_render_rgb call per pose. Multiple cameras.

    `intrinsics_overrides`, when provided, is a dict[str, dict] keyed by
    `logical_id`; each camera's per-call CameraSpec is overlaid with
    its entry (cameras absent from the dict use the recorded intrinsics).
    """
    ext = _FORMAT_TO_EXT[args.image_format]
    for name in cam_names:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    per_pose_ms = []
    n_ok_items = 0
    n_failed_items = 0
    file_records = []
    for i, (ts_us, T_rig_world) in enumerate(trajectory):
        items = []
        for cam, name in zip(cams, cam_names):
            override = intrinsics_overrides.get(name) if intrinsics_overrides else None
            req = build_rgb_request(
                scene_id=scene_id,
                camera=cam,
                T_rig_world=T_rig_world,
                rig_offset_se3=rig_offset_se3,
                timestamp_us=ts_us,
                height=args.height,
                image_format=args.image_format,
                image_quality=args.image_quality,
                intrinsics_override=override,
            )
            items.append(
                sensorsim_pb2.BatchRGBRenderRequestItem(
                    camera_name=name,
                    request=req,
                )
            )
        batch_req = sensorsim_pb2.BatchRGBRenderRequest(items=items)
        t0 = time.perf_counter()
        resp = stub.batch_render_rgb(batch_req, metadata=metadata)
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000.0
        per_pose_ms.append(wall_ms)
        per_pose_files = {}
        seen_names: set[str] = set()
        expected_names = set(cam_names)
        for item in resp.items:
            name = item.camera_name
            # Validate the name BEFORE it can reach a filesystem path. `out_dir /
            # name` DISCARDS out_dir entirely when name is absolute, and "../"
            # traverses out of it -- so a bad or hostile camera_name in the
            # response is a write primitive. Only names we ourselves requested,
            # and only once each, may be used to build a path.
            if name not in expected_names:
                print(
                    f"WARN [thin] pose {i} unexpected camera {name!r} in the response",
                    file=sys.stderr,
                )
                n_failed_items += 1
                continue
            if name in seen_names:
                print(
                    f"WARN [thin] pose {i} duplicate response for camera {name}",
                    file=sys.stderr,
                )
                n_failed_items += 1
                continue
            seen_names.add(name)
            if not item.success:
                print(
                    f"WARN [thin] pose {i} camera {name} failed: {item.error_message}",
                    file=sys.stderr,
                )
                n_failed_items += 1
                continue
            n_ok_items += 1
            fname = f"{i:06d}.{ext}"
            (out_dir / name / fname).write_bytes(item.result.image_bytes)
            per_pose_files[name] = fname
        # Requested cameras the server never answered for at all.
        for cam in cam_names:
            if cam not in seen_names:
                print(
                    f"WARN [thin] pose {i} camera {cam} missing from the response",
                    file=sys.stderr,
                )
                n_failed_items += 1
        file_records.append(
            {
                "i": i,
                "files": per_pose_files,
                "timestamp_us": ts_us,
                "wall_ms": round(wall_ms, 3),
            }
        )
        if i % 50 == 0 or i + 1 == len(trajectory):
            print(
                f"[thin]   [{i + 1}/{len(trajectory)}] last_call_ms={wall_ms:.1f}",
                file=sys.stderr,
            )
    return per_pose_ms, file_records, n_ok_items, n_failed_items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args) -> int:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        print("STOP: --cameras requires at least one camera id", file=sys.stderr)
        return 2
    # Every camera id becomes a directory under out_dir. `out_dir / name`
    # DISCARDS out_dir when name is absolute, and ".." climbs out of it, so a
    # camera id must be exactly one ordinary path component. Server membership
    # is NOT a substitute: get_cameras_by_id only proves the server advertises
    # the logical_id, and a server is free to advertise "/tmp/x" or "../x".
    bad = [c for c in cameras if c in (".", "..") or "/" in c or "\\" in c or Path(c).is_absolute()]
    if bad:
        print(
            f"STOP: --cameras entries must be plain names, not paths: {bad}",
            file=sys.stderr,
        )
        return 2
    dupes = sorted({c for c in cameras if cameras.count(c) > 1})
    if dupes:
        print(f"STOP: --cameras has duplicate entries: {dupes}", file=sys.stderr)
        return 2

    mode = args.mode or ("batch" if len(cameras) > 1 else "single")
    if mode == "single" and len(cameras) > 1:
        print(
            f"STOP: --mode single requires exactly 1 camera (got {len(cameras)})",
            file=sys.stderr,
        )
        return 2

    # Scene-loading metadata: --scene-url requires --scene-id (the server
    # uses the id as the cache key + filename; we can't auto-discover it
    # because the scene isn't loaded yet).
    if args.scene_url and not args.scene_id:
        print(
            "STOP: --scene-url requires --scene-id (server uses it as the "
            "cache key + filename; no auto-discovery before the scene is "
            "loaded).",
            file=sys.stderr,
        )
        return 2
    if args.scene_id and not _SCENE_ID_RE.match(args.scene_id):
        print(
            f"STOP: --scene-id={args.scene_id!r} fails server-side validation (allowed: letters, numbers, _ and -).",
            file=sys.stderr,
        )
        return 2
    if args.scene_url:
        metadata = [
            ("x-nre-scene-url", args.scene_url),
            ("x-nre-scene-id", args.scene_id),
        ]
    else:
        metadata = []

    t_connect_start = time.perf_counter()
    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}",
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    stub = sensorsim_pb2_grpc.SensorsimServiceStub(channel)

    scene_id = args.scene_id or discover_scene_id(stub, metadata)
    cams = get_cameras_by_id(stub, scene_id, cameras, metadata)
    trajectory_keyframes = get_trajectory(stub, scene_id, metadata)
    n_keyframes = len(trajectory_keyframes)
    if args.target_fps and args.target_fps > 0:
        trajectory = interpolate_trajectory_at_fps(trajectory_keyframes, args.target_fps)
    else:
        trajectory = trajectory_keyframes

    rig_offset_se3 = pose_offsets_to_se3(
        tuple(args.rig_translation_offset),
        tuple(args.rig_rotation_offset),
    )
    is_novel_view = bool(any(args.rig_translation_offset) or any(args.rig_rotation_offset))

    rig_overrides = None
    rig_overrides_for_cams: dict[str, dict] = {}
    if args.rig_file:
        rig_path = Path(args.rig_file).expanduser().resolve()
        if not rig_path.is_file():
            print(
                f"STOP: --rig-file={rig_path} does not exist or is not a file",
                file=sys.stderr,
            )
            return 2
        try:
            rig_overrides = parse_rig_file(rig_path)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"STOP: failed to parse --rig-file={rig_path}: {exc}",
                file=sys.stderr,
            )
            return 2
        if not rig_overrides:
            print(
                f"WARN [thin] --rig-file={rig_path} parsed 0 ftheta cameras; "
                "intrinsics will fall through to recorded values.",
                file=sys.stderr,
            )
        for name in cameras:
            if name in rig_overrides:
                rig_overrides_for_cams[name] = rig_overrides[name]
        missing = [n for n in cameras if n not in rig_overrides_for_cams]
        if missing:
            print(
                f"[thin]   --rig-file: no override for {missing} (using recorded intrinsics for these cameras)",
                file=sys.stderr,
            )

    setup_s = time.perf_counter() - t_connect_start

    n_total = len(trajectory)
    n = n_total if args.n_poses <= 0 else min(args.n_poses, n_total)
    interp_note = (
        f"  (interpolated from {n_keyframes} keyframes @ {args.target_fps:g} fps)"
        if args.target_fps and args.target_fps > 0
        else "  (raw keyframes)"
    )
    print(
        f"[thin] scene_id={scene_id}  cameras={cameras}  mode={mode}  "
        f"trajectory_poses={n_total}{interp_note}  rendering={n}  "
        f"setup={setup_s * 1000:.0f}ms",
        file=sys.stderr,
    )
    if args.scene_url:
        print(
            f"[thin]   scene_url={args.scene_url}  (server-side load on first RPC; cached thereafter)",
            file=sys.stderr,
        )
    if is_novel_view:
        print(
            f"[thin]   rig_translation_offset={tuple(args.rig_translation_offset)} m  "
            f"rig_rotation_offset={tuple(args.rig_rotation_offset)} deg",
            file=sys.stderr,
        )
    if rig_overrides_for_cams:
        ws_cams = [n for n, ov in rig_overrides_for_cams.items() if ov.get("windshield") is not None]
        print(
            f"[thin]   rig_file: overrode intrinsics for "
            f"{sorted(rig_overrides_for_cams)} "
            f"(windshield-distortion on: {sorted(ws_cams) or 'none'})",
            file=sys.stderr,
        )

    t_render_start = time.perf_counter()
    if mode == "single":
        per_pose_ms, file_records, n_ok_items, n_failed_items = _run_single(
            stub,
            scene_id,
            cams[0],
            cameras[0],
            trajectory[:n],
            rig_offset_se3,
            metadata,
            out_dir,
            args,
            intrinsics_override=rig_overrides_for_cams.get(cameras[0]),
        )
    else:
        per_pose_ms, file_records, n_ok_items, n_failed_items = _run_batch(
            stub,
            scene_id,
            cams,
            cameras,
            trajectory[:n],
            rig_offset_se3,
            metadata,
            out_dir,
            args,
            intrinsics_overrides=rig_overrides_for_cams or None,
        )
    render_s = time.perf_counter() - t_render_start

    stats_full = _stats(per_pose_ms)
    stats_steady = _stats(per_pose_ms[1:]) if len(per_pose_ms) > 1 else stats_full

    (out_dir / "timestamps.json").write_text(json.dumps(file_records, indent=2))
    rig_file_overrides_summary: dict[str, dict] = {}
    for cam_name, ov in rig_overrides_for_cams.items():
        rw = ov.get("rig_width")
        rh = ov.get("rig_height")
        rig_file_overrides_summary[cam_name] = {
            "ftheta": True,
            "windshield": ov.get("windshield") is not None,
            "rig_resolution": [int(rw), int(rh)] if rw and rh else None,
        }
    summary = {
        "mode": f"thin-{mode}",
        "host": f"{args.host}:{args.port}",
        "scene_id": scene_id,
        "cameras": cameras,
        "n_poses_rendered": len(per_pose_ms),
        "setup_s": round(setup_s, 3),
        "render_s": round(render_s, 3),
        "wall_s": round(setup_s + render_s, 3),
        "frame_stats_all": stats_full,
        "frame_stats_excl_frame_0": stats_steady,
        "image_format": args.image_format,
        "image_quality": args.image_quality,
        "height": args.height,
        "rig_translation_offset_m": list(args.rig_translation_offset),
        "rig_rotation_offset_deg": list(args.rig_rotation_offset),
        "is_novel_view": is_novel_view,
        "rig_file": args.rig_file,
        "rig_file_overrides": rig_file_overrides_summary,
        "scene_url": args.scene_url,
        "target_fps": args.target_fps,
        "trajectory_keyframes": n_keyframes,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[thin] DONE  images_ok={n_ok_items}  images_failed={n_failed_items}  "
        f"rpcs={len(per_pose_ms)}  setup={setup_s:.2f}s  "
        f"render={render_s:.2f}s  median_ms={stats_steady.get('median_ms', 0)}  "
        f"out={out_dir}",
        file=sys.stderr,
    )
    # Each RPC can return while every item inside it failed, so exiting 0 on
    # len(per_pose_ms) reports success with no images on disk.
    if n_ok_items == 0:
        print(
            f"STOP [thin] no images rendered ({n_failed_items} item(s) failed).",
            file=sys.stderr,
        )
        return 1
    return 0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--scene-id",
        default=None,
        help="Scene id (auto-discovered via get_available_scenes if "
        "omitted, REQUIRED when --scene-url is set). Allowed: "
        "letters, numbers, _ and - (server-side validation).",
    )
    p.add_argument(
        "--scene-url",
        default=None,
        help="Optional: tell the warm server to load this scene on the "
        "first RPC of this run, before continuing. Sent as "
        "`x-nre-scene-url` gRPC metadata. Use "
        "`file:///path/inside/container/...usdz` (or an absolute "
        "host-relative path that is bind-mounted into the server "
        "container) to register a local file; use "
        "`https://...usdz` to download. The server caches the "
        "scene under its --cache-size LRU (caller's choice; "
        "session_warm_server.sh defaults to --cache-size 3 on "
        "A100-40GB-class hardware) so a session can switch "
        "between several scenes without restarting the warm server. "
        "Requires --scene-id.",
    )
    p.add_argument(
        "--cameras",
        required=True,
        help="Camera id, or comma-separated list for batch mode.",
    )
    p.add_argument(
        "--mode",
        choices=["single", "batch"],
        default=None,
        help="Auto: 'single' for 1 camera, 'batch' for >1.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--n-poses",
        type=int,
        default=0,
        help="Limit number of trajectory poses to render (0 = all).",
    )
    p.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Resample the recorded ego trajectory to this many poses per "
        "second via slerp (rotations) + lerp (translations) before "
        "rendering. The gRPC API only exposes the recorded trajectory "
        "keyframes (typically ~10 Hz from the odometry channel), but "
        "`nurec render --frame-step 1` renders at the camera's native "
        "~30 Hz cadence; resampling keeps the thin-client output "
        "frame-count comparable with the CLI baseline (~30 Hz "
        "default) and downstream 30 fps MP4 encoders. Set <= 0 "
        "to disable and render the raw keyframes verbatim. Same "
        "pattern as pacsim's per-tick simulation drive.",
    )
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg")
    p.add_argument(
        "--image-quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (server default 95). Ignored for PNG.",
    )
    p.add_argument(
        "--rig-translation-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Rig translation offset in meters in rig frame (X Y Z). "
        "Default 0 0 0 (renders the recorded view). Matches "
        "`nurec render --rig-translation-offset`.",
    )
    p.add_argument(
        "--rig-rotation-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("YAW", "NROLL", "NPITCH"),
        help="Rig rotation offset in degrees: (yaw, -roll, -pitch). "
        "Default 0 0 0. Axis convention matches "
        "`nurec render --rig-rotation-offset` -- the val-mode axis "
        "permutation hack is replicated client-side so input "
        "tuples are bytewise compatible.",
    )
    p.add_argument(
        "--rig-file",
        default=None,
        help="Optional path to a Hyperion-style rig JSON to override "
        "the per-camera intrinsics that the warm server reports "
        "from the USDZ-baked rig. F-theta lens (cx/cy, "
        "pixeldist_to_angle_poly, optional linear-cde) is "
        "overridden on every listed camera; bivariate windshield "
        "distortion (forward + inverse polys) is set when all "
        "four polynomials are present in the rig. Cameras not "
        "listed in the rig file fall through to the recorded "
        "intrinsics. Camera-name normalization is colon -> "
        "underscore (e.g. `camera:cross:left:120fov` matches "
        "logical_id `camera_cross_left_120fov`).",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
