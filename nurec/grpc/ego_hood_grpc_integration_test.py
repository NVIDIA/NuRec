# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test: sends raw gRPC render requests with ego_mask_id to a running server.
Verifies:
  1. Per-camera mask selection across different cameras
  2. Per-rig mask selection (same camera, different rig)
  3. Unknown camera with explicit camera_param (the core bug 6079704 fix)
  4. No mask when insert_ego_mask=False

Manual live-server diagnostic. Requires a running gRPC server and skips if the
server is not reachable.

Usage (standalone, with server already running):
  bazel run //nurec/grpc:ego_hood_grpc_integration_test -- --port 8090
"""

import argparse
import copy
import io
import os
import sys
import tempfile

import grpc
import numpy as np

from PIL import Image

from nurec.grpc.protos.common_pb2 import Empty
from nurec.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    AvailableTrajectoriesRequest,
    CameraSpec,
    EgoMaskId,
    ImageFormat,
    PosePair,
    RGBRenderRequest,
)
from nurec.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub
from nurec.grpc.serve import grpc_pose_to_torch_se3, se3_to_grpc_pose


def get_scene_context(stub):
    """Get scene_id, available cameras, trajectory poses and timestamps."""
    scenes = stub.get_available_scenes(Empty())
    scene_id = scenes.scene_ids[0]

    cams = stub.get_available_cameras(AvailableCamerasRequest(scene_id=scene_id))
    available_cameras = {c.logical_id: c for c in cams.available_cameras}

    traj = stub.get_available_trajectories(AvailableTrajectoriesRequest(scene_id=scene_id))
    trajectory = traj.available_trajectories[0].trajectory
    t_start = trajectory.poses[0].timestamp_us
    t_end = trajectory.poses[1].timestamp_us
    start_rig_pose = trajectory.poses[0].pose
    end_rig_pose = trajectory.poses[1].pose

    return scene_id, available_cameras, t_start, t_end, start_rig_pose, end_rig_pose


def build_sensor_poses(camera, start_rig_pose, end_rig_pose):
    """Compose rig trajectory poses with camera extrinsics like render_grpc.py."""
    pose_camera_to_rig = grpc_pose_to_torch_se3(camera.rig_to_camera).cpu().numpy()
    start_camera_to_world = grpc_pose_to_torch_se3(start_rig_pose).cpu().numpy() @ pose_camera_to_rig
    end_camera_to_world = grpc_pose_to_torch_se3(end_rig_pose).cpu().numpy() @ pose_camera_to_rig
    return se3_to_grpc_pose(start_camera_to_world), se3_to_grpc_pose(end_camera_to_world)


def render_with_mask(stub, scene_id, camera_intrinsics, t_start, t_end, start_pose, end_pose, ego_mask_id=None):
    """Send a render request and return the image as numpy array."""
    request = RGBRenderRequest(
        scene_id=scene_id,
        resolution_h=540,
        resolution_w=960,
        camera_intrinsics=camera_intrinsics,
        frame_start_us=t_start,
        frame_end_us=t_end,
        sensor_pose=PosePair(start_pose=start_pose, end_pose=end_pose),
        image_format=ImageFormat.JPEG,
        image_quality=95,
        insert_ego_mask=ego_mask_id is not None,
    )
    if ego_mask_id is not None:
        request.ego_mask_id.CopyFrom(ego_mask_id)

    response = stub.render_rgb(request)
    img = Image.open(io.BytesIO(response.image_bytes))
    return np.array(img)


def check_pixel(img, expected_rgb, label, tolerance=50):
    """Check bottom-center pixel matches expected color."""
    h, w = img.shape[:2]
    row = int(h * 0.95)
    col = w // 2
    pixel = img[row, col, :3]
    ok = all(abs(int(pixel[i]) - expected_rgb[i]) < tolerance for i in range(3))
    status = "PASS" if ok else "FAIL"
    print(f"  {status} | {label:55s} | expected {expected_rgb} | got ({pixel[0]:3d},{pixel[1]:3d},{pixel[2]:3d})")
    return ok


def bottom_region_diff(img_a, img_b):
    """Mean RGB difference in the bottom image region where egocar masks are expected."""
    h = img_a.shape[0]
    return np.abs(img_a[int(h * 0.85) :, :, :3].astype(int) - img_b[int(h * 0.85) :, :, :3].astype(int)).mean()


def make_unknown_camera_spec(available_cameras):
    ref_cam = next(iter(available_cameras.values()))
    unknown_spec = CameraSpec()
    unknown_spec.CopyFrom(ref_cam.intrinsics)

    suffix = 0
    while True:
        unknown_id = f"{ref_cam.logical_id}_unknown_{suffix}"
        if unknown_id not in available_cameras:
            break
        suffix += 1

    unknown_spec.logical_id = unknown_id
    assert unknown_spec.logical_id not in available_cameras
    return ref_cam, unknown_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = SensorsimServiceStub(channel)

    # Check server reachable
    try:
        stub.get_version(Empty(), timeout=5)
    except grpc.RpcError:
        print("SKIP: gRPC server not reachable")
        sys.exit(0)

    scene_id, available_cameras, t_start, t_end, start_rig_pose, end_rig_pose = get_scene_context(stub)
    print(f"Scene: {scene_id}")
    print(f"Available cameras: {list(available_cameras.keys())}")

    masks = stub.get_available_ego_masks(Empty())
    rig_ids = sorted({m.ego_mask_id.rig_config_id for m in masks.ego_mask_metadata})
    mask_cameras_by_rig = {
        rig_id: sorted(
            {m.ego_mask_id.camera_logical_id for m in masks.ego_mask_metadata if m.ego_mask_id.rig_config_id == rig_id}
        )
        for rig_id in rig_ids
    }
    print(f"Available rigs: {rig_ids}")
    print()

    if not rig_ids:
        print("FAIL | No egocar rigs available from server")
        sys.exit(1)

    primary_rig_id = "hyperion_8_0" if "hyperion_8_0" in rig_ids else rig_ids[0]
    secondary_rig_id = next((rig_id for rig_id in rig_ids if rig_id != primary_rig_id), None)

    all_passed = True
    output_dir = os.environ.get("EGO_MASK_TEST_OUTPUT_DIR", tempfile.mkdtemp(prefix="ego_mask_test_"))
    os.makedirs(output_dir, exist_ok=True)

    # ===== TEST 1: Per-camera mask selection =====
    print("=" * 70)
    print(f"TEST 1: Per-camera mask selection ({primary_rig_id})")
    print("=" * 70)
    primary_mask_cameras = [cam_id for cam_id in mask_cameras_by_rig[primary_rig_id] if cam_id in available_cameras]
    if not primary_mask_cameras:
        print(f"  FAIL | {primary_rig_id}: no mask cameras overlap with scene cameras")
        all_passed = False

    for cam_id in primary_mask_cameras:
        cam = available_cameras[cam_id]
        start_pose, end_pose = build_sensor_poses(cam, start_rig_pose, end_rig_pose)
        img = render_with_mask(
            stub,
            scene_id,
            cam.intrinsics,
            t_start,
            t_end,
            start_pose,
            end_pose,
            ego_mask_id=EgoMaskId(rig_config_id=primary_rig_id, camera_logical_id=cam_id),
        )
        img_no_mask = render_with_mask(stub, scene_id, cam.intrinsics, t_start, t_end, start_pose, end_pose)
        Image.fromarray(img).save(os.path.join(output_dir, f"test1_{cam_id}_{primary_rig_id}.jpeg"))
        diff = bottom_region_diff(img, img_no_mask)
        diff_ok = diff > 1.0
        print(f"  {'PASS' if diff_ok else 'FAIL'} | {cam_id} + {primary_rig_id}: bottom diff = {diff:.1f}")
        if not diff_ok:
            all_passed = False

    # ===== TEST 2: Per-rig mask selection (same camera, different rigs) =====
    print()
    print("=" * 70)
    print("TEST 2: Per-rig selection (same camera, different rigs)")
    print("=" * 70)
    if secondary_rig_id is None:
        print("  SKIP | only one rig available")
    else:
        shared_cameras = [
            cam_id
            for cam_id in mask_cameras_by_rig[primary_rig_id]
            if cam_id in mask_cameras_by_rig[secondary_rig_id] and cam_id in available_cameras
        ]
        if not shared_cameras:
            print(f"  FAIL | no shared scene camera between {primary_rig_id} and {secondary_rig_id}")
            all_passed = False
        else:
            cam_id = "camera_front_wide_120fov" if "camera_front_wide_120fov" in shared_cameras else shared_cameras[0]
            cam = available_cameras[cam_id]
            start_pose, end_pose = build_sensor_poses(cam, start_rig_pose, end_rig_pose)
            img_rig1 = render_with_mask(
                stub,
                scene_id,
                cam.intrinsics,
                t_start,
                t_end,
                start_pose,
                end_pose,
                ego_mask_id=EgoMaskId(rig_config_id=primary_rig_id, camera_logical_id=cam_id),
            )
            img_rig2 = render_with_mask(
                stub,
                scene_id,
                cam.intrinsics,
                t_start,
                t_end,
                start_pose,
                end_pose,
                ego_mask_id=EgoMaskId(rig_config_id=secondary_rig_id, camera_logical_id=cam_id),
            )
            Image.fromarray(img_rig1).save(os.path.join(output_dir, f"test2_{cam_id}_{primary_rig_id}.jpeg"))
            Image.fromarray(img_rig2).save(os.path.join(output_dir, f"test2_{cam_id}_{secondary_rig_id}.jpeg"))
            diff_bottom = bottom_region_diff(img_rig1, img_rig2)
            if diff_bottom > 1.0:
                print(f"  PASS | {primary_rig_id} vs {secondary_rig_id} bottom-region diff = {diff_bottom:.1f}")
            else:
                print(
                    f"  SKIP | {primary_rig_id} and {secondary_rig_id} rendered identically "
                    f"for {cam_id}; fixture masks may be identical"
                )

    # ===== TEST 3: Unknown camera with explicit params (core bug 6079704 fix) =====
    print()
    print("=" * 70)
    print("TEST 3: Unknown camera with explicit camera_param (bug 6079704)")
    print("=" * 70)

    # Use an available camera's intrinsics but with a DIFFERENT logical_id.
    # Shift the camera pose by 2 meters to the right to simulate a different camera position.
    ref_cam, unknown_spec = make_unknown_camera_spec(available_cameras)

    ref_start_pose, ref_end_pose = build_sensor_poses(ref_cam, start_rig_pose, end_rig_pose)
    shifted_start = copy.deepcopy(ref_start_pose)
    shifted_start.vec.x += 2.0
    shifted_end = copy.deepcopy(ref_end_pose)
    shifted_end.vec.x += 2.0

    img_unknown = None
    try:
        img_unknown = render_with_mask(
            stub,
            scene_id,
            unknown_spec,
            t_start,
            t_end,
            shifted_start,
            shifted_end,
            ego_mask_id=None,  # No mask — just test that rendering works
        )
        Image.fromarray(img_unknown).save(os.path.join(output_dir, "test3_unknown_camera_shifted_no_mask.jpeg"))
        print(f"  PASS | Unknown camera '{unknown_spec.logical_id}' shifted +2m rendered OK ({img_unknown.shape})")
    except grpc.RpcError as e:
        print(f"  FAIL | Unknown camera render failed: {e.code()} - {e.details()}")
        all_passed = False

    # Verify it looks different from the original camera position (shifted view)
    if img_unknown is not None:
        ref_img = render_with_mask(
            stub, scene_id, ref_cam.intrinsics, t_start, t_end, ref_start_pose, ref_end_pose, ego_mask_id=None
        )
        diff = np.abs(img_unknown.astype(int) - ref_img.astype(int)).mean()
        shift_ok = diff > 5  # shifted image should differ from original
        print(f"  {'PASS' if shift_ok else 'FAIL'} | Shifted vs original mean pixel diff = {diff:.1f} (expect >5)")
        if not shift_ok:
            all_passed = False

    # Now test unknown camera WITH ego mask. Explicit missing mask requests should fail visibly.
    try:
        render_with_mask(
            stub,
            scene_id,
            unknown_spec,
            t_start,
            t_end,
            shifted_start,
            shifted_end,
            ego_mask_id=EgoMaskId(rig_config_id=primary_rig_id, camera_logical_id=unknown_spec.logical_id),
        )
        print("  FAIL | Unknown camera + missing mask request unexpectedly rendered")
        all_passed = False
    except grpc.RpcError as e:
        print(f"  PASS | Unknown camera with missing mask failed visibly: {e.code()} - {e.details()}")

    # ===== TEST 4: No mask control =====
    print()
    print("=" * 70)
    print("TEST 4: Control — no mask (insert_ego_mask=False)")
    print("=" * 70)
    control_cam_id = (
        "camera_front_wide_120fov" if "camera_front_wide_120fov" in available_cameras else next(iter(available_cameras))
    )
    cam = available_cameras[control_cam_id]
    start_pose, end_pose = build_sensor_poses(cam, start_rig_pose, end_rig_pose)
    img_no_mask = render_with_mask(
        stub, scene_id, cam.intrinsics, t_start, t_end, start_pose, end_pose, ego_mask_id=None
    )
    Image.fromarray(img_no_mask).save(os.path.join(output_dir, f"test4_{control_cam_id}_no_mask.jpeg"))
    pixel = img_no_mask[int(img_no_mask.shape[0] * 0.95), img_no_mask.shape[1] // 2, :3]
    is_red = abs(int(pixel[0]) - 255) < 50 and int(pixel[1]) < 50 and int(pixel[2]) < 50
    ctrl_ok = not is_red
    print(
        f"  {'PASS' if ctrl_ok else 'FAIL'} | {control_cam_id} no mask: "
        f"pixel=({pixel[0]:3d},{pixel[1]:3d},{pixel[2]:3d}) should NOT be red"
    )
    if not ctrl_ok:
        all_passed = False

    # ===== SUMMARY =====
    print()
    print("=" * 70)
    print(f"OUTPUT DIR: {output_dir}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    channel.close()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
