# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the focused SE3 <-> gRPC pose helpers (nurec.grpc.pose).

These pin the behavior of the helpers extracted out of serve.py so the lightweight benchmark
harness can import them without the GPU serving stack (finding F008).
"""

import numpy as np
import pytest

from scipy.spatial.transform import Rotation as R

import nurec.grpc.protos.common_pb2 as common_pb2

from nurec.grpc.pose import grpc_pose_to_torch_se3, se3_to_grpc_pose


def _pose(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> common_pb2.Pose:
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=x, y=y, z=z),
        quat=common_pb2.Quat(x=qx, y=qy, z=qz, w=qw),
    )


def test_grpc_pose_to_torch_se3_identity_rotation_and_translation() -> None:
    se3 = grpc_pose_to_torch_se3(_pose(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))

    assert tuple(se3.shape) == (4, 4)
    np.testing.assert_allclose(se3[:3, 3].numpy(), [1.0, 2.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(se3[:3, :3].numpy(), np.eye(3), atol=1e-5)
    np.testing.assert_allclose(se3[3].numpy(), [0.0, 0.0, 0.0, 1.0], atol=1e-5)


def test_se3_to_grpc_pose_round_trips_rotation_and_translation() -> None:
    quat = R.from_euler("z", 90, degrees=True).as_quat(canonical=False)  # [x, y, z, w]
    original = _pose(0.5, -1.5, 2.0, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    se3 = grpc_pose_to_torch_se3(original).numpy()
    round_tripped = se3_to_grpc_pose(se3)

    assert round_tripped.vec.x == pytest.approx(0.5, abs=1e-5)
    assert round_tripped.vec.y == pytest.approx(-1.5, abs=1e-5)
    assert round_tripped.vec.z == pytest.approx(2.0, abs=1e-5)

    rot_in = R.from_quat([original.quat.x, original.quat.y, original.quat.z, original.quat.w]).as_matrix()
    rot_out = R.from_quat(
        [round_tripped.quat.x, round_tripped.quat.y, round_tripped.quat.z, round_tripped.quat.w]
    ).as_matrix()
    np.testing.assert_allclose(rot_in, rot_out, atol=1e-5)
