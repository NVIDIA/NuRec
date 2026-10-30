# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused SE3 <-> gRPC pose conversion helpers.

These live in their own module (rather than in the monolithic ``nurec.grpc.serve``) so that
lightweight clients and tooling -- notably the concurrent-render benchmark harness -- can
import the pure pose/SE3 math without transitively pulling in the entire GPU serving stack
(gaussians, render, viewer, pytorch_lightning, ...). ``nurec.grpc.serve`` re-exports these for
backwards compatibility.
"""

from __future__ import annotations

import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

import nurec.grpc.protos.common_pb2 as grpc_types

from nurec.utils.geometry import tquat_to_se3_matrix


# Re-exported from nurec.grpc.serve via the PEP 484 `import X as X` idiom.
def grpc_pose_to_tquat(pose: grpc_types.Pose) -> torch.Tensor:
    """
    Converts a single gRPC Pose into a 7d tquat (translation + quaternion) defined as [x,y,z,q_x,q_y,q_z,q_w]

    Args:
        single gRPC Pose [grpc_types.Pose]

    Returns:
        single 7d tquat List[float]
    """
    return torch.tensor([pose.vec.x, pose.vec.y, pose.vec.z, pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w])


def tquat_to_grpc_pose(tquat: np.ndarray | torch.Tensor) -> grpc_types.Pose:
    """
    Converts a single 7d tquat consisting of format [x,y,z,q_x,q_y,q_z,q_w] into a gRPC Pose

    Args:
        single tquat [np.ndarray | torch.Tensor]

    Returns:
        single gRPC Pose [grpc_types.Pose]
    """
    assert tquat.shape == (7,)
    return grpc_types.Pose(
        vec=grpc_types.Vec3(
            x=tquat[0].item(),
            y=tquat[1].item(),
            z=tquat[2].item(),
        ),
        quat=grpc_types.Quat(
            x=tquat[3].item(),
            y=tquat[4].item(),
            z=tquat[5].item(),
            w=tquat[6].item(),
        ),
    )


def grpc_pose_to_torch_se3(grpc_pose: grpc_types.Pose, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Converts a single gRPC Pose into a SE3 4x4 torch Tensor

    Args:
        single gRPC Pose [grpc_types.Pose]

    Returns:
        single torch Tensor [torch.Tensor]
    """
    return tquat_to_se3_matrix(grpc_pose_to_tquat(grpc_pose).to(dtype))


def se3_to_grpc_pose(se3: np.ndarray) -> grpc_types.Pose:
    """
    Converts a single SE3 4x4 matrix into a gRPC Pose

    Args:
        single SE3 matrix [np.ndarray]

    Returns:
        single gRPC Pose [grpc_types.Pose]
    """
    quat = R.from_matrix(se3[..., :3, :3]).as_quat(canonical=False)
    vec3 = se3[..., :3, 3]

    return grpc_types.Pose(
        vec=grpc_types.Vec3(
            x=vec3[0],
            y=vec3[1],
            z=vec3[2],
        ),
        quat=grpc_types.Quat(
            x=quat[0],
            y=quat[1],
            z=quat[2],
            w=quat[3],
        ),
    )
