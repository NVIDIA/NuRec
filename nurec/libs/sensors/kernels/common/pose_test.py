# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from nurec.libs.sensors.kernels.common.pose import DynamicPose, Pose


def test_control_times_cache_is_separate_for_inference_and_autograd() -> None:
    pose = Pose(
        translation=torch.zeros(3),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    dynamic_pose = DynamicPose(pose, pose)

    with torch.inference_mode():
        inference_control_times = dynamic_pose.to_trajectory().control_times
        assert inference_control_times.is_inference()
        assert dynamic_pose.to_trajectory().control_times is inference_control_times

    control_times = dynamic_pose.to_trajectory().control_times

    assert not control_times.is_inference()
    assert control_times is not inference_control_times
    assert dynamic_pose.to_trajectory().control_times is control_times
    with torch.no_grad():
        assert dynamic_pose.to_trajectory().control_times is control_times

    differentiable_pose = torch.ones(2, requires_grad=True)
    (differentiable_pose * control_times).sum().backward()
    torch.testing.assert_close(differentiable_pose.grad, control_times)
