# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nurec.libs.sensors.kernels.cameras import camera_rays_to_world_rays_batch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_camera_batch_uses_input_device_and_stream() -> None:
    with torch.cuda.device(0):
        stream = torch.cuda.Stream(device=1)
        with torch.cuda.stream(stream):
            camera_rays = torch.tensor([[1.0, 0.0, 0.0]], device="cuda:1")
            translations = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda:1")
            rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, device="cuda:1")
            with torch.cuda.device(0):
                world_rays = camera_rays_to_world_rays_batch(camera_rays, translations, rotations)
                assert torch.cuda.current_device() == 0
            consumed = world_rays + 1
        assert torch.cuda.current_device() == 0
        stream.synchronize()

    assert world_rays.device == torch.device("cuda:1")
    expected = torch.tensor([[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0]], [[4.0, 5.0, 6.0, 1.0, 0.0, 0.0]]])
    torch.testing.assert_close(consumed.cpu(), expected + 1)
