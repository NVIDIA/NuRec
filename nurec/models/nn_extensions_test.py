# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from nurec.models.nn_extensions import BufferList


def test_buffer_list() -> None:
    class Example(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.blist = BufferList([])

    example = Example()
    example.blist = BufferList([torch.randn(10, 10) for i in range(10)])

    example.to("cuda")
    for i in range(10):
        assert example.blist[i].device.type == "cuda"

    example.to("cpu")
    for i in range(10):
        assert example.blist[i].device.type == "cpu"
        assert example.blist[i].dtype == torch.float32

    example.blist.to(dtype=torch.float16)
    for i in range(10):
        assert example.blist[i].dtype == torch.float16
