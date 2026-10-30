# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU parity test for the fused render-pack kernel.

Serve-grpc's CUDA IPC delivery transport replaces the PyTorch chain
`(raster*255).clamp(0,255).to(uint8)` + HWC->CHW permute with a single CUDA kernel that
writes directly into the transport's destination buffer. This test asserts the kernel is
bit-exact vs that reference. Needs a GPU; no transport-specific host libraries.
"""

import pytest
import torch

from nurec.libs.render_pack import _render_pack  # type: ignore[attr-defined]


def _reference_chw_u8(src_hwc_f32: torch.Tensor) -> torch.Tensor:
    # Matches the kernel: scale, clamp, truncate-to-uint8, HWC->CHW, flatten.
    return (src_hwc_f32 * 255.0).clamp(0, 255).to(torch.uint8).permute(2, 0, 1).reshape(-1).contiguous()


@pytest.mark.parametrize("h,w", [(64, 48), (65, 47), (1, 1), (270, 480)])
def test_render_pack_matches_torch(h: int, w: int):
    if not torch.cuda.is_available():
        pytest.skip("render_pack parity needs a CUDA device")
    torch.manual_seed(0)
    # Include out-of-[0,1] values to exercise saturation on both ends.
    src = (torch.rand(h, w, 3, device="cuda", dtype=torch.float32) * 1.4 - 0.2).contiguous()
    dst = torch.zeros(3 * h * w, device="cuda", dtype=torch.uint8)

    _render_pack.run(src.data_ptr(), dst.data_ptr(), w, h)

    assert torch.equal(dst, _reference_chw_u8(src))
