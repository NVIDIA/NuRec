# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused render-pack CUDA kernel shared by serve-grpc's zero-copy delivery transports.

This package has no Python-level API of its own — it only carries the `render_pack_cuda`
BUILD target and its test-only `_render_pack` pybind binding. Transport packages (e.g.
`nurec.libs.cuda_ipc_transport`) depend on `//nurec/libs/render_pack:render_pack_cuda` directly.
"""
