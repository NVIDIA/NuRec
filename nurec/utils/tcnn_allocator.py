# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

from typing import Any


_configured_modules: set[int] = set()


def use_torch_allocator_for_tcnn(tcnn_module: Any) -> None:
    module_id = id(tcnn_module)
    if module_id in _configured_modules:
        return

    set_allocator = getattr(tcnn_module, "set_gpu_memory_allocator", None)
    if set_allocator is None:
        return

    # Pre-load dynamic torch dependencies before importing the torch-specific
    # allocator bridge shared object.
    import torch  # noqa: F401

    allocator_bridge = importlib.import_module("nurec.utils.libtcnn_allocator_cc")
    set_allocator(allocator_bridge.torch_cuda_allocate(), allocator_bridge.torch_cuda_deallocate())
    _configured_modules.add(module_id)
