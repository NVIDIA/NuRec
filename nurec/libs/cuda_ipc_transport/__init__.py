# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA IPC transport for NRE — pybind11 bindings around cudaIpcGetMemHandle.

The native `_nre_cuda_ipc` module is always built. The import is still wrapped
defensively: if it ever fails, `IpcManager` is None so importing this package
still succeeds — callers gate on `IpcManager is not None`.
"""

try:
    from nurec.libs.cuda_ipc_transport._nre_cuda_ipc import IpcManager  # type: ignore
except ImportError:
    IpcManager = None  # type: ignore


__all__ = ["IpcManager"]
