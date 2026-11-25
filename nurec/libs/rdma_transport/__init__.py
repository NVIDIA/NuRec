# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDMA transport for NRE — pybind11 bindings around libibverbs.

Only built with `--config=rdma`, and importing it loads libibverbs.so.1. serve.py imports
this package defensively and reports why it is unavailable when it is not.
"""

from nurec.libs.rdma_transport._nre_rdma import PreconditionError, RdmaManager  # type: ignore


__all__ = ["PreconditionError", "RdmaManager"]
