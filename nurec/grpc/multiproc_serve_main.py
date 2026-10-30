# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin executable entrypoint for the multi-process gRPC serving CLI.

The CLI implementation, argument parser, launch logic, and exit-code mapping all live in
``nurec.grpc.multiproc_serve`` so tests can call ``main()`` directly and the module stays an
importable library. This wrapper exists only so the ``//nurec/grpc:multiproc_serve`` binary has
its own compile source and the library remains the single compile owner of ``multiproc_serve.py``.
"""

from nurec.grpc.multiproc_serve import main


if __name__ == "__main__":
    raise SystemExit(main())
