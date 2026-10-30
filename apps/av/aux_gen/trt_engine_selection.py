# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Mask2Former TensorRT-engine selection.

Kept torch-free so the (GPU model -> engine) mapping can be unit-tested in the
pre-merge ``build+test`` job. The full TRT-vs-torch accuracy comparison
(``trt_torch_comparison_test.py``) needs a GPU + inference and stays a manual,
on-demand test.
"""

import re

from typing import Optional


# Native TensorRT engines keyed by (GPU model short-name, CPU arch). A TensorRT plan
# is GPU-model-specific: a plan built for one GPU can depend on model details (e.g. SM
# count) and is not guaranteed to run correctly on a different GPU of the same compute
# capability, so a native engine is used only on the exact model it was built for. Build
# one per GPU model.
TRT_CHECKPOINTS = {
    ("A40", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.6.A40.x86_64.fp16.dynamic.engine",
    ("L40", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.9.L40.x86_64.fp16.dynamic.engine",
    ("L40", "aarch64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.9.L40.aarch64.fp16.dynamic.engine",
    ("A100", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.0.A100.x86_64.fp16.dynamic.engine",
    ("A6000", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.6.A6000.x86_64.fp16.dynamic.engine",
    ("5880", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.9.5880.x86_64.fp16.dynamic.engine",
    ("L4", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.9.L4.x86_64.fp16.dynamic.engine",
}

# Generic hardware-compatible fallback per CPU arch: one engine that runs on any
# Ampere-or-newer GPU (~20% slower than a native engine, still ~4x the PyTorch path),
# used when no native engine matches. GPUs older than Ampere (sm < 8.0) cannot load it
# and fall back to PyTorch.
TRT_FALLBACK_CHECKPOINTS = {
    "x86_64": "mask2former_dinov2_b1_0428.trt10.13.ampereplus.x86_64.fp16.dynamic.engine",
    "aarch64": "mask2former_dinov2_b1_0428.trt10.13.ampereplus.aarch64.fp16.dynamic.engine",
}


def select_trt_checkpoint(
    gpu_name: Optional[str],
    sm_version: Optional[str],
    machine: str,
    trt_checkpoints: dict = TRT_CHECKPOINTS,
    trt_fallback_checkpoints: dict = TRT_FALLBACK_CHECKPOINTS,
) -> Optional[str]:
    """Return the Mask2Former engine filename to use, or ``None`` to fall back to PyTorch.

    Selection order: (1) native engine for this GPU model, matched by whole-word GPU
    name so "L40" matches "NVIDIA L40" and "A100" matches "NVIDIA A100-SXM4-80GB" (the
    hyphen is a boundary) but "L40" does NOT match "L40S"; (2) the generic Ampere+
    fallback for any sm >= 8.0 GPU without a native engine; (3) ``None`` otherwise.

    Args:
        gpu_name: ``torch.cuda.get_device_name(0)`` (or ``None`` when no GPU).
        sm_version: compute capability as ``"<major>.<minor>"`` (or ``None``).
        machine: ``platform.machine()`` (CPU arch, e.g. "x86_64" / "aarch64").
    """
    if not gpu_name:
        return None
    native = next(
        (
            engine
            for (gpu_model, arch), engine in trt_checkpoints.items()
            if arch == machine and re.search(rf"\b{re.escape(gpu_model)}\b", gpu_name)
        ),
        None,
    )
    if native is not None:
        return native
    if sm_version and tuple(int(x) for x in sm_version.split(".")) >= (8, 0):  # Ampere or newer
        return trt_fallback_checkpoints.get(machine)
    return None
