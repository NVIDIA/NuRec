# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Mask2Former TensorRT-engine selection.

Kept torch-free so the (SM/CPU architecture -> engine) mapping can be unit-tested
in the pre-merge ``build+test`` job. The full TRT-vs-torch accuracy comparison
(``trt_torch_comparison_test.py``) needs a GPU + inference and stays a manual,
on-demand test.
"""

import re

from typing import Optional


# These plans use SAME_COMPUTE_CAPABILITY, so GPU model names do not constrain
# their use. CPU architecture and the full compute capability must match.
TRT_CHECKPOINTS = {
    ("8.0", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.0.samecc.x86_64.fp16.dynamic.engine",
    ("8.6", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.6.samecc.x86_64.fp16.dynamic.engine",
    ("8.9", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm8.9.samecc.x86_64.fp16.dynamic.engine",
    ("9.0", "x86_64"): "mask2former_dinov2_b1_0428.trt10.13.sm9.0.samecc.x86_64.fp16.dynamic.engine",
    ("10.3", "aarch64"): "mask2former_dinov2_b1_0428.trt10.13.sm10.3.samecc.aarch64.fp16.dynamic.engine",
}

# Generic fallback per CPU architecture when no same-SM plan matches.
# GPUs older than Ampere (sm < 8.0) use PyTorch.
TRT_FALLBACK_CHECKPOINTS = {
    "x86_64": "mask2former_dinov2_b1_0428.trt10.13.ampereplus.x86_64.fp16.dynamic.engine",
    "aarch64": "mask2former_dinov2_b1_0428.trt10.13.ampereplus.aarch64.fp16.dynamic.engine",
}

# SMs whose SAME_COMPUTE_CAPABILITY plan fails on some GPUs of that SM, mapped to the GPUs it is validated on;
# other GPUs of that SM use the Ampere+ plan. Add a GPU once SQA passes on it.
SAME_SM_VALIDATED_GPUS = {"9.0": {"H100PCIe", "H10080GBHBM3"}}


def gpu_identifier(name: str) -> str:
    """Normalize a CUDA GPU ``name`` to a filename-safe identifier."""
    return re.sub(r"[^A-Za-z0-9_-]", "", name.removeprefix("NVIDIA "))


def select_trt_checkpoint(
    sm_version: Optional[str],
    machine: str,
    gpu_name: Optional[str] = None,
    trt_checkpoints: dict = TRT_CHECKPOINTS,
    trt_fallback_checkpoints: dict = TRT_FALLBACK_CHECKPOINTS,
) -> Optional[str]:
    """Return the Mask2Former engine filename to use, or ``None`` to fall back to PyTorch.

    Prefer a SAME_COMPUTE_CAPABILITY plan for the exact SM and CPU architecture,
    on SMs listed in ``SAME_SM_VALIDATED_GPUS`` only for the GPUs listed there.
    Otherwise use the CPU architecture's Ampere+ fallback for sm >= 8.0,
    or return ``None`` for the PyTorch path. SM103 does not match SM100.

    Args:
        sm_version: compute capability as ``"<major>.<minor>"`` (or ``None``).
        machine: ``platform.machine()`` (CPU arch, e.g. "x86_64" / "aarch64").
        gpu_name: CUDA device name (or ``None``).
        trt_checkpoints: Portable engine filenames keyed by ``(SM, CPU arch)``.
        trt_fallback_checkpoints: Ampere+ engine filenames keyed by CPU arch.

    Returns:
        The compatible engine filename, or ``None`` to use PyTorch.

    Raises:
        ValueError: Fallback selection cannot parse the SM version.
    """
    compatible = trt_checkpoints.get((sm_version, machine))
    gpu = gpu_identifier(gpu_name) if gpu_name else None
    if compatible is not None and gpu in SAME_SM_VALIDATED_GPUS.get(sm_version or "", {gpu}):
        return compatible
    if sm_version and tuple(int(x) for x in sm_version.split(".")) >= (8, 0):  # Ampere or newer
        return trt_fallback_checkpoints.get(machine)
    return None
