# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast unit tests for Mask2Former TensorRT engine selection.

Pure logic — no GPU, no engine/model download, no built image — so it runs in the
pre-merge ``build+test`` job. The full TRT-vs-torch accuracy comparison
(``trt_torch_comparison_test.py``) needs a GPU + inference and stays a manual,
on-demand (SQA) test.
"""

import pytest

from apps.av.aux_gen.trt_engine_selection import (
    TRT_CHECKPOINTS,
    TRT_FALLBACK_CHECKPOINTS,
    select_trt_checkpoint,
)


@pytest.mark.parametrize(
    "gpu_name, sm_version, machine, expected",
    [
        # Native engine per GPU model (including the newly added 5880 + L4).
        ("NVIDIA RTX 5880 Ada Generation", "8.9", "x86_64", TRT_CHECKPOINTS[("5880", "x86_64")]),
        ("NVIDIA L4", "8.9", "x86_64", TRT_CHECKPOINTS[("L4", "x86_64")]),
        ("NVIDIA L40", "8.9", "x86_64", TRT_CHECKPOINTS[("L40", "x86_64")]),
        ("NVIDIA L40", "8.9", "aarch64", TRT_CHECKPOINTS[("L40", "aarch64")]),
        ("NVIDIA A100-SXM4-80GB", "8.0", "x86_64", TRT_CHECKPOINTS[("A100", "x86_64")]),
        ("NVIDIA A40", "8.6", "x86_64", TRT_CHECKPOINTS[("A40", "x86_64")]),
        ("NVIDIA RTX A6000", "8.6", "x86_64", TRT_CHECKPOINTS[("A6000", "x86_64")]),
        # Whole-word match: "L40S" must NOT match the "L40" native -> Ampere+ fallback.
        ("NVIDIA L40S", "8.9", "x86_64", TRT_FALLBACK_CHECKPOINTS["x86_64"]),
        # Ampere-or-newer GPU with no native engine -> generic Ampere+ fallback.
        ("NVIDIA GeForce RTX 4090", "8.9", "x86_64", TRT_FALLBACK_CHECKPOINTS["x86_64"]),
        ("NVIDIA H100 80GB HBM3", "9.0", "x86_64", TRT_FALLBACK_CHECKPOINTS["x86_64"]),
        # A100 native is x86_64-only, so the same GPU on aarch64 uses the aarch64 fallback.
        ("NVIDIA A100-SXM4-80GB", "8.0", "aarch64", TRT_FALLBACK_CHECKPOINTS["aarch64"]),
        # Pre-Ampere (sm < 8.0) cannot load the engines -> PyTorch (None).
        ("Tesla V100-SXM2-16GB", "7.0", "x86_64", None),
        # No GPU -> PyTorch (None).
        (None, None, "x86_64", None),
    ],
)
def test_select_trt_checkpoint(gpu_name, sm_version, machine, expected):
    assert select_trt_checkpoint(gpu_name, sm_version, machine) == expected


def test_checkpoint_names_are_wellformed_and_unique():
    """Every native + fallback engine name is a distinct ``.engine`` filename."""
    names = list(TRT_CHECKPOINTS.values()) + list(TRT_FALLBACK_CHECKPOINTS.values())
    assert all(n.endswith(".engine") for n in names)
    assert len(names) == len(set(names))
