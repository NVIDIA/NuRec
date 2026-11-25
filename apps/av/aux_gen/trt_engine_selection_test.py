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
    "sm_version, machine, expected",
    [
        ("8.0", "x86_64", TRT_CHECKPOINTS[("8.0", "x86_64")]),
        ("8.6", "x86_64", TRT_CHECKPOINTS[("8.6", "x86_64")]),
        ("8.9", "x86_64", TRT_CHECKPOINTS[("8.9", "x86_64")]),
        ("9.0", "x86_64", TRT_FALLBACK_CHECKPOINTS["x86_64"]),
        ("10.3", "aarch64", TRT_CHECKPOINTS[("10.3", "aarch64")]),
        # Full SM and CPU architecture must match; otherwise use Ampere+.
        ("10.0", "aarch64", TRT_FALLBACK_CHECKPOINTS["aarch64"]),
        ("10.3", "x86_64", TRT_FALLBACK_CHECKPOINTS["x86_64"]),
        ("8.9", "aarch64", TRT_FALLBACK_CHECKPOINTS["aarch64"]),
        ("9.0", "aarch64", TRT_FALLBACK_CHECKPOINTS["aarch64"]),
        ("8.0", "aarch64", TRT_FALLBACK_CHECKPOINTS["aarch64"]),
        ("10.0", "x86_64", TRT_FALLBACK_CHECKPOINTS["x86_64"]),
        # Pre-Ampere, missing capability/GPU, and unsupported CPUs use PyTorch.
        ("7.0", "x86_64", None),
        (None, "x86_64", None),
        ("8.6", "riscv64", None),
    ],
)
def test_select_trt_checkpoint(sm_version, machine, expected):
    assert select_trt_checkpoint(sm_version, machine) == expected


@pytest.mark.parametrize(
    "gpu, same_sm",
    [("NVIDIA H100 PCIe", True), ("NVIDIA H20", False), ("NVIDIA H100 80GB HBM3", True), (None, False)],
)
def test_sm90_same_sm_plan_only_on_validated_gpu(gpu, same_sm):
    expected = TRT_CHECKPOINTS[("9.0", "x86_64")] if same_sm else TRT_FALLBACK_CHECKPOINTS["x86_64"]
    assert select_trt_checkpoint("9.0", "x86_64", gpu) == expected


def test_checkpoint_names_are_wellformed_and_unique():
    """Every portable and fallback engine name is a distinct ``.engine`` filename."""
    names = [*TRT_CHECKPOINTS.values(), *TRT_FALLBACK_CHECKPOINTS.values()]
    assert all(n.endswith(".engine") for n in names)
    assert len(names) == len(set(names))
