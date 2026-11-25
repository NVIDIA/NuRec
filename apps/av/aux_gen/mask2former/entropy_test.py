# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact entropy/threshold parity and captured-input regression checks."""

import pytest
import torch

from apps.av.aux_gen.mask2former.entropy import _quantile, categorical_entropy, mask2former_postprocess


def reference(x):
    """Independent pinned Torch entropy oracle, with no native dispatch."""
    return torch.special.entr(x.softmax(0)).sum(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("pixels", [128, 4096])
def test_native_entropy(pixels):
    """Pin stream ordering, replay after input changes, and special-value propagation."""
    torch.manual_seed(5)
    x = torch.randn(20, 1, pixels, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = categorical_entropy(x)
        expected = reference(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = categorical_entropy(x)
        x.mul_(20)
        graph.replay()
        changed = reference(x)
    stream.synchronize()
    # Exact parity targets Torch 2.11.0+cu128.nre1. On upgrades, recheck entropy.cu:
    # categorical_entropy_kernel's spatial-softmax ordering,
    # four-accumulator sum and pixels >= 128 and pixels % 4 == 0 vectorized-reduction gate.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(captured, changed, rtol=0, atol=0)
    threshold = torch.quantile(captured, 0.9)
    torch.testing.assert_close(threshold, torch.quantile(changed, 0.9), rtol=0, atol=0)
    assert torch.equal(captured > threshold, changed > threshold)
    x[:, :, 0] = -torch.inf
    x[0, :, 1] = torch.nan
    torch.testing.assert_close(categorical_entropy(x), reference(x), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("shape", [(7, 2, 3), (20, 3, 101)])
def test_generic_entropy_and_grad(shape):
    """CPU float64 must bypass native dispatch without changing values or derivatives."""
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    actual = categorical_entropy(x)
    expected = reference(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "pixels,dtype,update,scale",
    [
        (128, torch.int64, False, 1),
        (128, torch.int32, True, None),
        (4096, torch.uint8, True, 1),
        (2160 * 3840, torch.int64, True, 20),
        (4096**2 + 4, torch.int32, True, 1),
    ],
)
def test_native_postprocess(pixels, dtype, update, scale):
    """Cover label dtypes and exact FP32 selection across Torch's input-size limit."""
    torch.manual_seed(31)
    if scale is None:
        # Quantile neighbors in distinct histogram bins: zero versus log(20).
        x = torch.zeros(20, 1, pixels, device="cuda")
        x[0, :, :115] = 1000
    else:
        x = (torch.randn(20, 1, pixels, device="cuda") * scale).bfloat16().float()
    predictions = torch.randint(20, (1, 1, pixels), device="cuda", dtype=dtype)
    expected_logits = x.clone()
    entropy = reference(x)
    if pixels <= 4096**2:
        threshold = torch.quantile(entropy, 0.9)
    else:
        values = entropy.flatten().sort().values
        # Extend Torch's FP32 rank/lerp rule beyond its public API size limit.
        rank = torch.tensor(0.9, device=x.device) * (pixels - 1)
        threshold = torch.lerp(values[rank.floor().long()], values[rank.ceil().long()], rank - rank.floor())
    expected_mask = entropy > threshold
    expected_labels = predictions.to(torch.uint8).squeeze(0).clone()
    expected_labels[expected_mask] = 3
    if update:
        expected_logits[:, expected_mask] = 0
        expected_logits[3, expected_mask] = 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        labels, logits, mask = mask2former_postprocess(x, predictions, 3, update)
    stream.synchronize()
    # In entropy.cu, also recheck histogram_quantile_state's FP32 rank calculation
    # and quantile_value's lerp on Torch upgrades before changing these exact assertions. Above 2^24,
    # the native rule extends Torch's arithmetic beyond its public API size limit.
    for actual, expected in ((labels, expected_labels), (logits, expected_logits), (mask, expected_mask)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert logits.data_ptr() == x.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_postprocess_graph_and_nan():
    """Replay must clear prior NaN state and leave tied entropy values unmasked."""
    x = torch.zeros(20, 1, 128, device="cuda")
    predictions = torch.zeros(1, 1, 128, device="cuda", dtype=torch.int64)
    # Uniform entropy ties must stay unmasked; a NaN quantile masks nothing.
    mask2former_postprocess(x, predictions, 3, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        labels, _, mask = mask2former_postprocess(x, predictions, 3, True)
    for value in (None, torch.nan, 0.0):
        if value is None:
            x.zero_()
        else:
            x.normal_()
            x[0, 0, 0] = value
        entropy = reference(x)
        expected_mask = entropy > torch.quantile(entropy, 0.9)
        graph.replay()
        assert torch.equal(mask, expected_mask)
        assert torch.equal(labels, expected_mask.to(torch.uint8) * 3)


@pytest.mark.parametrize("shape,dim,keepdim", [((2, 3), 1, True), ((200, 300), None, False)])
def test_quantile(shape, dim, keepdim):
    """Preserve dimension/keepdim and flattened quantiles after moving the helper."""
    a = torch.randn(shape)
    torch.testing.assert_close(
        _quantile(a, 0.90, dim=dim, keepdim=keepdim), torch.quantile(a, 0.90, dim=dim, keepdim=keepdim)
    )


def test_quantile_large():
    """The legacy interpolation works beyond torch.quantile's input-size limit."""
    # Revisit the workaround if a Torch upgrade lifts this limit.
    a = torch.randn(4096**2 + 1)
    with pytest.raises(RuntimeError):
        torch.quantile(a, 0.90)
    _quantile(a, 0.90)
