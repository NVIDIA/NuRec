# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest

import torch

from nurec.libs.gaussian_collective_payload_ops.cross_stream_test_utils import CrossStreamAllocationTestHarness
from nurec.libs.gaussian_collective_payload_ops.interface import gaussian_collective_payload_ops


def _malformed_metadata_worker(case: str) -> None:
    gradient = torch.ones(1, device="cuda")
    output = torch.empty(1, device="cuda")
    rows = [[0, 1, 0, 0, 1]]
    num_blocks = 1
    if case == "destination_range":
        rows[0][0] = 1
    elif case == "source_range":
        rows[0][3] = 1
    elif case == "block-count-deficit":
        values_per_block = gaussian_collective_payload_ops._GRADIENT_PACK_VALUES_PER_BLOCK
        gradient = torch.ones(values_per_block + 1, device="cuda")
        output = torch.empty_like(gradient)
        rows = [[0, gradient.numel(), 0, 0, 2]]
    elif case == "trailing-segment-deficit":
        values_per_block = gaussian_collective_payload_ops._GRADIENT_PACK_VALUES_PER_BLOCK
        gradient = torch.ones(values_per_block + 1, device="cuda")
        output = torch.empty_like(gradient)
        rows = [[0, values_per_block, 0, 0, 1], [values_per_block, 1, 0, values_per_block, 2]]
    elif case == "present-empty":
        gradient = torch.empty(0, device="cuda")
    elif case == "present-empty-view":
        gradient = torch.ones(1, device="cuda").narrow(0, 1, 0)
    else:
        raise ValueError(f"unknown malformed-metadata case {case!r}")
    copy_rows = torch.tensor(rows, device="cuda", dtype=torch.int64)
    gaussian_collective_payload_ops._pack_gradients_trusted(output, [gradient], copy_rows, num_blocks)
    torch.cuda.synchronize()
    raise AssertionError(f"malformed metadata case {case!r} did not trigger a device assertion")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GradientPackTest(unittest.TestCase):
    def test_packs_optional_gradients_and_padding_on_current_stream(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                first = torch.arange(8, device="cuda", dtype=dtype).reshape(4, 2)
                third = torch.arange(20, 26, device="cuda", dtype=dtype)
                copy_rows = torch.tensor(
                    [
                        [0, 4, 0, 0, 1],
                        [4, 3, 1, 0, 2],  # None input: dynamic zero segment.
                        [7, 2, 2, 1, 3],
                        [9, 3, -1, 0, 4],  # Explicit padding.
                        [12, 4, 0, 4, 5],
                        [16, 4, 2, 2, 6],
                    ],
                    device="cuda",
                    dtype=torch.int64,
                )
                # Inputs are produced on the default stream. Make that producer
                # work visible before validating a different current stream.
                torch.cuda.synchronize()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    output = torch.empty(20, device="cuda", dtype=dtype)
                    result = gaussian_collective_payload_ops._pack_gradients_trusted(
                        output, [first, None, third], copy_rows, 6
                    )
                    expected = torch.tensor(
                        [0, 1, 2, 3, 0, 0, 0, 21, 22, 0, 0, 0, 4, 5, 6, 7, 22, 23, 24, 25],
                        device="cuda",
                        dtype=dtype,
                    )
                    torch.testing.assert_close(result, expected)
                self.assertEqual(result.data_ptr(), output.data_ptr())
                self.assertEqual(output._version, 1)

    def test_large_segment_uses_multiple_blocks(self) -> None:
        values_per_block = gaussian_collective_payload_ops._GRADIENT_PACK_VALUES_PER_BLOCK
        gradient = torch.arange(2 * values_per_block + 8, device="cuda", dtype=torch.float32)
        output = torch.empty_like(gradient)
        copy_rows = torch.tensor([[0, gradient.numel(), 0, 0, 3]], device="cuda", dtype=torch.int64)

        gaussian_collective_payload_ops._pack_gradients_trusted(output, [gradient], copy_rows, 3)

        torch.testing.assert_close(output, gradient)

    def test_input_capacity_is_validated_by_native_pack(self) -> None:
        capacity = gaussian_collective_payload_ops._MAX_GRADIENT_PACK_INPUTS
        self.assertEqual(capacity, 128)
        gradients = [torch.empty(0, device="cuda") for _ in range(capacity + 1)]
        output = torch.empty(0, device="cuda")
        copy_rows = torch.empty((0, 5), device="cuda", dtype=torch.int64)

        with self.assertRaisesRegex(
            RuntimeError,
            rf"pack_gradients supports at most {capacity} input tensors, got {capacity + 1}",
        ):
            gaussian_collective_payload_ops._pack_gradients_trusted(output, gradients, copy_rows, 0)

    def test_malformed_trusted_metadata_fails_loudly(self) -> None:
        # Range violations, deficient launches, and referenced present-empty
        # gradients must all fail loudly rather than leave output unwritten.
        cases = (
            "destination_range",
            "source_range",
            "block-count-deficit",
            "trailing-segment-deficit",
            "present-empty",
            "present-empty-view",
        )
        for case in cases:
            with self.subTest(case=case):
                completed = subprocess.run(
                    [sys.executable, __file__, "--malformed-metadata-worker", case],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0, completed.stdout)
                self.assertRegex(
                    completed.stdout,
                    r"(?i)(device-side assert|malformed trusted gradient-pack plan)",
                )

    def test_accepts_unreferenced_present_empty_gradients(self) -> None:
        backing = torch.ones(1, device="cuda")
        empty_gradients = (
            torch.empty(0, device="cuda"),
            backing.narrow(0, 1, 0),
        )
        for empty_gradient in empty_gradients:
            with self.subTest(has_storage=empty_gradient.untyped_storage().nbytes() > 0):
                gradient = torch.tensor([3.0, 4.0], device="cuda")
                output = torch.empty_like(gradient)
                copy_rows = torch.tensor([[0, 2, 1, 0, 1]], device="cuda", dtype=torch.int64)

                result = gaussian_collective_payload_ops._pack_gradients_trusted(
                    output, [empty_gradient, gradient], copy_rows, 1
                )

                torch.testing.assert_close(result, gradient, rtol=0, atol=0)

    def test_rejects_noncontiguous_gradient(self) -> None:
        gradient = torch.ones((4, 2), device="cuda").t()
        output = torch.empty(gradient.numel(), device="cuda")
        copy_rows = torch.tensor([[0, gradient.numel(), 0, 0, 1]], device="cuda", dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "gradient 0 must be contiguous"):
            gaussian_collective_payload_ops._pack_gradients_trusted(output, [gradient], copy_rows, 1)

    def test_rejects_output_that_requires_gradients(self) -> None:
        gradient = torch.ones(4, device="cuda")
        output = torch.empty(4, device="cuda", requires_grad=True)
        copy_rows = torch.tensor([[0, 4, 0, 0, 1]], device="cuda", dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "output must not require gradients"):
            gaussian_collective_payload_ops._pack_gradients_trusted(output, [gradient], copy_rows, 1)

    def test_rejects_float8_output_before_dispatch(self) -> None:
        float8_dtype = getattr(torch, "float8_e4m3fn", None)
        if float8_dtype is None:
            self.skipTest("this torch build does not expose float8_e4m3fn")

        gradient = torch.empty(1, dtype=float8_dtype, device="cuda")
        output = torch.empty_like(gradient)
        copy_rows = torch.tensor([[0, 1, 0, 0, 1]], dtype=torch.int64, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "unsupported gradient output dtype"):
            gaussian_collective_payload_ops._pack_gradients_trusted(output, [gradient], copy_rows, 1)

    def test_rejects_inference_output_before_mutation(self) -> None:
        gradient = torch.arange(4, device="cuda", dtype=torch.float32)
        copy_rows = torch.tensor([[0, 4, 0, 0, 1]], device="cuda", dtype=torch.int64)
        with torch.inference_mode():
            output = torch.full((4,), -1.0, device="cuda")

        with self.assertRaisesRegex(RuntimeError, "output must not be an inference tensor"):
            gaussian_collective_payload_ops._pack_gradients_trusted(output, [gradient], copy_rows, 1)

        torch.testing.assert_close(output, torch.full_like(output, -1.0), rtol=0, atol=0)

    def test_rejects_output_aliasing_a_gradient(self) -> None:
        gradient = torch.arange(8, device="cuda", dtype=torch.float32)
        output = gradient.narrow(0, 2, 4)
        copy_rows = torch.tensor([[0, 4, 0, 0, 1]], device="cuda", dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, r"overlap|single memory location"):
            gaussian_collective_payload_ops._pack_gradients_trusted(output, [gradient], copy_rows, 1)

    def test_records_cross_stream_input_lifetime(self) -> None:
        values_per_block = gaussian_collective_payload_ops._GRADIENT_PACK_VALUES_PER_BLOCK
        numel = 4 * values_per_block
        lifetime = CrossStreamAllocationTestHarness.create()

        # Finish every allocation and host-to-device metadata copy before the
        # consumer waits. A pageable metadata copy issued after the wait may
        # block the host until the gate completes and destroy the test window.
        with torch.cuda.stream(lifetime.producer):
            gradient = torch.arange(numel, device="cuda", dtype=torch.float32)
            output = torch.empty(numel, device="cuda", dtype=torch.float32)
            copy_rows = torch.tensor([[0, numel, 0, 0, 4]], device="cuda", dtype=torch.int64)
            lifetime.producer_ready.record()
        source_pointer = gradient.data_ptr()

        consumer_done = lifetime.enqueue_gated(
            gaussian_collective_payload_ops._pack_gradients_trusted,
            output,
            [gradient],
            copy_rows,
            4,
        )

        del gradient
        # ``gradient`` is not part of a reference cycle, so CPython releases
        # it immediately at ``del``.  A forced cyclic-GC sweep is unrelated to
        # allocator lifetime and can consume the entire CUDA gate window on a
        # loaded test node before either assertion executes.
        # The source allocation remains pending rather than becoming reusable
        # while the consumer kernel is still waiting to read it.
        lifetime.assert_allocations_pending(source_pointer)

        consumer_done.synchronize()
        expected = torch.arange(numel, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--malformed-metadata-worker":
        _malformed_metadata_worker(sys.argv[2])
    else:
        unittest.main()
