# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest

import torch

from nurec.libs.gaussian_collective_payload_ops.cross_stream_test_utils import CrossStreamAllocationTestHarness
from nurec.libs.gaussian_collective_payload_ops.interface import gaussian_collective_payload_ops


def _field_major_source_and_expected(
    counts: tuple[int, ...],
    widths: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_count = max(counts)
    row_width = sum(widths)
    fields_by_rank: list[list[torch.Tensor]] = []
    rank_payloads: list[torch.Tensor] = []
    for rank, count in enumerate(counts):
        fields: list[torch.Tensor] = []
        for field, width in enumerate(widths):
            values = torch.arange(count * width, dtype=torch.float32, device="cuda").to(dtype)
            fields.append(values + rank * 1_000 + field * 100)
        fields_by_rank.append(fields)
        valid = torch.cat(fields)
        padding = torch.full(
            (max_count * row_width - valid.numel(),),
            float("nan"),
            dtype=dtype,
            device="cuda",
        )
        rank_payloads.append(torch.cat((valid, padding)))

    if max_count * row_width == 0:
        source = torch.empty((len(counts) * max_count, row_width), dtype=dtype, device="cuda")
    else:
        source = torch.cat(rank_payloads).reshape(len(counts) * max_count, row_width)
    expected = torch.cat([fields_by_rank[rank][field] for field in range(len(widths)) for rank in range(len(counts))])
    return source, expected


def _grouped_field_major_source_and_expected(
    counts_by_layer: tuple[tuple[int, ...], ...],
    widths_by_layer: tuple[tuple[int, ...], ...],
    dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
]:
    if not counts_by_layer or len(counts_by_layer) != len(widths_by_layer):
        raise ValueError("grouped test inputs must describe the same nonzero layer count")
    if len({len(counts) for counts in counts_by_layer}) != 1:
        raise ValueError("grouped test counts must describe one shared rank count")

    max_counts = tuple(max(counts) for counts in counts_by_layer)
    layer_spans = tuple(max_count * sum(widths) for max_count, widths in zip(max_counts, widths_by_layer, strict=True))
    layer_offsets = tuple(sum(layer_spans[:layer]) for layer in range(len(layer_spans)))
    rank_stride = sum(layer_spans)
    rank_count = len(counts_by_layer[0])
    grouped = torch.full(
        (rank_count, rank_stride),
        float("nan"),
        dtype=dtype,
        device="cuda",
    )

    expected: list[torch.Tensor] = []
    for counts, widths, offset, span in zip(
        counts_by_layer,
        widths_by_layer,
        layer_offsets,
        layer_spans,
        strict=True,
    ):
        compact, layer_expected = _field_major_source_and_expected(counts, widths, dtype)
        grouped[:, offset : offset + span].copy_(compact.reshape(rank_count, span))
        expected.append(layer_expected)

    return (
        grouped.reshape(-1),
        tuple(expected),
        max_counts,
        layer_offsets,
        layer_spans,
        rank_stride,
    )


def _grouped_payload_segments(
    counts_by_layer: tuple[tuple[int, ...], ...],
    widths_by_layer: tuple[tuple[int, ...], ...],
    layer_offsets: tuple[int, ...],
    rank_stride: int,
) -> tuple[torch.Tensor, int]:
    """Build trusted contiguous-copy rows for the scalable grouped test op."""
    num_blocks = 0
    rows: list[tuple[int, int, int, int, int]] = []
    for output_index, (counts, widths, layer_offset) in enumerate(
        zip(counts_by_layer, widths_by_layer, layer_offsets, strict=True)
    ):
        destination = 0
        field_offset = 0
        for width in widths:
            for rank, count in enumerate(counts):
                length = count * width
                if length:
                    source = rank * rank_stride + layer_offset + count * field_offset
                    num_blocks += (length + gaussian_collective_payload_ops._PAYLOAD_UNPACK_VALUES_PER_BLOCK - 1) // (
                        gaussian_collective_payload_ops._PAYLOAD_UNPACK_VALUES_PER_BLOCK
                    )
                    rows.append((output_index, destination, length, source, num_blocks))
                destination += length
            field_offset += width
    segments = (
        torch.tensor(rows, dtype=torch.int64, device="cuda")
        if rows
        else torch.empty((0, 5), dtype=torch.int64, device="cuda")
    )
    return segments, num_blocks


def _malformed_metadata_worker(case: str) -> None:
    source = torch.ones((1, 2), device="cuda")
    output = torch.empty(2, device="cuda")
    rows = [[0, 2, 0, 2, 1]]
    num_blocks = 1
    if case == "source-range-overflow":
        rows[0][2] = torch.iinfo(torch.int64).max
    elif case == "segment-length-overflow":
        rows[0][1] = torch.iinfo(torch.int64).max
    elif case == "block-count-deficit":
        source = torch.ones((4097, 1), device="cuda")
        output = torch.empty(4097, device="cuda")
        rows = [[0, 4097, 0, 1, 2]]
    elif case == "trailing-segment-deficit":
        source = torch.ones((4097, 1), device="cuda")
        output = torch.empty(4097, device="cuda")
        rows = [[0, 4096, 0, 1, 1], [4096, 1, 4096, 1, 2]]
    else:
        raise ValueError(f"unknown malformed-metadata case {case!r}")
    segments = torch.tensor(rows, dtype=torch.int64, device="cuda")
    gaussian_collective_payload_ops._unpack_payload_trusted(output, source, segments, num_blocks, source.size(1))
    torch.cuda.synchronize()
    raise AssertionError(f"malformed metadata case {case!r} did not trigger a device assertion")


def _grouped_malformed_metadata_worker(case: str) -> None:
    source = torch.arange(8, dtype=torch.float32, device="cuda")
    outputs: tuple[torch.Tensor, ...] = (source.new_empty(4), source.new_empty(4))
    rows = [[0, 0, 4, 0, 1], [1, 0, 4, 4, 2]]
    num_blocks = 2
    if case == "output-index":
        rows[0][0] = 2
    elif case == "output-index-eight":
        outputs = tuple(source.new_empty(1) for _ in range(8))
        rows = [[layer, 0, 1, layer, layer + 1] for layer in range(8)]
        rows[-1][0] = 8
        num_blocks = 8
    elif case == "source-range":
        rows[1][3] = torch.iinfo(torch.int64).max
    elif case == "destination-range":
        rows[1][1] = 4
    elif case == "source-offset-overflow":
        source = torch.arange(4097, dtype=torch.float32, device="cuda")
        outputs = (source.new_empty(4097),)
        rows = [[0, 0, 4097, torch.iinfo(torch.int64).max, 1]]
        num_blocks = 1
    elif case == "destination-offset-overflow":
        source = torch.arange(4097, dtype=torch.float32, device="cuda")
        outputs = (source.new_empty(1),)
        rows = [[0, torch.iinfo(torch.int64).max, 4097, 0, 1]]
        num_blocks = 1
    elif case == "block-end":
        rows[1][4] = 1
    elif case == "segment-length-overflow":
        rows[0][2] = torch.iinfo(torch.int64).max
    elif case == "block-count-deficit":
        source = torch.arange(4097, dtype=torch.float32, device="cuda")
        outputs = (source.new_empty(4097),)
        rows = [[0, 0, 4097, 0, 2]]
        num_blocks = 1
    elif case == "trailing-segment-deficit":
        source = torch.arange(4097, dtype=torch.float32, device="cuda")
        outputs = (source.new_empty(4097),)
        rows = [[0, 0, 4096, 0, 1], [0, 4096, 1, 4096, 2]]
        num_blocks = 1
    else:
        raise ValueError(f"unknown grouped malformed-metadata case {case!r}")
    segments = torch.tensor(rows, dtype=torch.int64, device="cuda")
    gaussian_collective_payload_ops._unpack_grouped_payload_trusted(outputs, source, segments, num_blocks)
    torch.cuda.synchronize()
    raise AssertionError(f"malformed grouped metadata case {case!r} did not trigger a device assertion")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PayloadUnpackTest(unittest.TestCase):
    def test_unpacks_uneven_and_empty_rank_fields_on_current_stream(self) -> None:
        # Three padded rank-major AoS payloads with counts (2, 0, 1) and
        # field widths (2, 1). Padding must not appear in the SoA output.
        rank_zero = torch.tensor([[0.0, 1.0, 10.0], [2.0, 3.0, 11.0]], device="cuda")
        rank_one = torch.full((2, 3), float("nan"), device="cuda")
        rank_two = torch.tensor([[4.0, 5.0, 12.0], [float("nan")] * 3], device="cuda")
        rank_major_aos = torch.cat((rank_zero, rank_one, rank_two))
        segments = torch.tensor(
            [
                [0, 4, 0, 2, 1],
                [4, 2, 12, 2, 2],
                [6, 2, 2, 1, 3],
                [8, 1, 14, 1, 4],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            output = rank_major_aos.new_empty(9)
            result = gaussian_collective_payload_ops._unpack_payload_trusted(
                output,
                rank_major_aos,
                segments,
                4,
                3,
            )
            self.assertEqual(result.data_ptr(), output.data_ptr())
            self.assertEqual(output._version, 1)
            torch.testing.assert_close(
                result,
                torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0], device="cuda"),
                rtol=0,
                atol=0,
            )

    def test_device_metadata_grouped_path_scales_through_72_ranks(self) -> None:
        widths_by_layer = (
            tuple(index % 4 + 1 for index in range(16)),
            (1, 0, 3),
            (2, 5, 1, 0),
            (4, 1),
        )
        for rank_count in (9, 16, 32, 64, 72):
            counts_by_layer = tuple(
                tuple((rank * (layer + 1) + layer) % 4 for rank in range(rank_count)) for layer in range(4)
            )
            if rank_count == 9:
                counts_by_layer = ((5001, *counts_by_layer[0][1:]), *counts_by_layer[1:])
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                with self.subTest(rank_count=rank_count, dtype=dtype):
                    source, expected, _max_counts, layer_offsets, _layer_spans, rank_stride = (
                        _grouped_field_major_source_and_expected(counts_by_layer, widths_by_layer, dtype)
                    )
                    segments, num_blocks = _grouped_payload_segments(
                        counts_by_layer,
                        widths_by_layer,
                        layer_offsets,
                        rank_stride,
                    )
                    outputs = tuple(source.new_empty(values.numel()) for values in expected)

                    result = gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                        outputs,
                        source,
                        segments,
                        num_blocks,
                    )

                    self.assertEqual(len(result), len(outputs))
                    for output, returned, values in zip(outputs, result, expected, strict=True):
                        self.assertEqual(returned.data_ptr(), output.data_ptr())
                        self.assertEqual(output._version, 1)
                        torch.testing.assert_close(output, values, rtol=0, atol=0)

    def test_device_metadata_grouped_path_reselects_outputs_across_relocation_generations(self) -> None:
        widths_by_layer = ((3, 1, 4), (2, 5), (1, 6, 2), (4, 3))
        rank_count = 28
        for generation in range(20):
            counts_by_layer = tuple(
                tuple((rank * (layer + 3) + generation * (layer + 1) + layer) % 129 for rank in range(rank_count))
                for layer in range(len(widths_by_layer))
            )
            source, expected, _max_counts, layer_offsets, _layer_spans, rank_stride = (
                _grouped_field_major_source_and_expected(counts_by_layer, widths_by_layer)
            )
            segments, num_blocks = _grouped_payload_segments(
                counts_by_layer,
                widths_by_layer,
                layer_offsets,
                rank_stride,
            )

            # Keep unrelated allocations around each destination so repeated
            # topology changes exercise fresh output addresses and every output
            # selector. Synchronize at the native boundary so an address-space
            # fault cannot be misattributed to the next CUDA operation.
            allocation_guards: list[torch.Tensor] = []
            outputs: list[torch.Tensor] = []
            for output_index, values in enumerate(expected):
                allocation_guards.append(source.new_empty((generation + 1) * (output_index + 1) * 257))
                outputs.append(source.new_empty(values.numel()))

            result = gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                tuple(outputs),
                source,
                segments,
                num_blocks,
            )
            torch.cuda.synchronize()

            self.assertEqual(len(result), len(outputs))
            for output, returned, values in zip(outputs, result, expected, strict=True):
                self.assertEqual(returned.data_ptr(), output.data_ptr())
                self.assertEqual(output._version, 1)
                torch.testing.assert_close(output, values, rtol=0, atol=0)
            self.assertEqual(len(allocation_guards), len(outputs))

    def test_grouped_path_handles_one_through_eight_uneven_empty_and_wide_layers(self) -> None:
        for num_layers in range(1, 9):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                with self.subTest(num_layers=num_layers, dtype=dtype):
                    # Include a wholly empty layer when it is not the final
                    # output and empty ranks in other layers. The final output
                    # crosses copy-block boundaries, exercising every selector.
                    counts_by_layer = tuple(
                        (0, 0, 0) if layer == 1 and num_layers > 2 else (layer + 1, 0, 2) for layer in range(num_layers)
                    )
                    widths_by_layer = tuple(
                        (2, 0, 4097 if layer == num_layers - 1 else 3) for layer in range(num_layers)
                    )
                    source, expected, _, layer_offsets, _, rank_stride = _grouped_field_major_source_and_expected(
                        counts_by_layer, widths_by_layer, dtype
                    )
                    segments, num_blocks = _grouped_payload_segments(
                        counts_by_layer, widths_by_layer, layer_offsets, rank_stride
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        outputs = tuple(torch.empty_like(values) for values in expected)
                        gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                            outputs, source, segments, num_blocks
                        )
                    stream.synchronize()
                    for output, values in zip(outputs, expected, strict=True):
                        self.assertEqual(output._version, 1)
                        torch.testing.assert_close(output, values, rtol=0, atol=0)

    def test_grouped_path_preserves_one_through_eight_empty_outputs(self) -> None:
        for num_layers in range(1, 9):
            with self.subTest(num_layers=num_layers):
                source = torch.empty(0, device="cuda")
                segments = torch.empty((0, 5), dtype=torch.int64, device="cuda")
                outputs = tuple(torch.empty_like(source) for _ in range(num_layers))
                gaussian_collective_payload_ops._unpack_grouped_payload_trusted(outputs, source, segments, 0)
                self.assertTrue(all(output._version == 1 for output in outputs))

    def test_device_metadata_grouped_path_preserves_empty_and_inference_semantics(self) -> None:
        counts_by_layer = ((0,) * 72, (0,) * 72, (0,) * 72, (0,) * 72)
        widths_by_layer = ((2, 1), (4,), (3, 0, 1), (5,))
        source, expected, _max_counts, layer_offsets, _layer_spans, rank_stride = (
            _grouped_field_major_source_and_expected(counts_by_layer, widths_by_layer)
        )
        segments, num_blocks = _grouped_payload_segments(
            counts_by_layer,
            widths_by_layer,
            layer_offsets,
            rank_stride,
        )
        outputs = tuple(source.new_empty(0) for _ in expected)

        result = gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
            outputs,
            source,
            segments,
            num_blocks,
        )

        self.assertEqual(num_blocks, 0)
        self.assertEqual(segments.shape, (0, 5))
        for output, returned in zip(outputs, result, strict=True):
            self.assertEqual(returned.data_ptr(), output.data_ptr())
            self.assertEqual(output._version, 1)

        with torch.inference_mode():
            inference_outputs = tuple(source.new_empty(0) for _ in expected)
            inference_result = gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                inference_outputs,
                source,
                segments,
                num_blocks,
            )
        self.assertTrue(all(output.is_inference() for output in inference_result))

    def test_device_metadata_grouped_path_rejects_bad_tensors_and_aliases(self) -> None:
        counts_by_layer = ((2,) * 9, (1,) * 9)
        widths_by_layer = ((2, 1), (3,))
        source, expected, _max_counts, layer_offsets, _layer_spans, rank_stride = (
            _grouped_field_major_source_and_expected(counts_by_layer, widths_by_layer)
        )
        segments, num_blocks = _grouped_payload_segments(
            counts_by_layer,
            widths_by_layer,
            layer_offsets,
            rank_stride,
        )

        def outputs() -> tuple[torch.Tensor, ...]:
            return tuple(source.new_empty(values.numel()) for values in expected)

        max_grouped_layers = gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS
        with self.assertRaisesRegex(RuntimeError, f"at most {max_grouped_layers} outputs"):
            gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                tuple(source.new_empty(0) for _ in range(max_grouped_layers + 1)),
                source,
                segments,
                num_blocks,
            )
        with self.assertRaisesRegex(RuntimeError, "shape \\[N, 5\\]"):
            gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                outputs(),
                source,
                segments[:, :4],
                num_blocks,
            )
        aliased = outputs()
        with self.assertRaisesRegex(RuntimeError, r"overlap|single memory location|distinct storage"):
            gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                (aliased[0], aliased[0]),
                source,
                segments,
                num_blocks,
            )
        with self.assertRaisesRegex(RuntimeError, r"overlap|single memory location|distinct storage"):
            gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                (source[: expected[0].numel()], outputs()[1]),
                source,
                segments,
                num_blocks,
            )

    def test_device_metadata_grouped_path_rejects_malformed_rows(self) -> None:
        for case in (
            "output-index",
            "output-index-eight",
            "source-range",
            "destination-range",
            "source-offset-overflow",
            "destination-offset-overflow",
            "block-end",
            "segment-length-overflow",
            "block-count-deficit",
            "trailing-segment-deficit",
        ):
            with self.subTest(case=case):
                completed = subprocess.run(
                    [sys.executable, __file__, "--grouped-malformed-metadata-worker", case],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0, completed.stdout)
                self.assertRegex(
                    completed.stdout,
                    r"(?i)(device-side assert|malformed trusted grouped payload metadata)",
                )

    def test_device_metadata_grouped_path_supports_nonempty_inference_outputs(self) -> None:
        counts_by_layer = ((2,) * 9, (0,) * 9, (1,) * 9)
        widths_by_layer = ((2, 1), (4,), (3, 0, 1))
        source, expected, _max_counts, layer_offsets, _layer_spans, rank_stride = (
            _grouped_field_major_source_and_expected(counts_by_layer, widths_by_layer)
        )
        segments, num_blocks = _grouped_payload_segments(
            counts_by_layer,
            widths_by_layer,
            layer_offsets,
            rank_stride,
        )

        normal_outputs = tuple(source.new_empty(values.numel()) for values in expected)
        normal_result = gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
            normal_outputs,
            source,
            segments,
            num_blocks,
        )
        for output, returned, values in zip(normal_outputs, normal_result, expected, strict=True):
            self.assertEqual(output._version, 1)
            self.assertEqual(returned.data_ptr(), output.data_ptr())
            torch.testing.assert_close(output, values, rtol=0, atol=0)

        with torch.inference_mode():
            outputs = tuple(source.new_empty(values.numel()) for values in expected)
            result = gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                outputs,
                source,
                segments,
                num_blocks,
            )
        for output, returned, values in zip(outputs, result, expected, strict=True):
            self.assertTrue(output.is_inference())
            self.assertEqual(returned.data_ptr(), output.data_ptr())
            torch.testing.assert_close(output, values, rtol=0, atol=0)

        with torch.inference_mode():
            stale_outputs = tuple(source.new_empty(values.numel()) for values in expected)
        with self.assertRaisesRegex(RuntimeError, "only valid while inference mode is enabled"):
            gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                stale_outputs,
                source,
                segments,
                num_blocks,
            )

    def test_device_metadata_grouped_path_records_cross_stream_allocation_lifetimes(self) -> None:
        counts_by_layer = (
            (4096, *(0 for _ in range(71))),
            (2049, *(0 for _ in range(71))),
            (1025, *(0 for _ in range(71))),
        )
        widths_by_layer = ((4,), (3,), (2,))
        lifetime = CrossStreamAllocationTestHarness.create()

        with torch.cuda.stream(lifetime.producer):
            source, expected, _max_counts, layer_offsets, _layer_spans, rank_stride = (
                _grouped_field_major_source_and_expected(counts_by_layer, widths_by_layer)
            )
            segments, num_blocks = _grouped_payload_segments(
                counts_by_layer,
                widths_by_layer,
                layer_offsets,
                rank_stride,
            )
            outputs = tuple(source.new_empty(values.numel()) for values in expected)
            lifetime.producer_ready.record()
        source_pointer = source.data_ptr()
        segment_pointer = segments.data_ptr()
        released_output_pointer = outputs[0].data_ptr()

        consumer_done = lifetime.enqueue_gated(
            gaussian_collective_payload_ops._unpack_grouped_payload_trusted,
            outputs,
            source,
            segments,
            num_blocks,
        )

        retained_outputs = outputs[1:]
        del source, segments, outputs
        lifetime.assert_allocations_pending(source_pointer, segment_pointer, released_output_pointer)

        consumer_done.synchronize()
        for output, values in zip(retained_outputs, expected[1:], strict=True):
            torch.testing.assert_close(output, values, rtol=0, atol=0)

    def test_empty_unpack_launches_no_work(self) -> None:
        source = torch.empty((0, 3), device="cuda")
        segments = torch.empty((0, 5), dtype=torch.int64, device="cuda")
        output = source.new_empty(0)

        result = gaussian_collective_payload_ops._unpack_payload_trusted(output, source, segments, 0, 3)

        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertEqual(output._version, 1)
        self.assertEqual(result.numel(), 0)

    def test_grouped_path_records_eight_output_cross_stream_allocation_lifetimes(self) -> None:
        counts_by_layer = tuple((4097 - layer, 0, layer) for layer in range(8))
        widths_by_layer = ((2,),) * 8
        lifetime = CrossStreamAllocationTestHarness.create()
        with torch.cuda.stream(lifetime.producer):
            source, expected, _, layer_offsets, _, rank_stride = _grouped_field_major_source_and_expected(
                counts_by_layer, widths_by_layer
            )
            segments, num_blocks = _grouped_payload_segments(
                counts_by_layer, widths_by_layer, layer_offsets, rank_stride
            )
            outputs = tuple(torch.empty_like(values) for values in expected)
            lifetime.producer_ready.record()
        pointers = (source.data_ptr(), segments.data_ptr(), outputs[-1].data_ptr())
        consumer_done = lifetime.enqueue_gated(
            gaussian_collective_payload_ops._unpack_grouped_payload_trusted, outputs, source, segments, num_blocks
        )
        retained_outputs = outputs[:-1]
        del source, segments, outputs
        lifetime.assert_allocations_pending(*pointers)
        consumer_done.synchronize()
        for output, values in zip(retained_outputs, expected[:-1], strict=True):
            torch.testing.assert_close(output, values, rtol=0, atol=0)

    def test_supports_inference_mode(self) -> None:
        with torch.inference_mode():
            source = torch.arange(6, dtype=torch.float32, device="cuda").reshape(3, 2)
            segments = torch.tensor([[0, 6, 0, 2, 1]], dtype=torch.int64, device="cuda")
            output = source.new_empty(6)
            result = gaussian_collective_payload_ops._unpack_payload_trusted(output, source, segments, 1, 2)

        self.assertTrue(result.is_inference())
        self.assertEqual(result.data_ptr(), output.data_ptr())
        torch.testing.assert_close(result, source.reshape(-1), rtol=0, atol=0)

    def test_large_uneven_fields_cross_multiple_blocks(self) -> None:
        counts = (5001, 4097)
        max_count = max(counts)
        widths = (3, 45)
        row_width = sum(widths)
        first = torch.arange(counts[0] * row_width, dtype=torch.float32, device="cuda").reshape(counts[0], row_width)
        second = (
            torch.arange(counts[1] * row_width, dtype=torch.float32, device="cuda").reshape(counts[1], row_width)
            + 1_000_000
        )
        second_padded = torch.full((max_count, row_width), float("nan"), device="cuda")
        second_padded[: counts[1]].copy_(second)
        source = torch.cat((first, second_padded))

        destination = 0
        cumulative_blocks = 0
        field_offset = 0
        segment_rows: list[list[int]] = []
        for width in widths:
            for rank, count in enumerate(counts):
                length = count * width
                cumulative_blocks += (
                    length + gaussian_collective_payload_ops._PAYLOAD_UNPACK_VALUES_PER_BLOCK - 1
                ) // gaussian_collective_payload_ops._PAYLOAD_UNPACK_VALUES_PER_BLOCK
                segment_rows.append(
                    [
                        destination,
                        length,
                        rank * max_count * row_width + field_offset,
                        width,
                        cumulative_blocks,
                    ]
                )
                destination += length
            field_offset += width
        segments = torch.tensor(segment_rows, dtype=torch.int64, device="cuda")
        output = source.new_empty(destination)

        result = gaussian_collective_payload_ops._unpack_payload_trusted(
            output,
            source,
            segments,
            cumulative_blocks,
            row_width,
        )

        expected = torch.cat(
            (
                first[:, : widths[0]].reshape(-1),
                second[:, : widths[0]].reshape(-1),
                first[:, widths[0] :].reshape(-1),
                second[:, widths[0] :].reshape(-1),
            )
        )
        self.assertEqual(result.data_ptr(), output.data_ptr())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_accepts_source_range_ending_at_input_boundary(self) -> None:
        source = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
        segments = torch.tensor([[0, 6, 2, 2, 1]], dtype=torch.int64, device="cuda")
        output = source.new_empty(6)

        result = gaussian_collective_payload_ops._unpack_payload_trusted(output, source, segments, 1, 4)

        torch.testing.assert_close(
            result,
            torch.tensor([2, 3, 6, 7, 10, 11], dtype=torch.float32, device="cuda"),
            rtol=0,
            atol=0,
        )

    def test_rejects_malformed_metadata(self) -> None:
        for case in (
            "source-range-overflow",
            "segment-length-overflow",
            "block-count-deficit",
            "trailing-segment-deficit",
        ):
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
                    r"(?i)(device-side assert|malformed trusted payload-unpack metadata)",
                )

    def test_rejects_noncontiguous_or_autograd_input(self) -> None:
        segments = torch.tensor([[0, 6, 0, 2, 1]], dtype=torch.int64, device="cuda")
        output = torch.empty(6, device="cuda")

        with self.assertRaisesRegex(RuntimeError, "must be contiguous"):
            gaussian_collective_payload_ops._unpack_payload_trusted(
                output,
                torch.ones((3, 4), device="cuda")[:, ::2],
                segments,
                1,
                2,
            )

        with self.assertRaisesRegex(RuntimeError, "output must not require gradients"):
            gaussian_collective_payload_ops._unpack_payload_trusted(
                torch.empty(6, device="cuda", requires_grad=True),
                torch.ones((3, 2), device="cuda"),
                segments,
                1,
                2,
            )

        with self.assertRaisesRegex(RuntimeError, "must not require gradients"):
            gaussian_collective_payload_ops._unpack_payload_trusted(
                output,
                torch.ones((3, 2), device="cuda", requires_grad=True),
                segments,
                1,
                2,
            )

    def test_rejects_float8_before_dispatch(self) -> None:
        float8_dtype = getattr(torch, "float8_e4m3fn", None)
        if float8_dtype is None:
            self.skipTest("this torch build does not expose float8_e4m3fn")

        source = torch.empty((0, 1), dtype=float8_dtype, device="cuda")
        segments = torch.empty((0, 5), dtype=torch.int64, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "unsupported payload output dtype"):
            gaussian_collective_payload_ops._unpack_payload_trusted(
                source.new_empty(0),
                source,
                segments,
                0,
                1,
            )

        with self.assertRaisesRegex(RuntimeError, "unsupported rank-major grouped payload dtype"):
            gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
                (source.new_empty(0),),
                source.reshape(-1),
                segments,
                0,
            )

    def test_rejects_alias_and_inference_output_outside_inference_mode(self) -> None:
        source = torch.arange(6, dtype=torch.float32, device="cuda").reshape(3, 2)
        segments = torch.tensor([[0, 6, 0, 2, 1]], dtype=torch.int64, device="cuda")

        with self.assertRaisesRegex(RuntimeError, r"overlap|single memory location"):
            gaussian_collective_payload_ops._unpack_payload_trusted(
                source.reshape(-1),
                source,
                segments,
                1,
                2,
            )

        with torch.inference_mode():
            inference_output = torch.full((6,), -1.0, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "only valid while inference mode is enabled"):
            gaussian_collective_payload_ops._unpack_payload_trusted(
                inference_output,
                source,
                segments,
                1,
                2,
            )
        torch.testing.assert_close(inference_output, torch.full_like(inference_output, -1.0), rtol=0, atol=0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--malformed-metadata-worker":
        _malformed_metadata_worker(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--grouped-malformed-metadata-worker":
        _grouped_malformed_metadata_worker(sys.argv[2])
    else:
        unittest.main()
