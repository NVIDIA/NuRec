# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from nurec.libs.gaussian_collective_payload_ops.cross_stream_test_utils import CrossStreamAllocationTestHarness
from nurec.libs.gaussian_collective_payload_ops.interface import gaussian_collective_payload_ops


def _mixed_layers() -> tuple[
    list[list[torch.Tensor]],
    list[list[int]],
    list[int],
    list[int],
]:
    fields_by_layer = [
        [
            torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device="cuda"),
            torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float16, device="cuda"),
            torch.empty((3, 0), dtype=torch.float64, device="cuda"),
        ],
        [
            torch.empty((0, 3), dtype=torch.bfloat16, device="cuda"),
            torch.empty((0, 1), dtype=torch.bool, device="cuda"),
        ],
        [
            torch.tensor([[7], [8]], dtype=torch.int64, device="cuda"),
            torch.tensor([[9.25, 10.5], [11.75, 12.0]], dtype=torch.float64, device="cuda"),
        ],
    ]
    return fields_by_layer, [[2, 1, 0], [3, 1], [1, 2]], [3, 0, 2], [5, 4, 2]


def _reference_single_layer_packs(
    dtype: torch.dtype,
    fields_by_layer: list[list[torch.Tensor]],
    widths_by_layer: list[list[int]],
    local_counts: list[int],
    max_counts: list[int],
) -> torch.Tensor:
    total = sum(max_count * sum(widths) for max_count, widths in zip(max_counts, widths_by_layer, strict=True))
    output = torch.empty(total, dtype=dtype, device="cuda")
    offset = 0
    for fields, widths, local_count, max_count in zip(
        fields_by_layer,
        widths_by_layer,
        local_counts,
        max_counts,
        strict=True,
    ):
        span = max_count * sum(widths)
        view = output.narrow(0, offset, span)
        gaussian_collective_payload_ops._pack_payload_trusted(view, fields, widths, local_count, max_count)
        offset += span
    return output


def _independent_grouped_oracle(
    dtype: torch.dtype,
    fields_by_layer: list[list[torch.Tensor]],
    widths_by_layer: list[list[int]],
    local_counts: list[int],
    max_counts: list[int],
) -> torch.Tensor:
    r"""_independent_grouped_oracle(dtype, fields_by_layer, widths_by_layer, local_counts, max_counts) -> Tensor

    Build the field-major grouped wire payload with :func:`torch.cat`.

    Each layer contributes its fields flattened in row-major order and cast
    to ``dtype``, followed by one zero-filled padding region. Padding is after
    all fields in a layer, not after each field. No native pack operation is
    called, and no input tensor is modified.

    Args:
        dtype (torch.dtype): Output dtype: float16, bfloat16, float32, or float64.
        fields_by_layer (list[list[Tensor]]): One or more nonempty field lists.
          Each field is contiguous, has shape ``(local_counts[layer], width)``,
          and has a supported real boolean, integral, or floating-point dtype.
          All fields are on the same CUDA device; mixed dtypes are supported.
        widths_by_layer (list[list[int]]): Nonnegative widths, with one entry
          per field and one list per layer. Zero-width fields are supported.
        local_counts (list[int]): Nonnegative row counts, one per layer;
          zero-count layers are supported.
        max_counts (list[int]): Row capacities, one per layer, each at least
          its local count. The four layer-indexed arguments have equal lengths.

    Returns:
        Tensor: Newly allocated contiguous one-dimensional tensor on the input
          device with ``dtype`` and ``sum(max_count * sum(widths))`` elements.

    Examples::

        >>> field = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
        >>> result = _independent_grouped_oracle(torch.float32, [[field]], [[2]], [1], [2])
        >>> result.tolist()
        [1.0, 2.0, 0.0, 0.0]
    """
    chunks: list[torch.Tensor] = []
    for fields, widths, local_count, max_count in zip(
        fields_by_layer, widths_by_layer, local_counts, max_counts, strict=True
    ):
        chunks.extend(field.to(dtype).reshape(-1) for field in fields)
        chunks.append(torch.zeros((max_count - local_count) * sum(widths), dtype=dtype, device=fields[0].device))
    return torch.cat(chunks)


class PayloadBindingSurfaceTest(unittest.TestCase):
    def test_binding_exports_exact_trusted_surface(self) -> None:
        expected = {
            "_GRADIENT_PACK_VALUES_PER_BLOCK",
            "_MAX_GRADIENT_PACK_INPUTS",
            "_MAX_GROUPED_PAYLOAD_LAYERS",
            "_PAYLOAD_UNPACK_VALUES_PER_BLOCK",
            "_pack_gradients_trusted",
            "_pack_grouped_payload_trusted",
            "_pack_payload_trusted",
            "_unpack_grouped_payload_trusted",
            "_unpack_payload_trusted",
        }
        actual = {name for name in dir(gaussian_collective_payload_ops) if not name.startswith("__")}
        self.assertEqual(actual, expected)
        for name in (
            "_GRADIENT_PACK_VALUES_PER_BLOCK",
            "_MAX_GRADIENT_PACK_INPUTS",
            "_MAX_GROUPED_PAYLOAD_LAYERS",
            "_PAYLOAD_UNPACK_VALUES_PER_BLOCK",
        ):
            value = getattr(gaussian_collective_payload_ops, name)
            self.assertIsInstance(value, int)
            self.assertGreater(value, 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PayloadPackTest(unittest.TestCase):
    def test_single_and_grouped_pack_match_independent_oracle_for_all_dtypes(self) -> None:
        fields = [
            torch.tensor([[False, True], [True, False]], dtype=torch.bool, device="cuda"),
            torch.tensor([[1], [255]], dtype=torch.uint8, device="cuda"),
            torch.tensor([[-2], [3]], dtype=torch.int8, device="cuda"),
            torch.tensor([[-1000, 1000], [7, -8]], dtype=torch.int16, device="cuda"),
            torch.tensor([[-9], [10]], dtype=torch.int32, device="cuda"),
            torch.tensor([[11], [-12]], dtype=torch.int64, device="cuda"),
            torch.tensor([[-1.5], [2.25]], dtype=torch.float16, device="cuda"),
            torch.tensor([[0.5], [-3.0]], dtype=torch.bfloat16, device="cuda"),
            torch.tensor([[1.25], [-2.5]], dtype=torch.float32, device="cuda"),
            torch.tensor([[3.5], [-4.0]], dtype=torch.float64, device="cuda"),
        ]
        widths = [field.size(1) for field in fields]
        local_count = 2
        max_count = 3
        row_width = sum(widths)
        padding = max_count * row_width - sum(field.numel() for field in fields)
        self.assertGreater(padding, 0)

        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                expected = torch.cat(
                    [field.to(dtype).reshape(-1) for field in fields]
                    + [torch.zeros(padding, dtype=dtype, device="cuda")]
                )
                single = torch.empty_like(expected)
                grouped = torch.empty_like(expected)

                gaussian_collective_payload_ops._pack_payload_trusted(
                    single,
                    fields,
                    widths,
                    local_count,
                    max_count,
                )
                gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                    grouped,
                    [fields],
                    [widths],
                    [local_count],
                    [max_count],
                )

                self.assertTrue(torch.equal(single.view(torch.uint8), expected.view(torch.uint8)))
                self.assertTrue(torch.equal(grouped.view(torch.uint8), expected.view(torch.uint8)))

    def test_single_pack_validates_flat_geometry_and_supports_zero_sizes(self) -> None:
        source = torch.ones((2, 1), dtype=torch.float32, device="cuda")
        call = gaussian_collective_payload_ops._pack_payload_trusted

        with self.assertRaisesRegex(RuntimeError, "one-dimensional"):
            call(torch.empty((2, 1), device="cuda"), [source], [1], 2, 2)
        with self.assertRaisesRegex(RuntimeError, "2 values, expected 3"):
            call(torch.empty(2, device="cuda"), [source], [1], 2, 3)
        with self.assertRaisesRegex(RuntimeError, "between zero and maximum count"):
            call(torch.empty(2, device="cuda"), [source], [1], 2, 1)

        zero_width_output = torch.empty(0, dtype=torch.float32, device="cuda")
        zero_width_field = torch.empty((3, 0), dtype=torch.int16, device="cuda")
        result = call(zero_width_output, [zero_width_field], [0], 3, 5)
        self.assertEqual(result.data_ptr(), zero_width_output.data_ptr())
        self.assertEqual(zero_width_output._version, 1)

        zero_count_output = torch.empty(6, dtype=torch.float32, device="cuda")
        zero_count_field = torch.empty((0, 2), dtype=torch.float16, device="cuda")
        call(zero_count_output, [zero_count_field], [2], 0, 3)
        torch.testing.assert_close(zero_count_output, torch.zeros_like(zero_count_output), rtol=0, atol=0)

        zero_max_output = torch.empty(0, dtype=torch.float32, device="cuda")
        zero_max_field = torch.empty((0, 2), dtype=torch.float16, device="cuda")
        call(zero_max_output, [zero_max_field], [2], 0, 0)
        self.assertEqual(zero_max_output._version, 1)

    def test_grouped_pack_is_byte_exact_to_three_single_layer_calls_for_all_output_dtypes(self) -> None:
        fields_by_layer, widths_by_layer, local_counts, max_counts = _mixed_layers()
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                expected = _reference_single_layer_packs(
                    dtype,
                    fields_by_layer,
                    widths_by_layer,
                    local_counts,
                    max_counts,
                )
                actual = torch.empty_like(expected)
                result = gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                    actual,
                    fields_by_layer,
                    widths_by_layer,
                    local_counts,
                    max_counts,
                )

                self.assertEqual(result.data_ptr(), actual.data_ptr())
                self.assertEqual(actual._version, 1)
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

    def test_grouped_pack_is_byte_exact_at_production_scale(self) -> None:
        count = 1_000_000
        width = 8
        source = torch.arange(count * width, dtype=torch.int32, device="cuda").reshape(count, width)
        output = torch.empty(count * width, dtype=torch.float32, device="cuda")

        result = gaussian_collective_payload_ops._pack_grouped_payload_trusted(
            output,
            [[source]],
            [[width]],
            [count],
            [count],
        )

        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertEqual(output._version, 1)
        self.assertTrue(torch.equal(output, source.reshape(-1).to(output.dtype)))

    def test_grouped_pack_matches_oracle_for_five_and_eight_layers(self) -> None:
        for num_layers in (5, 8):
            local_counts = [layer % 4 for layer in range(num_layers)]
            max_counts = [count + 2 for count in local_counts]
            widths_by_layer = [[2, 0, 4097 if layer == 3 else 3] for layer in range(num_layers)]
            fields_by_layer = [
                [
                    (torch.arange(count * width, device="cuda") % 97 + layer).to(dtype).reshape(count, width)
                    for width, dtype in zip(widths, (torch.int32, torch.float64, torch.float16), strict=True)
                ]
                for layer, (count, widths) in enumerate(zip(local_counts, widths_by_layer, strict=True))
            ]
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                with self.subTest(num_layers=num_layers, dtype=dtype):
                    expected = _independent_grouped_oracle(
                        dtype, fields_by_layer, widths_by_layer, local_counts, max_counts
                    )
                    output = torch.empty_like(expected)
                    gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                        output, fields_by_layer, widths_by_layer, local_counts, max_counts
                    )
                    self.assertEqual(output._version, 1)
                    self.assertTrue(torch.equal(output.view(torch.uint8), expected.view(torch.uint8)))

    def test_grouped_pack_runs_on_nondefault_stream_and_handles_zero_padding(self) -> None:
        fields_by_layer, widths_by_layer, local_counts, _ = _mixed_layers()
        max_counts = list(local_counts)
        expected = _reference_single_layer_packs(
            torch.float32,
            fields_by_layer,
            widths_by_layer,
            local_counts,
            max_counts,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = torch.empty_like(expected)
            gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                actual,
                fields_by_layer,
                widths_by_layer,
                local_counts,
                max_counts,
            )
            done = torch.cuda.Event()
            done.record()
        done.synchronize()

        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

    def test_grouped_pack_supports_64_and_128_fields_with_padding(self) -> None:
        for num_layers in (4, 8):
            with self.subTest(num_layers=num_layers, num_fields=num_layers * 16, num_segments=num_layers * 17):
                fields_by_layer = [
                    [torch.tensor([[layer * 100 + field]], dtype=torch.int32, device="cuda") for field in range(16)]
                    for layer in range(num_layers)
                ]
                widths_by_layer = [[1] * 16 for _ in range(num_layers)]
                local_counts = [1] * num_layers
                max_counts = [2] * num_layers
                expected = _independent_grouped_oracle(
                    torch.float32,
                    fields_by_layer,
                    widths_by_layer,
                    local_counts,
                    max_counts,
                )
                actual = torch.empty_like(expected)

                gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                    actual,
                    fields_by_layer,
                    widths_by_layer,
                    local_counts,
                    max_counts,
                )

                self.assertEqual(actual._version, 1)
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

    def test_grouped_pack_supports_70_fields_across_four_layers(self) -> None:
        field_counts = (18, 18, 17, 17)
        local_counts = [2, 1, 3, 1]
        max_counts = [3, 2, 4, 1]
        widths_by_layer = [[1] * field_count for field_count in field_counts]
        widths_by_layer[-1][-1] = 4097
        input_dtypes = (torch.int32, torch.float64, torch.float16)
        fields_by_layer = [
            [
                (torch.arange(count * width, device="cuda") % 97 + layer + field)
                .to(input_dtypes[field % len(input_dtypes)])
                .reshape(count, width)
                for field, width in enumerate(widths)
            ]
            for layer, (count, widths) in enumerate(zip(local_counts, widths_by_layer, strict=True))
        ]
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                expected = _independent_grouped_oracle(
                    dtype, fields_by_layer, widths_by_layer, local_counts, max_counts
                )
                output = torch.empty_like(expected)
                gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                    output, fields_by_layer, widths_by_layer, local_counts, max_counts
                )
                self.assertEqual(output._version, 1)
                self.assertTrue(torch.equal(output.view(torch.uint8), expected.view(torch.uint8)))

    def test_grouped_pack_matches_oracle_at_128_and_129_aggregate_fields(self) -> None:
        for field_counts in ([127, 1], [128, 1]):
            with self.subTest(num_fields=sum(field_counts)):
                fields_by_layer = [
                    [torch.tensor([[layer * 1000 + field]], device="cuda") for field in range(field_count)]
                    for layer, field_count in enumerate(field_counts)
                ]
                widths_by_layer = [[1] * field_count for field_count in field_counts]
                expected = _independent_grouped_oracle(torch.float32, fields_by_layer, widths_by_layer, [1, 1], [2, 2])
                output = torch.empty_like(expected)
                result = gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                    output, fields_by_layer, widths_by_layer, [1, 1], [2, 2]
                )
                self.assertEqual(result.data_ptr(), output.data_ptr())
                self.assertEqual(output._version, 1)
                self.assertTrue(torch.equal(output.view(torch.uint8), expected.view(torch.uint8)))

    def test_grouped_pack_falls_back_to_per_layer_native_pack_above_descriptor_capacity(self) -> None:
        # Each case exceeds 136 nonempty field/padding segments, so removing
        # the fallback gate must fail the grouped descriptor capacity check.
        for field_counts in ([128, 9], [65, 72], [27] * 5, [17] * 8):
            with self.subTest(field_counts=field_counts):
                fields_by_layer = [
                    [
                        torch.tensor(
                            [[layer * 1000 + field], [layer * 1000 + field + 100]],
                            dtype=torch.int32,
                            device="cuda",
                        )
                        for field in range(field_count)
                    ]
                    for layer, field_count in enumerate(field_counts)
                ]
                widths_by_layer = [[1] * field_count for field_count in field_counts]
                local_counts = [2] * len(field_counts)
                max_counts = [3] * len(field_counts)
                expected = _reference_single_layer_packs(
                    torch.float32,
                    fields_by_layer,
                    widths_by_layer,
                    local_counts,
                    max_counts,
                )
                actual = torch.empty_like(expected)

                result = gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                    actual,
                    fields_by_layer,
                    widths_by_layer,
                    local_counts,
                    max_counts,
                )

                self.assertEqual(result.data_ptr(), actual.data_ptr())
                self.assertEqual(actual._version, 1)
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

    def test_grouped_pack_fallback_validates_every_layer_before_launch(self) -> None:
        first_layer = [torch.ones((1, 1), dtype=torch.float32, device="cuda") for _ in range(128)]
        # Keep 137 valid fields before the malformed field so a missing
        # fallback gate hits descriptor capacity before the expected shape error.
        malformed_second_layer = [torch.ones((1, 1), dtype=torch.float32, device="cuda") for _ in range(9)]
        malformed_second_layer.append(torch.ones((2, 1), dtype=torch.float32, device="cuda"))
        output = torch.full((138,), -1.0, dtype=torch.float32, device="cuda")
        expected = output.clone()

        with self.assertRaisesRegex(RuntimeError, r"expected \[1, 1\]"):
            gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                output,
                [first_layer, malformed_second_layer],
                [[1] * 128, [1] * 10],
                [1, 1],
                [1, 1],
            )

        self.assertEqual(output._version, 0)
        self.assertTrue(torch.equal(output, expected))

    def test_grouped_pack_records_cross_stream_input_lifetime(self) -> None:
        count = 16_384
        lifetime = CrossStreamAllocationTestHarness.create()

        with torch.cuda.stream(lifetime.producer):
            source = torch.arange(count, dtype=torch.float32, device="cuda").reshape(count, 1)
            output = torch.empty(count, dtype=torch.float32, device="cuda")
            lifetime.producer_ready.record()
        source_pointer = source.data_ptr()

        consumer_done = lifetime.enqueue_gated(
            gaussian_collective_payload_ops._pack_grouped_payload_trusted,
            output,
            [[source]],
            [[1]],
            [count],
            [count],
        )

        del source
        lifetime.assert_allocations_pending(source_pointer)

        consumer_done.synchronize()
        torch.testing.assert_close(
            output,
            torch.arange(count, dtype=torch.float32, device="cuda"),
            rtol=0,
            atol=0,
        )

    def test_grouped_pack_empty_payload_launches_no_work(self) -> None:
        fields_by_layer = [
            [torch.empty((0, 0), dtype=torch.float16, device="cuda")],
            [torch.empty((0, 0), dtype=torch.int64, device="cuda")],
            [torch.empty((0, 0), dtype=torch.bfloat16, device="cuda")],
        ]
        output = torch.empty(0, dtype=torch.float32, device="cuda")
        result = gaussian_collective_payload_ops._pack_grouped_payload_trusted(
            output,
            fields_by_layer,
            [[0], [0], [0]],
            [0, 0, 0],
            [0, 0, 0],
        )

        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertEqual(output._version, 1)
        self.assertEqual(output.numel(), 0)

    def test_grouped_pack_records_all_eight_layer_cross_stream_input_lifetimes(self) -> None:
        count = 4097
        lifetime = CrossStreamAllocationTestHarness.create()
        with torch.cuda.stream(lifetime.producer):
            sources = [torch.full((count, 1), layer, device="cuda", dtype=torch.float32) for layer in range(8)]
            output = torch.empty(count * 8, device="cuda")
            lifetime.producer_ready.record()
        pointers = [source.data_ptr() for source in sources]
        consumer_done = lifetime.enqueue_gated(
            gaussian_collective_payload_ops._pack_grouped_payload_trusted,
            output,
            [[source] for source in sources],
            [[1]] * 8,
            [count] * 8,
            [count] * 8,
        )
        del sources
        lifetime.assert_allocations_pending(*pointers)
        consumer_done.synchronize()
        expected = torch.arange(8, device="cuda", dtype=torch.float32).repeat_interleave(count)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_grouped_pack_accepts_five_and_eight_empty_layers(self) -> None:
        for num_layers in (5, 8):
            with self.subTest(num_layers=num_layers):
                output = torch.empty(0, device="cuda")
                fields = [[torch.empty((0, 2), device="cuda")] for _ in range(num_layers)]
                gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                    output, fields, [[2]] * num_layers, [0] * num_layers, [0] * num_layers
                )
                self.assertEqual(output._version, 1)
                self.assertEqual(output.numel(), 0)

    def test_grouped_pack_validates_topology_geometry_dtypes_and_aliasing(self) -> None:
        source = torch.ones((2, 1), dtype=torch.float32, device="cuda")
        output = torch.empty(2, dtype=torch.float32, device="cuda")
        call = gaussian_collective_payload_ops._pack_grouped_payload_trusted
        max_grouped_layers = gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS

        invalid_calls: tuple[tuple[tuple[object, ...], str], ...] = (
            ((output, [], [], [], []), "at least one layer"),
            (
                (
                    output,
                    [[source]] * (max_grouped_layers + 1),
                    [[1]] * (max_grouped_layers + 1),
                    [2] * (max_grouped_layers + 1),
                    [2] * (max_grouped_layers + 1),
                ),
                f"at most {max_grouped_layers} layers",
            ),
            ((output, [[source]], [], [2], [2]), "width layer count"),
            ((output, [[source]], [[1]], [], [2]), "local-count length"),
            ((output, [[source]], [[1]], [2], []), "max-count length"),
            ((output, [[]], [[]], [0], [0]), "at least one input"),
            ((output, [[source]], [[]], [2], [2]), "does not match width count"),
            ((output, [[source]], [[-1]], [2], [2]), "negative width"),
            ((output, [[source]], [[1]], [-1], [2]), "must be between zero"),
            ((output, [[source]], [[1]], [2], [1]), "must be between zero"),
            ((output, [[source]], [[2]], [2], [2]), r"expected \[2, 2\]"),
            ((torch.empty(3, device="cuda"), [[source]], [[1]], [2], [2]), "values, expected 2"),
            ((torch.empty((1, 2), device="cuda"), [[source]], [[1]], [2], [2]), "one-dimensional"),
            ((torch.empty(2, dtype=torch.int32, device="cuda"), [[source]], [[1]], [2], [2]), "floating-point"),
            (
                (
                    torch.empty(4, device="cuda"),
                    [[torch.ones((2, 2), device="cuda").t()]],
                    [[2]],
                    [2],
                    [2],
                ),
                "must be contiguous",
            ),
            ((output, [[torch.ones((2, 1), dtype=torch.complex64, device="cuda")]], [[1]], [2], [2]), "unsupported"),
        )
        for arguments, message in invalid_calls:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                call(*arguments)

        too_many_fields = 129
        with self.assertRaisesRegex(RuntimeError, "at most 128 inputs"):
            call(
                torch.empty(too_many_fields, device="cuda"),
                [[torch.ones((1, 1), device="cuda") for _ in range(too_many_fields)]],
                [[1] * too_many_fields],
                [1],
                [1],
            )

        storage = torch.arange(4, dtype=torch.float32, device="cuda")
        with self.assertRaisesRegex(RuntimeError, r"overlap|single memory location"):
            call(storage[:2], [[storage[:2].reshape(2, 1)]], [[1]], [2], [2])

        with self.assertRaisesRegex(RuntimeError, "must be a CUDA tensor"):
            call(torch.empty(2), [[torch.ones((2, 1), device="cuda")]], [[1]], [2], [2])
        with self.assertRaisesRegex(RuntimeError, "must be a CUDA tensor"):
            call(output, [[torch.ones((2, 1))]], [[1]], [2], [2])
        with self.assertRaisesRegex(RuntimeError, "output must not require gradients"):
            call(torch.empty(2, device="cuda", requires_grad=True), [[source]], [[1]], [2], [2])

    def test_grouped_pack_binding_rejects_malformed_nested_lists(self) -> None:
        output = torch.empty(1, device="cuda")
        call = gaussian_collective_payload_ops._pack_grouped_payload_trusted
        invalid_calls = (
            ((output, [torch.ones(1, device="cuda")], [[1]], [1], [1]), "sequence of tensor sequences"),
            ((output, [[object()]], [[1]], [1], [1]), "only torch.Tensor"),
            (
                (output, [[torch.ones((1, 1), device="cuda")]], [object()], [1], [1]),
                "sequence of integer sequences",
            ),
            ((output, [[torch.ones((1, 1), device="cuda")]], [[object()]], [1], [1]), "only integer"),
            ((output, [[torch.ones((1, 1), device="cuda")]], [[1]], [object()], [1]), "local counts"),
            ((output, [[torch.ones((1, 1), device="cuda")]], [[1]], [1], [object()]), "maximum counts"),
        )
        for arguments, message in invalid_calls:
            with self.subTest(message=message), self.assertRaisesRegex(TypeError, message):
                call(*arguments)

    def test_grouped_pack_binding_accepts_nested_tuples(self) -> None:
        source = torch.tensor([[1.0], [2.0]], device="cuda")
        output = torch.empty(2, device="cuda")

        result = gaussian_collective_payload_ops._pack_grouped_payload_trusted(
            output,
            ((source,),),
            ((1,),),
            (2,),
            (2,),
        )

        torch.testing.assert_close(result, source.reshape(-1), rtol=0, atol=0)

    def test_grouped_pack_supports_inference_mode_and_rejects_stale_output(self) -> None:
        source = torch.tensor([[1.0], [2.0]], device="cuda")
        with torch.inference_mode():
            output = torch.empty(2, device="cuda")
            result = gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                output,
                [[source]],
                [[1]],
                [2],
                [2],
            )
        self.assertTrue(result.is_inference())
        torch.testing.assert_close(result, source.reshape(-1), rtol=0, atol=0)

        with torch.inference_mode():
            stale = torch.full((2,), -1.0, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "only valid while inference mode is enabled"):
            gaussian_collective_payload_ops._pack_grouped_payload_trusted(stale, [[source]], [[1]], [2], [2])
        torch.testing.assert_close(stale, torch.full_like(stale, -1.0), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
