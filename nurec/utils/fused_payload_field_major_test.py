# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from unittest.mock import patch

import torch

import nurec.models.gaussians.fused_parameter_gather as fused_gather


def _rank_fields(
    rank: int,
    count: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    offset = rank * 64
    values = torch.arange(offset, offset + count * 3, dtype=torch.float32, device="cuda")
    return {
        "values": (values.reshape(count, 3) * 0.25).to(dtype),
        "empty": torch.empty((count, 0), dtype=dtype, device="cuda"),
        "ids": torch.arange(offset, offset + count, dtype=torch.int64, device="cuda"),
    }


def _mixed_rank_fields(
    rank: int,
    count: int,
    id_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    base = torch.arange(rank * 32, rank * 32 + count * 2, dtype=torch.float32, device="cuda")
    return {
        "fp16": (base.reshape(count, 2) * 0.5).to(torch.float16),
        "bf16": (base.reshape(count, 2) * 0.25).to(torch.bfloat16),
        "fp32": (base.reshape(count, 2) * 0.125).to(torch.float32),
        "fp64": (base.reshape(count, 2) * 0.0625).to(torch.float64),
        "ids": torch.arange(rank * 32, rank * 32 + count, dtype=id_dtype, device="cuda"),
        "empty": torch.empty((count, 0), dtype=torch.float32, device="cuda"),
    }


def _torch_cat_padding_oracle(
    fields_by_rank: tuple[dict[str, torch.Tensor], ...],
    max_count: int,
) -> dict[str, torch.Tensor]:
    keys = tuple(fields_by_rank[0])
    local_payloads: list[torch.Tensor] = []
    widths = tuple(math.prod(tensor.shape[1:]) for tensor in fields_by_rank[0].values())
    for fields in fields_by_rank:
        count = next(iter(fields.values())).shape[0]
        payload = torch.cat(
            [fields[key].reshape(count, width) for key, width in zip(keys, widths, strict=True)],
            dim=-1,
        )
        if count < max_count:
            payload = torch.cat(
                [
                    payload,
                    payload.new_full((max_count - count, payload.shape[1]), float("NaN")),
                ],
                dim=0,
            )
        local_payloads.append(payload)

    gathered = torch.cat(local_payloads, dim=0)
    outputs: dict[str, torch.Tensor] = {}
    field_offset = 0
    for key, width in zip(keys, widths, strict=True):
        valid = [
            gathered.narrow(0, rank * max_count, fields[key].shape[0]).narrow(1, field_offset, width).reshape(-1)
            for rank, fields in enumerate(fields_by_rank)
        ]
        destination_dtype = fields_by_rank[0][key].dtype
        gathered_count = sum(fields[key].shape[0] for fields in fields_by_rank)
        outputs[key] = (
            torch.cat(valid)
            .reshape(
                gathered_count,
                *fields_by_rank[0][key].shape[1:],
            )
            .to(destination_dtype)
        )
        field_offset += width
    return outputs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FusedPayloadFieldMajorTest(unittest.TestCase):
    def tearDown(self) -> None:
        fused_gather._cached_payload_unpack_layout.cache_clear()

    def _assert_round_trip(
        self,
        fields_by_rank: tuple[dict[str, torch.Tensor], ...],
    ) -> None:
        counts = tuple(next(iter(fields.values())).shape[0] for fields in fields_by_rank)
        max_count = max(counts)
        layout = fused_gather._make_layer_layout("test", fields_by_rank[0])
        packed = tuple(
            fused_gather._pack_layer_payload(layout, tuple(fields.values()), max_count) for fields in fields_by_rank
        )

        for fields, payload, count in zip(fields_by_rank, packed, counts, strict=True):
            field_major = torch.cat([tensor.reshape(-1).to(layout.payload_dtype) for tensor in fields.values()])
            padding = payload.numel() - field_major.numel()
            expected_payload = torch.cat(
                [
                    field_major,
                    torch.zeros(padding, dtype=layout.payload_dtype, device="cuda"),
                ]
            )
            torch.testing.assert_close(payload.reshape(-1), expected_payload, rtol=0, atol=0)
            self.assertEqual(payload.shape[0], max_count)
            self.assertEqual(field_major.numel(), count * sum(layout.widths))

        gathered = torch.cat(packed, dim=0)
        actual = fused_gather._unpack_layer_payload(layout, gathered, counts, max_count)
        expected_fields = _torch_cat_padding_oracle(fields_by_rank, max_count)
        self.assertEqual(tuple(actual), tuple(expected_fields))
        for key in expected_fields:
            self.assertEqual(actual[key].dtype, expected_fields[key].dtype)
            self.assertEqual(actual[key].shape, expected_fields[key].shape)
            self.assertTrue(actual[key].is_contiguous())
            torch.testing.assert_close(actual[key], expected_fields[key], rtol=0, atol=0)
            if actual[key].dtype.is_floating_point:
                self.assertFalse(torch.isnan(actual[key]).any())

    def test_matches_torch_cat_padding_oracle_across_dtypes_and_counts(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for counts in ((3, 1), (2, 2), (0, 3), (0, 0)):
                with self.subTest(dtype=dtype, counts=counts):
                    self._assert_round_trip(
                        tuple(_rank_fields(rank, count, dtype) for rank, count in enumerate(counts))
                    )

    def test_mixed_float_and_integer_fields_match_oracle(self) -> None:
        for id_dtype in (torch.int32, torch.int64):
            with self.subTest(id_dtype=id_dtype):
                self._assert_round_trip(
                    (
                        _mixed_rank_fields(0, 3, id_dtype),
                        _mixed_rank_fields(1, 1, id_dtype),
                    )
                )

    def test_production_unpack_uses_metadata_path_at_old_16_field_8_rank_boundary(self) -> None:
        counts = (3, 0, 2, 1, 3, 2, 0, 1)
        widths = tuple(index % 4 for index in range(16))
        fields_by_rank = tuple(
            {
                f"field_{field}": (
                    torch.arange(count * width, dtype=torch.float32, device="cuda").reshape(count, width)
                    + rank * 100_000
                    + field * 1_000
                )
                for field, width in enumerate(widths)
            }
            for rank, count in enumerate(counts)
        )
        real_generic_unpack = fused_gather.gaussian_collective_payload_ops._unpack_payload_trusted

        with patch.object(
            fused_gather.gaussian_collective_payload_ops,
            "_unpack_payload_trusted",
            wraps=real_generic_unpack,
        ) as generic_unpack:
            self._assert_round_trip(fields_by_rank)

        generic_unpack.assert_called_once()
        self.assertEqual(fused_gather._cached_payload_unpack_layout.cache_info().misses, 1)

    def test_production_unpack_keeps_the_same_metadata_path_for_larger_layouts(self) -> None:
        topologies = (
            ((2, 1), (1,) * 17),
            ((2, 1, 0, 2, 1, 0, 2, 1, 1), (2,)),
        )
        for counts, widths in topologies:
            with self.subTest(ranks=len(counts), fields=len(widths)):
                fused_gather._cached_payload_unpack_layout.cache_clear()
                fields_by_rank = tuple(
                    {
                        f"field_{field}": (
                            torch.arange(count * width, dtype=torch.float32, device="cuda").reshape(count, width)
                            + rank * 100_000
                            + field * 1_000
                        )
                        for field, width in enumerate(widths)
                    }
                    for rank, count in enumerate(counts)
                )
                real_generic_unpack = fused_gather.gaussian_collective_payload_ops._unpack_payload_trusted

                with patch.object(
                    fused_gather.gaussian_collective_payload_ops,
                    "_unpack_payload_trusted",
                    wraps=real_generic_unpack,
                ) as generic_unpack:
                    self._assert_round_trip(fields_by_rank)

                generic_unpack.assert_called_once()
                self.assertEqual(fused_gather._cached_payload_unpack_layout.cache_info().misses, 1)

    def test_payload_binding_preserves_validation_and_inference_contract(self) -> None:
        source = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)
        with torch.inference_mode():
            output = torch.empty(12, dtype=torch.float32, device="cuda")
            result = fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
                output,
                [source],
                [2],
                4,
                6,
            )
        self.assertTrue(result.is_inference())
        torch.testing.assert_close(
            result.reshape(-1),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0], dtype=torch.float32, device="cuda"),
            rtol=0,
            atol=0,
        )

        normal_output = torch.empty(8, dtype=torch.float32, device="cuda")
        fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
            normal_output,
            [source],
            [2],
            4,
            4,
        )
        self.assertEqual(normal_output._version, 1)

        with self.assertRaisesRegex(RuntimeError, "payload input 0 must be contiguous"):
            fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
                torch.empty(8, dtype=torch.float32, device="cuda"),
                [torch.ones((2, 4), device="cuda").t()],
                [2],
                4,
                4,
            )

        mixed_output = torch.empty(4, dtype=torch.float32, device="cuda")
        fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
            mixed_output,
            [
                torch.tensor([[1.5], [2.5]], dtype=torch.float16, device="cuda"),
                torch.tensor([[3], [4]], dtype=torch.int32, device="cuda"),
            ],
            [1, 1],
            2,
            2,
        )
        torch.testing.assert_close(
            mixed_output.reshape(-1),
            torch.tensor([1.5, 2.5, 3.0, 4.0], dtype=torch.float32, device="cuda"),
            rtol=0,
            atol=0,
        )

        storage = torch.arange(12, dtype=torch.float32, device="cuda")
        with self.assertRaisesRegex(RuntimeError, r"overlap|single memory location"):
            fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
                storage[:8],
                [storage[4:].reshape(4, 2)],
                [2],
                4,
                4,
            )

        with torch.inference_mode():
            stale_inference_output = torch.full((8,), -1.0, dtype=torch.float32, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "only valid while inference mode is enabled"):
            fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
                stale_inference_output,
                [source],
                [2],
                4,
                4,
            )
        torch.testing.assert_close(stale_inference_output, torch.full_like(stale_inference_output, -1.0))

    def test_payload_binding_uses_fast_and_fallback_descriptor_capacities(self) -> None:
        for field_count in (16, 17, 128):
            with self.subTest(field_count=field_count):
                fields = [torch.full((1, 1), index, dtype=torch.int32, device="cuda") for index in range(field_count)]
                output = torch.empty(field_count, dtype=torch.float32, device="cuda")
                fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
                    output,
                    fields,
                    [1] * field_count,
                    1,
                    1,
                )
                torch.testing.assert_close(
                    output.reshape(-1),
                    torch.arange(field_count, dtype=torch.float32, device="cuda"),
                    rtol=0,
                    atol=0,
                )

        over_capacity = 129
        with self.assertRaisesRegex(RuntimeError, "supports at most"):
            fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
                torch.empty(over_capacity, dtype=torch.float32, device="cuda"),
                [torch.empty((1, 1), dtype=torch.float32, device="cuda") for _ in range(over_capacity)],
                [1] * over_capacity,
                1,
                1,
            )

    def test_payload_binding_casts_every_supported_input_dtype(self) -> None:
        fields = [
            torch.tensor([[False], [True]], dtype=torch.bool, device="cuda"),
            torch.tensor([[1], [2]], dtype=torch.uint8, device="cuda"),
            torch.tensor([[-3], [4]], dtype=torch.int8, device="cuda"),
            torch.tensor([[-5], [6]], dtype=torch.int16, device="cuda"),
            torch.tensor([[-7], [8]], dtype=torch.int32, device="cuda"),
            torch.tensor([[-9], [10]], dtype=torch.int64, device="cuda"),
            torch.tensor([[-1.5], [2.5]], dtype=torch.float16, device="cuda"),
            torch.tensor([[-3.5], [4.5]], dtype=torch.bfloat16, device="cuda"),
            torch.tensor([[-5.5], [6.5]], dtype=torch.float32, device="cuda"),
            torch.tensor([[-7.5], [8.5]], dtype=torch.float64, device="cuda"),
        ]
        output = torch.empty(2 * len(fields), dtype=torch.float64, device="cuda")
        fused_gather.gaussian_collective_payload_ops._pack_payload_trusted(
            output,
            fields,
            [1] * len(fields),
            2,
            2,
        )
        torch.testing.assert_close(
            output.reshape(-1),
            torch.cat([field.reshape(-1).to(torch.float64) for field in fields]),
            rtol=0,
            atol=0,
        )


class FusedPayloadFieldMajorMetadataTest(unittest.TestCase):
    def test_payload_input_dtype_rejects_unsupported_complex_fields(self) -> None:
        layout = fused_gather._LayerPayloadLayout(
            node_id="complex",
            keys=("complex",),
            trailing_shapes=((1,),),
            dtypes=(torch.complex64,),
            widths=(1,),
            payload_dtype=torch.float32,
        )
        with self.assertRaisesRegex(RuntimeError, "payload packing supports input dtypes"):
            fused_gather._validate_fused_schema((layout,), (False,))

    def test_payload_input_capacity_is_per_layer_not_aggregate(self) -> None:
        layouts = tuple(
            fused_gather._make_layer_layout(
                node_id,
                {f"field_{index}": torch.ones(1, 1, dtype=torch.float32) for index in range(65)},
            )
            for node_id in ("first", "second")
        )

        # The two layers have 130 aggregate fields, but each pack launch uses
        # only its own 65 inputs and remains below the 128-input CUDA ABI cap.
        fused_gather._validate_fused_schema(layouts, (False,) * 130)

    def test_schema_validation_leaves_input_capacity_to_native_pack(self) -> None:
        layout = fused_gather._make_layer_layout(
            "oversized",
            {f"field_{index}": torch.ones(1, 1, dtype=torch.float32) for index in range(129)},
        )

        fused_gather._validate_fused_schema(
            (layout,),
            (False,) * len(layout.keys),
        )

    def test_unpack_rows_cover_zero_width_and_empty_rank(self) -> None:
        unpack_rows, unpack_numel, unpack_blocks = fused_gather._payload_unpack_segment_rows(
            counts=(2, 0, 1),
            widths=(2, 0, 1),
            max_count=2,
        )
        self.assertEqual(
            unpack_rows,
            (
                (0, 4, 0, 1, 1),
                (4, 2, 12, 1, 2),
                (6, 2, 4, 1, 3),
                (8, 1, 14, 1, 4),
            ),
        )
        self.assertEqual(unpack_numel, 9)
        self.assertEqual(unpack_blocks, 4)


if __name__ == "__main__":
    unittest.main()
