# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused multi-layer Gaussian parameter communication.

This module is the rank-symmetric custom-autograd communication boundary used
by ``GaussiansComposite``. The private entry point at the bottom converts the
model's per-node parameter dictionaries into an ordered field list and passes
that list through one custom autograd function.

The protocol has three phases:

1. **Rank agreement.** Every rank exchanges one fixed-width control packet and
   agrees on layer count, communication schema, and per-layer row counts before
   any variable-sized payload collective. The communication-schema hash covers
   node and field order, trailing shapes, dtypes, promoted payload dtypes, and
   ``requires_grad`` flags. It detects rank divergence; it is not a data checksum.
2. **Forward AllGather.** Compatible layers are packed field-major into one
   padded rank-local buffer, gathered once, then unpacked by CUDA into contiguous
   per-field tensors. Valid layouts that cannot share a grouped payload retain
   the safe per-layer AllGather path, but all payload collectives are enqueued
   before unpack kernels.
3. **Backward ReduceScatter.** Gradients of the gathered fields are packed by
   destination rank into one padded buffer. When the global payload is nonempty,
   one ReduceScatter returns this rank's local gradients, which are split back
   into the original field shapes.

Only tensor first-dimension sizes may vary across ranks. Layer/field order and the
rest of the schema must be rank-symmetric. Unsupported schemas fail before
payload communication, integer fields must be exactly representable in the
promoted floating payload dtype, and higher-order gradients are intentionally
unsupported. Locally empty shards still participate when another rank has rows;
globally empty forward or backward payloads skip their respective data
collective.

When the owning strategy guarantees a stable topology, a cached generation
plan reuses rank-agreed metadata and may prelaunch the grouped payload. That
path removes steady-state metadata collectives without changing the collective
order or the forward/backward wire layouts described above.
"""

from __future__ import annotations

import functools
import hashlib
import operator
import struct
import weakref

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, cast

import torch

from nurec.libs.gaussian_collective_payload_ops.interface import gaussian_collective_payload_ops
from nurec.utils.profiling import ScopedTimer


@dataclass(frozen=True, slots=True)
class _LayerPayloadLayout:
    """Metadata needed to unpack one layer's fields from its row payload."""

    node_id: str
    keys: tuple[str, ...]
    trailing_shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    widths: tuple[int, ...]
    payload_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _GradientPackPlan:
    """Device copy plan for one destination-rank-major gradient buffer."""

    copy_rows: torch.Tensor
    output_numel: int
    num_blocks: int


@dataclass(frozen=True, slots=True)
class _PayloadUnpackLayout:
    """Cached device metadata for unpacking one gathered layer."""

    segments: torch.Tensor
    output_numel: int
    num_blocks: int


@dataclass(frozen=True, slots=True)
class _GroupedPayloadUnpackLayout:
    """Cached device metadata for one scalable grouped unpack launch."""

    segments: torch.Tensor
    output_numels: tuple[int, ...]
    num_blocks: int


@dataclass(frozen=True, slots=True)
class _GroupedPayloadLayout:
    """Offsets for several layer payloads inside one rank-local buffer."""

    layer_offsets: tuple[int, ...]
    layer_spans: tuple[int, ...]
    rank_stride: int


@dataclass(frozen=True, slots=True)
class _PreparedFusedGatherInputs:
    """Strong references and dynamic metadata captured for one forward."""

    layouts: tuple[_LayerPayloadLayout, ...]
    local_fields: tuple[torch.Tensor, ...]
    fields_by_layer: tuple[tuple[torch.Tensor, ...], ...]
    local_counts: tuple[int, ...]
    field_needs_grad: tuple[bool, ...]
    schema_hash: tuple[int, ...]
    device: torch.device
    world_rank: int
    world_size: int


@dataclass(frozen=True, slots=True, eq=False)
class _IntegerPayloadGenerationState:
    """One validated integer tensor that may change only at refresh."""

    node_id: str
    key: str
    tensor: torch.Tensor
    version: int


@dataclass(slots=True)
class _PendingLayerCountGather:
    """Own one in-flight fixed metadata packet and its tensor lifetimes."""

    local_packet: torch.Tensor
    gathered_packet: torch.Tensor
    work: torch.distributed.Work
    local_counts: tuple[int, ...]
    schema_hash: tuple[int, ...]
    device: torch.device
    world_rank: int
    world_size: int
    process_group: Optional[torch.distributed.ProcessGroup]
    state: Literal["pending", "finished", "discarded"] = "pending"


@dataclass(frozen=True, slots=True)
class _ResolvedLayerCountGather:
    """Validated same-forward counts and the metadata that produced them."""

    counts_by_layer: tuple[tuple[int, ...], ...]
    local_counts: tuple[int, ...]
    schema_hash: tuple[int, ...]
    device: torch.device
    world_rank: int
    world_size: int


@dataclass(slots=True)
class _PendingFusedPayloadGather:
    """Own one in-flight grouped payload and its tensor lifetimes."""

    local_grouped_payload: Optional[torch.Tensor]
    grouped_gathered_payload: Optional[torch.Tensor]
    work: Optional[torch.distributed.Work]
    state: Literal["pending", "finished", "discarded"] = "pending"


@dataclass(frozen=True, slots=True)
class _FusedGatherGenerationPlan:
    """Rank-agreed Gaussian topology reused until a structural refresh.

    Floating tensor identities and values are deliberately absent. Each
    forward captures and packs current floating parameters against immutable
    metadata. Integer payload identity/version is generation state because its
    exact-range validation intentionally runs only at structural refresh.
    """

    layouts: tuple[_LayerPayloadLayout, ...]
    field_needs_grad: tuple[bool, ...]
    resolved_counts: _ResolvedLayerCountGather
    grouped_payload_layout: _GroupedPayloadLayout
    integer_payload_state: tuple[_IntegerPayloadGenerationState, ...]

    @property
    def local_counts(self) -> tuple[int, ...]:
        return self.resolved_counts.local_counts

    @property
    def schema_hash(self) -> tuple[int, ...]:
        return self.resolved_counts.schema_hash

    @property
    def device(self) -> torch.device:
        return self.resolved_counts.device

    @property
    def world_rank(self) -> int:
        return self.resolved_counts.world_rank

    @property
    def world_size(self) -> int:
        return self.resolved_counts.world_size


@dataclass(frozen=True, slots=True)
class _PrefetchedFusedGather:
    """Pair immutable generation metadata with a payload-only work token."""

    plan: _FusedGatherGenerationPlan
    payload: _PendingFusedPayloadGather


@dataclass(frozen=True, slots=True)
class _DecodedLayerCountPacket:
    """Rank-agreed fixed-packet metadata and optional inline counts."""

    layer_count: int
    schema_hash: tuple[int, ...]
    all_integer_ranges_cached: bool
    inline_counts: Optional[tuple[tuple[int, ...], ...]]


@dataclass(frozen=True, slots=True)
class _FusedGatherResult:
    """Gathered fields plus same-forward layer counts."""

    parameters_by_node: dict[str, dict[str, torch.Tensor]]
    counts_by_node: dict[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True, eq=False)
class _PayloadIntegerRangeWatch:
    """Weakly watch one value-dependent integer field in one payload topology."""

    node_id: str
    key: str
    dtype: torch.dtype
    payload_dtype: torch.dtype
    tensor: weakref.ReferenceType[torch.Tensor]
    version: Optional[int]


class _PayloadIntegerRangeValidationState:
    """Cache range validation for one composite's live integer tensors.

    ``schema_hash`` deliberately excludes tensor values and therefore cannot
    validate integer-to-float promotion by itself.  This state advances a local
    topology generation whenever a range-sensitive field changes topology,
    tensor identity, or PyTorch version.  Weak references avoid retaining
    replaced Parameters/Buffers.  The fixed rank-agreement packet carries the
    resulting local cache bit so a miss on any rank makes every rank revalidate
    before payload communication.
    """

    __slots__ = (
        "generation",
        "_all_ranks_cached",
        "_backend",
        "_device",
        "_local_counts",
        "_process_group",
        "_schema_hash",
        "_validated_generation",
        "_watches",
        "_world_rank",
        "_world_size",
    )

    def __init__(self) -> None:
        self.generation = 0
        self._validated_generation = -1
        self._watches: tuple[_PayloadIntegerRangeWatch, ...] = ()
        self._all_ranks_cached = False
        self._schema_hash: Optional[tuple[int, ...]] = None
        self._local_counts: Optional[tuple[int, ...]] = None
        self._device: Optional[torch.device] = None
        self._world_rank: Optional[int] = None
        self._world_size: Optional[int] = None
        self._backend: Any = None
        self._process_group: Any = None

    def __reduce__(self) -> tuple[type[_PayloadIntegerRangeValidationState], tuple[Any, ...]]:
        """Serialize runtime observations as a fresh, empty cache."""
        return type(self), ()

    @staticmethod
    def _field_requires_value_validation(dtype: torch.dtype, payload_dtype: torch.dtype) -> bool:
        if dtype not in _CUDA_PAYLOAD_INTEGER_DTYPES:
            return False
        exact_limit = _FLOAT_INTEGER_EXACT_LIMIT.get(payload_dtype)
        if exact_limit is None:
            # The rank-symmetric schema validator will reject the unsupported
            # payload dtype after control agreement.  Treat it as uncached here
            # instead of failing rank-locally before that agreement.
            return True
        dtype_bounds = torch.iinfo(dtype)
        return dtype_bounds.min < -exact_limit or dtype_bounds.max > exact_limit

    @staticmethod
    def _tensor_version(tensor: torch.Tensor) -> Optional[int]:
        try:
            return tensor._version
        except RuntimeError:
            # Inference tensors have no version counter.  They remain valid
            # payload inputs while inference mode is active, but cannot enter
            # the warm cache safely, so observe them as dirty every time.
            return None

    def observe(
        self,
        layouts: tuple[_LayerPayloadLayout, ...],
        fields_by_layer: tuple[tuple[torch.Tensor, ...], ...],
        schema_hash: tuple[int, ...],
        world_rank: int,
        world_size: int,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        """Observe the current range-sensitive topology without scanning values."""
        if len(fields_by_layer) != len(layouts):
            raise RuntimeError(f"Gaussian payload has {len(fields_by_layer)} field layers, expected {len(layouts)}")

        current: list[tuple[str, str, torch.dtype, torch.dtype, torch.Tensor]] = []
        for layout, fields in zip(layouts, fields_by_layer, strict=True):
            if len(fields) != len(layout.keys):
                raise RuntimeError(
                    f"Gaussian node {layout.node_id!r} has {len(fields)} fields, expected {len(layout.keys)}"
                )
            for key, dtype, field in zip(layout.keys, layout.dtypes, fields, strict=True):
                if self._field_requires_value_validation(dtype, layout.payload_dtype):
                    current.append((layout.node_id, key, dtype, layout.payload_dtype, field))

        local_counts = tuple(fields[0].shape[0] for fields in fields_by_layer)
        device = fields_by_layer[0][0].device
        try:
            distributed_initialized = torch.distributed.is_initialized()
            if process_group is None and distributed_initialized:
                process_group = torch.distributed.distributed_c10d._get_default_group()
            backend: Any = (
                torch.distributed.get_backend(process_group)
                if distributed_initialized and process_group is not None
                else None
            )
        except (RuntimeError, ValueError):
            # Lightweight tests can mock the public distributed predicates
            # without constructing a default group.  The cache remains bound
            # to the explicit rank/world/device coordinates in that case.
            backend = None
            process_group = None
        unchanged = (
            self._schema_hash == schema_hash
            and self._local_counts == local_counts
            and self._device == device
            and self._world_rank == world_rank
            and self._world_size == world_size
            and self._backend == backend
            and self._process_group is process_group
            and len(current) == len(self._watches)
        )
        if unchanged:
            for candidate, watch in zip(current, self._watches, strict=True):
                node_id, key, dtype, payload_dtype, field = candidate
                if (
                    watch.node_id != node_id
                    or watch.key != key
                    or watch.dtype != dtype
                    or watch.payload_dtype != payload_dtype
                    or watch.tensor() is not field
                    or watch.version is None
                    or watch.version != self._tensor_version(field)
                ):
                    unchanged = False
                    break

        if not unchanged:
            self.generation += 1
            self._schema_hash = schema_hash
            self._local_counts = local_counts
            self._device = device
            self._world_rank = world_rank
            self._world_size = world_size
            self._backend = backend
            self._process_group = process_group
            self._watches = tuple(
                _PayloadIntegerRangeWatch(
                    node_id=node_id,
                    key=key,
                    dtype=dtype,
                    payload_dtype=payload_dtype,
                    tensor=weakref.ref(field),
                    version=self._tensor_version(field),
                )
                for node_id, key, dtype, payload_dtype, field in current
            )

        # A topology with no value-dependent fields is valid by construction.
        if not self._watches:
            self._validated_generation = self.generation
        self._all_ranks_cached = False

    @property
    def local_ranges_cached(self) -> bool:
        return self._validated_generation == self.generation

    @property
    def all_ranks_cached(self) -> bool:
        return self._all_ranks_cached

    def record_rank_agreement(self, all_integer_ranges_cached: bool) -> None:
        self._all_ranks_cached = all_integer_ranges_cached

    def mark_validated(self) -> None:
        self._validated_generation = self.generation
        self._all_ranks_cached = True


# Fixed int64 control-packet wire layout:
#
#   [protocol_version
#    | layer_count
#    | schema_hash[_COMMUNICATION_SCHEMA_HASH_WORD_COUNT]
#    | integer_ranges_cached
#    | inline_counts[_LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY]]
#
# The schema hash is the full SHA-256 digest represented as signed 32-bit words.
# Unused count slots are zero-filled by the encoder and ignored by the decoder.
# The fixed width lets ranks reject layer-count or schema divergence before any
# layer-count-dependent collective.  The protocol-owned inline capacity is
# intentionally independent of CUDA's grouped-launch ABI capacity: changing a
# kernel descriptor must not silently change the distributed wire format.
# Larger selections use a second count-vector AllGather. Every rank in a
# process group must run the same binary and therefore use the same packet
# width. The version field detects incompatible layouts only after a
# same-width packet exchange; it cannot make mixed-width binaries interoperable.
_COUNT_METADATA_DTYPE = torch.int64
_LAYER_COUNT_PACKET_PROTOCOL_VERSION = 1
_LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY = 8
_COMMUNICATION_SCHEMA_HASH_WORD_COUNT = 8
_LAYER_COUNT_PACKET_PROTOCOL_VERSION_INDEX = 0
_LAYER_COUNT_PACKET_LAYER_COUNT_INDEX = _LAYER_COUNT_PACKET_PROTOCOL_VERSION_INDEX + 1
_LAYER_COUNT_PACKET_SCHEMA_HASH_START = _LAYER_COUNT_PACKET_LAYER_COUNT_INDEX + 1
_LAYER_COUNT_PACKET_INTEGER_RANGES_CACHED_INDEX = (
    _LAYER_COUNT_PACKET_SCHEMA_HASH_START + _COMMUNICATION_SCHEMA_HASH_WORD_COUNT
)
_LAYER_COUNT_PACKET_INLINE_COUNTS_START = _LAYER_COUNT_PACKET_INTEGER_RANGES_CACHED_INDEX + 1
_LAYER_COUNT_PACKET_WIDTH = _LAYER_COUNT_PACKET_INLINE_COUNTS_START + _LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY
_CUDA_FUSED_GATHER_DTYPES = frozenset(
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
)
_CUDA_PAYLOAD_INTEGER_DTYPES = frozenset(
    (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
)
_CUDA_PAYLOAD_INPUT_DTYPES = _CUDA_FUSED_GATHER_DTYPES | _CUDA_PAYLOAD_INTEGER_DTYPES | frozenset((torch.bool,))
_FLOAT_INTEGER_EXACT_LIMIT = {
    torch.float16: 1 << 11,
    torch.bfloat16: 1 << 8,
    torch.float32: 1 << 24,
    torch.float64: 1 << 53,
}


# Rank agreement and payload-schema validation


@functools.lru_cache(maxsize=16)
def _communication_schema_hash(
    layouts: tuple[_LayerPayloadLayout, ...],
    field_needs_grad: tuple[bool, ...],
) -> tuple[int, ...]:
    """Hash the rank-symmetric topology that defines fused communication.

    The digest lets ranks reject ordering or schema differences before entering
    incompatible payload collectives. Dynamic first-dimension counts are
    excluded: inline counts occupy adjacent packet slots, while larger layer
    selections use a second count-vector AllGather.
    """
    expected_field_count = sum(len(layout.keys) for layout in layouts)
    if len(field_needs_grad) != expected_field_count:
        raise RuntimeError(
            f"Gaussian schema has {expected_field_count} fields but {len(field_needs_grad)} gradient flags"
        )
    schema = (
        tuple(
            (
                layout.node_id,
                layout.keys,
                layout.trailing_shapes,
                tuple(str(dtype) for dtype in layout.dtypes),
                layout.widths,
                str(layout.payload_dtype),
            )
            for layout in layouts
        ),
        field_needs_grad,
    )
    return struct.unpack(
        f">{_COMMUNICATION_SCHEMA_HASH_WORD_COUNT}i",
        hashlib.sha256(repr(schema).encode("utf-8")).digest(),
    )


def _validate_fused_schema(
    layouts: tuple[_LayerPayloadLayout, ...],
    field_needs_grad: tuple[bool, ...],
) -> None:
    """Reject unsupported topology after every rank agrees on its schema."""
    if any(not layout.payload_dtype.is_floating_point for layout in layouts):
        raise RuntimeError("fused Gaussian gathering requires floating-point layer payloads")
    unsupported_payload_dtypes = {layout.payload_dtype for layout in layouts} - _CUDA_FUSED_GATHER_DTYPES
    if unsupported_payload_dtypes:
        supported = ", ".join(str(dtype) for dtype in sorted(_CUDA_FUSED_GATHER_DTYPES, key=str))
        unsupported = ", ".join(str(dtype) for dtype in sorted(unsupported_payload_dtypes, key=str))
        raise RuntimeError(
            f"fused Gaussian gathering supports layer payload dtypes {{{supported}}}; got {{{unsupported}}}"
        )
    unsupported_input_dtypes = {
        dtype for layout in layouts for dtype in layout.dtypes if dtype not in _CUDA_PAYLOAD_INPUT_DTYPES
    }
    if unsupported_input_dtypes:
        supported = ", ".join(str(dtype) for dtype in sorted(_CUDA_PAYLOAD_INPUT_DTYPES, key=str))
        unsupported = ", ".join(str(dtype) for dtype in sorted(unsupported_input_dtypes, key=str))
        raise RuntimeError(
            f"fused Gaussian payload packing supports input dtypes {{{supported}}}; got {{{unsupported}}}"
        )

    active_payload_dtypes: set[torch.dtype] = set()
    field_offset = 0
    for layout in layouts:
        next_field_offset = field_offset + len(layout.keys)
        if any(field_needs_grad[field_offset:next_field_offset]):
            active_payload_dtypes.add(layout.payload_dtype)
        field_offset = next_field_offset
    if field_offset != len(field_needs_grad):
        raise RuntimeError(
            f"Gaussian layouts cover {field_offset} fields but received {len(field_needs_grad)} gradient flags"
        )

    _validate_gradient_pack_input_capacity(
        tuple(field for field, needs_grad in enumerate(field_needs_grad) if needs_grad)
    )

    if len(active_payload_dtypes) > 1:
        raise RuntimeError("fused Gaussian gradient packing requires one common trainable payload dtype")


def _validate_payload_integer_ranges(
    layouts: tuple[_LayerPayloadLayout, ...],
    fields_by_layer: tuple[tuple[torch.Tensor, ...], ...],
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Reject lossy integer-to-float payload promotion on every rank.

    Schema agreement has already established identical field arity and dtypes,
    so every rank either skips this check or joins the same single flag
    reduction. Empty local shards still contribute a false flag. Integer dtypes
    whose complete domain is exactly representable need no value-dependent
    check or collective.
    """
    if len(fields_by_layer) != len(layouts):
        raise RuntimeError(f"Gaussian payload has {len(fields_by_layer)} field layers, expected {len(layouts)}")

    violations: list[torch.Tensor] = []
    for layout, fields in zip(layouts, fields_by_layer, strict=True):
        if len(fields) != len(layout.dtypes):
            raise RuntimeError(
                f"Gaussian node {layout.node_id!r} has {len(fields)} fields, expected {len(layout.dtypes)}"
            )
        exact_limit = _FLOAT_INTEGER_EXACT_LIMIT[layout.payload_dtype]
        for dtype, field in zip(layout.dtypes, fields, strict=True):
            if dtype == torch.bool or dtype.is_floating_point:
                continue
            dtype_bounds = torch.iinfo(dtype)
            if dtype_bounds.min >= -exact_limit and dtype_bounds.max <= exact_limit:
                # Avoid comparing a narrow integer tensor with an out-of-range
                # Python scalar: PyTorch casts the scalar to the tensor dtype.
                continue
            violations.append(torch.any(field > exact_limit) | torch.any(field < -exact_limit))
    if not violations:
        return

    violation = torch.stack(violations).any().to(torch.int32)
    torch.distributed.all_reduce(
        violation,
        op=torch.distributed.ReduceOp.MAX,
        group=process_group,
    )
    if bool(violation.item()):
        raise RuntimeError(
            "fused Gaussian payload contains an integer value outside its promoted floating dtype's "
            "contiguous exact-integer range"
        )


def _capture_integer_payload_generation_state(
    layouts: tuple[_LayerPayloadLayout, ...],
    fields_by_layer: tuple[tuple[torch.Tensor, ...], ...],
) -> tuple[_IntegerPayloadGenerationState, ...]:
    """Capture integer identities after their generation range validation."""
    if len(fields_by_layer) != len(layouts):
        raise RuntimeError(f"Gaussian payload has {len(fields_by_layer)} field layers, expected {len(layouts)}")

    state: list[_IntegerPayloadGenerationState] = []
    for layout, fields in zip(layouts, fields_by_layer, strict=True):
        if len(fields) != len(layout.keys):
            raise RuntimeError(
                f"Gaussian node {layout.node_id!r} has {len(fields)} fields, expected {len(layout.keys)}"
            )
        for key, dtype, tensor in zip(layout.keys, layout.dtypes, fields, strict=True):
            if dtype == torch.bool or dtype.is_floating_point:
                continue
            state.append(
                _IntegerPayloadGenerationState(
                    node_id=layout.node_id,
                    key=key,
                    tensor=tensor,
                    version=tensor._version,
                )
            )
    return tuple(state)


def _validate_integer_payload_generation_state(
    plan: _FusedGatherGenerationPlan,
    fields_by_layer: tuple[tuple[torch.Tensor, ...], ...],
) -> None:
    """Reject integer mutation/replacement before a cached payload launch."""
    expected_state = plan.integer_payload_state
    state_index = 0
    for layout, fields in zip(plan.layouts, fields_by_layer, strict=True):
        for key, dtype, tensor in zip(layout.keys, layout.dtypes, fields, strict=True):
            if dtype == torch.bool or dtype.is_floating_point:
                continue
            if state_index >= len(expected_state):
                raise RuntimeError("Cached Gaussian integer payload state does not cover the published schema")
            expected = expected_state[state_index]
            state_index += 1
            if (expected.node_id, expected.key) != (layout.node_id, key):
                raise RuntimeError("Cached Gaussian integer payload state does not match the published schema")
            if tensor is not expected.tensor or tensor._version != expected.version:
                raise RuntimeError(
                    f"Cached Gaussian integer payload {layout.node_id!r}.{key} changed outside a structural refresh"
                )
    if state_index != len(expected_state):
        raise RuntimeError("Cached Gaussian integer payload state exceeds the published schema")


def _all_gather_layer_counts(
    world_size: int,
    local_counts: tuple[int, ...],
    device: torch.device,
    schema_hash: tuple[int, ...],
    integer_range_validation_state: Optional[_PayloadIntegerRangeValidationState] = None,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[tuple[int, ...], ...]:
    """Gather rank-symmetric schema and row counts without mismatched collectives.

    Every rank first gathers the same fixed-width packet, so layer-count or
    schema divergence fails before any collective whose width depends on that
    agreement. Counts that fit the inline capacity need no second collective;
    larger schemas then AllGather the complete count vector.
    """
    if device.type != "cuda":
        raise RuntimeError("coalesced Gaussian count gathering requires CUDA tensors")
    if world_size <= 1:
        raise RuntimeError("coalesced Gaussian count gathering requires multiple ranks")
    local_packet = _make_layer_count_packet(
        local_counts,
        device,
        schema_hash,
        integer_ranges_cached=(
            integer_range_validation_state.local_ranges_cached if integer_range_validation_state is not None else False
        ),
    )
    gathered_packet = torch.empty(
        world_size * local_packet.numel(),
        dtype=local_packet.dtype,
        device=device,
    )
    torch.distributed.all_gather_into_tensor(
        gathered_packet,
        local_packet,
        group=process_group,
    )
    decoded = _decode_layer_count_packet(gathered_packet, world_size)
    if decoded.layer_count != len(local_counts):
        raise RuntimeError(
            "fused Gaussian gather returned a layer count that differs from this rank's packet: "
            f"{decoded.layer_count} != {len(local_counts)}"
        )
    if decoded.schema_hash != schema_hash:
        raise RuntimeError("fused Gaussian gather returned a schema that differs from this rank's packet")
    if integer_range_validation_state is not None:
        integer_range_validation_state.record_rank_agreement(decoded.all_integer_ranges_cached)
    if decoded.inline_counts is not None:
        return decoded.inline_counts
    return _all_gather_full_layer_counts(
        world_size,
        local_counts,
        device,
        process_group=process_group,
    )


def _make_layer_count_packet(
    local_counts: tuple[int, ...],
    device: torch.device,
    schema_hash: tuple[int, ...],
    *,
    integer_ranges_cached: bool = False,
) -> torch.Tensor:
    """Encode one versioned, fixed-width rank-agreement packet."""
    if len(schema_hash) != _COMMUNICATION_SCHEMA_HASH_WORD_COUNT:
        raise ValueError(
            "fused Gaussian communication-schema hash must have "
            f"{_COMMUNICATION_SCHEMA_HASH_WORD_COUNT} words, got {len(schema_hash)}"
        )
    inline_counts = local_counts[:_LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY]
    return torch.tensor(
        (
            _LAYER_COUNT_PACKET_PROTOCOL_VERSION,
            len(local_counts),
            *schema_hash,
            int(integer_ranges_cached),
            *inline_counts,
            *(0 for _ in range(_LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY - len(inline_counts))),
        ),
        dtype=_COUNT_METADATA_DTYPE,
        device=device,
    )


def _decode_layer_count_packet(
    gathered_packet: torch.Tensor,
    world_size: int,
) -> _DecodedLayerCountPacket:
    """Validate every fixed packet row before any variable-width collective."""
    expected_numel = world_size * _LAYER_COUNT_PACKET_WIDTH
    if gathered_packet.numel() != expected_numel:
        raise RuntimeError(
            f"fused Gaussian count packet has {gathered_packet.numel()} values, expected {expected_numel}"
        )
    packet_rows: list[list[int]] = gathered_packet.reshape(world_size, _LAYER_COUNT_PACKET_WIDTH).tolist()
    protocol_versions = tuple(row[_LAYER_COUNT_PACKET_PROTOCOL_VERSION_INDEX] for row in packet_rows)
    invalid_protocol_ranks = tuple(
        rank for rank, version in enumerate(protocol_versions) if version != _LAYER_COUNT_PACKET_PROTOCOL_VERSION
    )
    if invalid_protocol_ranks:
        raise RuntimeError(
            "fused Gaussian count packet protocol version mismatch; "
            f"expected {_LAYER_COUNT_PACKET_PROTOCOL_VERSION}, got {protocol_versions} "
            f"(invalid ranks: {invalid_protocol_ranks})"
        )
    layer_counts_by_rank = tuple(row[_LAYER_COUNT_PACKET_LAYER_COUNT_INDEX] for row in packet_rows)
    mismatched_ranks = tuple(
        rank for rank, layer_count in enumerate(layer_counts_by_rank) if layer_count != layer_counts_by_rank[0]
    )
    if mismatched_ranks:
        raise RuntimeError(
            "fused Gaussian gather layer count differs across ranks; "
            f"rank 0 selected {layer_counts_by_rank[0]} layers "
            f"(ranks differing from rank 0: {mismatched_ranks})"
        )
    layer_count = layer_counts_by_rank[0]
    if layer_count <= 0:
        raise RuntimeError("coalesced Gaussian count gathering requires at least one layer")

    schema_hashes_by_rank = tuple(
        tuple(row[_LAYER_COUNT_PACKET_SCHEMA_HASH_START:_LAYER_COUNT_PACKET_INTEGER_RANGES_CACHED_INDEX])
        for row in packet_rows
    )
    mismatched_ranks = tuple(
        rank for rank, schema_hash in enumerate(schema_hashes_by_rank) if schema_hash != schema_hashes_by_rank[0]
    )
    if mismatched_ranks:
        raise RuntimeError(
            "fused Gaussian gather schema differs across ranks; "
            "layer/field order, shapes, dtypes, payload dtypes, and requires_grad must match "
            f"(ranks differing from rank 0: {mismatched_ranks})"
        )

    integer_range_cache_flags = tuple(row[_LAYER_COUNT_PACKET_INTEGER_RANGES_CACHED_INDEX] for row in packet_rows)
    invalid_cache_flags = tuple(
        (rank, flag) for rank, flag in enumerate(integer_range_cache_flags) if flag not in (0, 1)
    )
    if invalid_cache_flags:
        raise RuntimeError(f"fused Gaussian count packet has invalid integer-range cache flags: {invalid_cache_flags}")

    inline_layer_count = min(layer_count, _LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY)
    counts_by_rank = [
        row[_LAYER_COUNT_PACKET_INLINE_COUNTS_START : _LAYER_COUNT_PACKET_INLINE_COUNTS_START + inline_layer_count]
        for row in packet_rows
    ]
    invalid_counts = tuple(
        (rank, layer, count) for rank, row in enumerate(counts_by_rank) for layer, count in enumerate(row) if count < 0
    )
    if invalid_counts:
        raise RuntimeError(f"fused Gaussian gather received negative layer counts: {invalid_counts}")
    inline_counts = None
    if layer_count <= _LAYER_COUNT_PACKET_INLINE_COUNT_CAPACITY:
        inline_counts = tuple(
            tuple(counts_by_rank[rank][layer] for rank in range(world_size)) for layer in range(layer_count)
        )
    return _DecodedLayerCountPacket(
        layer_count=layer_count,
        schema_hash=schema_hashes_by_rank[0],
        all_integer_ranges_cached=all(integer_range_cache_flags),
        inline_counts=inline_counts,
    )


def _all_gather_full_layer_counts(
    world_size: int,
    local_counts: tuple[int, ...],
    device: torch.device,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[tuple[int, ...], ...]:
    """Gather the full dynamic count vector after fixed-packet agreement."""
    local = torch.tensor(local_counts, dtype=_COUNT_METADATA_DTYPE, device=device)
    gathered = torch.empty(world_size * local.numel(), dtype=local.dtype, device=device)
    torch.distributed.all_gather_into_tensor(gathered, local, group=process_group)
    counts_by_rank: list[list[int]] = gathered.reshape(world_size, len(local_counts)).tolist()
    invalid_counts = tuple(
        (rank, layer, count) for rank, row in enumerate(counts_by_rank) for layer, count in enumerate(row) if count < 0
    )
    if invalid_counts:
        raise RuntimeError(f"fused Gaussian gather received negative layer counts: {invalid_counts}")
    return tuple(tuple(counts_by_rank[rank][layer] for rank in range(world_size)) for layer in range(len(local_counts)))


def _start_layer_count_gather(
    world_size: int,
    local_counts: tuple[int, ...],
    device: torch.device,
    schema_hash: tuple[int, ...],
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> _PendingLayerCountGather:
    """Launch one fixed metadata packet and retain all asynchronous inputs."""
    if device.type != "cuda":
        raise RuntimeError("coalesced Gaussian count gathering requires CUDA tensors")
    if world_size <= 1:
        raise RuntimeError("coalesced Gaussian count gathering requires multiple ranks")
    if not torch.distributed.is_initialized():
        raise RuntimeError("coalesced Gaussian count gathering requires an initialized process group")
    actual_world_size = torch.distributed.get_world_size(process_group)
    if actual_world_size != world_size:
        raise RuntimeError(
            f"Gaussian count gather expected {world_size} ranks, but the process group has {actual_world_size}"
        )
    world_rank = torch.distributed.get_rank(process_group)
    if not 0 <= world_rank < world_size:
        raise RuntimeError(f"Gaussian count gather received invalid rank {world_rank} for world size {world_size}")

    local_packet = _make_layer_count_packet(local_counts, device, schema_hash)
    gathered_packet = torch.empty(
        world_size * local_packet.numel(),
        dtype=local_packet.dtype,
        device=device,
    )
    work: Optional[torch.distributed.Work] = None
    try:
        work = torch.distributed.all_gather_into_tensor(
            gathered_packet,
            local_packet,
            group=process_group,
            async_op=True,
        )
        if work is None:
            raise RuntimeError("asynchronous Gaussian count gathering did not return a work handle")
        return _PendingLayerCountGather(
            local_packet=local_packet,
            gathered_packet=gathered_packet,
            work=work,
            local_counts=local_counts,
            schema_hash=schema_hash,
            device=device,
            world_rank=world_rank,
            world_size=world_size,
            process_group=process_group,
        )
    except BaseException as launch_error:
        if work is not None:
            try:
                work.wait()
            except BaseException as drain_error:
                launch_error.add_note(f"Additionally failed to drain launched Gaussian count work: {drain_error!r}")
        raise


def _finish_layer_count_gather(pending: _PendingLayerCountGather) -> _ResolvedLayerCountGather:
    """Consume one fixed packet and gather counts beyond the protocol inline capacity."""
    if pending.state != "pending":
        raise RuntimeError(f"Gaussian count gather is already {pending.state}")
    try:
        pending.work.wait()
        decoded = _decode_layer_count_packet(pending.gathered_packet, pending.world_size)
        if decoded.layer_count != len(pending.local_counts):
            raise RuntimeError(
                "prefetched Gaussian layer count differs from this rank's snapshot: "
                f"{decoded.layer_count} != {len(pending.local_counts)}"
            )
        if decoded.schema_hash != pending.schema_hash:
            raise RuntimeError("prefetched Gaussian schema differs from this rank's snapshot")
        counts_by_layer = decoded.inline_counts
        if counts_by_layer is None:
            counts_by_layer = _all_gather_full_layer_counts(
                pending.world_size,
                pending.local_counts,
                pending.device,
                process_group=pending.process_group,
            )
        return _ResolvedLayerCountGather(
            counts_by_layer=counts_by_layer,
            local_counts=pending.local_counts,
            schema_hash=pending.schema_hash,
            device=pending.device,
            world_rank=pending.world_rank,
            world_size=pending.world_size,
        )
    finally:
        pending.state = "finished"


def _discard_layer_count_gather(pending: _PendingLayerCountGather) -> None:
    """Drain only the fixed asynchronous packet; repeated discards are no-ops."""
    if pending.state != "pending":
        return
    try:
        pending.work.wait()
    finally:
        pending.state = "discarded"


# Forward payload packing and collective layout


def _make_layer_layout(
    node_id: str,
    parameters: Mapping[str, torch.Tensor],
) -> _LayerPayloadLayout:
    """Record field insertion order and flattened row widths."""
    if not parameters:
        raise RuntimeError(f"Gaussian node {node_id!r} has no parameters to gather")
    keys = tuple(parameters)
    tensors = tuple(parameters[key] for key in keys)
    local_count = tensors[0].shape[0]
    if not all(tensor.shape[0] == local_count for tensor in tensors):
        raise RuntimeError(f"All parameters for Gaussian node {node_id!r} must have the same first dimension")

    trailing_shapes = tuple(tuple(tensor.shape[1:]) for tensor in tensors)
    dtypes = tuple(tensor.dtype for tensor in tensors)
    widths = tuple(functools.reduce(operator.mul, shape, 1) for shape in trailing_shapes)
    return _LayerPayloadLayout(
        node_id=node_id,
        keys=keys,
        trailing_shapes=trailing_shapes,
        dtypes=dtypes,
        widths=widths,
        payload_dtype=functools.reduce(torch.promote_types, dtypes),
    )


def _prepare_fused_gather_inputs(
    gaussian_nodes: list[str],
    parameters_by_node: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    process_group: Optional[torch.distributed.ProcessGroup],
) -> _PreparedFusedGatherInputs:
    """Capture exact tensor references and fresh dynamic metadata once."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("fused Gaussian gathering requires an initialized multi-rank process group")
    if process_group is None:
        raise RuntimeError("fused Gaussian gathering requires the dedicated process group")
    world_size = torch.distributed.get_world_size(process_group)
    if world_size <= 1:
        raise RuntimeError("fused Gaussian gathering requires an initialized multi-rank process group")
    if not gaussian_nodes or len(set(gaussian_nodes)) != len(gaussian_nodes):
        raise RuntimeError("fused Gaussian gathering requires at least one distinct Gaussian node")

    layouts: list[_LayerPayloadLayout] = []
    local_fields: list[torch.Tensor] = []
    fields_by_layer: list[tuple[torch.Tensor, ...]] = []
    local_counts: list[int] = []
    for node_id in gaussian_nodes:
        parameters = parameters_by_node[node_id]
        layout = _make_layer_layout(node_id, parameters)
        layouts.append(layout)
        fields = tuple(parameters[key] for key in layout.keys)
        fields_by_layer.append(fields)
        local_fields.extend(fields)
        local_counts.append(fields[0].shape[0])

    if len({field.device for field in local_fields}) != 1:
        raise RuntimeError("fused Gaussian gathering requires every field on one CUDA device")
    frozen_layouts = tuple(layouts)
    frozen_fields = tuple(local_fields)
    device = frozen_fields[0].device
    if device.type != "cuda":
        raise RuntimeError("fused Gaussian gradient packing requires CUDA tensors")
    if torch.distributed.get_backend(process_group) != torch.distributed.Backend.NCCL:
        raise RuntimeError("fused Gaussian gradient packing requires the NCCL backend")
    field_needs_grad = tuple(field.requires_grad for field in frozen_fields)
    return _PreparedFusedGatherInputs(
        layouts=frozen_layouts,
        local_fields=frozen_fields,
        fields_by_layer=tuple(fields_by_layer),
        local_counts=tuple(local_counts),
        field_needs_grad=field_needs_grad,
        schema_hash=_communication_schema_hash(frozen_layouts, field_needs_grad),
        device=device,
        world_rank=torch.distributed.get_rank(process_group),
        world_size=world_size,
    )


def _pack_layer_payload(
    layout: _LayerPayloadLayout,
    tensors: tuple[torch.Tensor, ...],
    max_count: Optional[int] = None,
) -> torch.Tensor:
    """Pack one rank's mixed-dtype fields into fresh contiguous storage."""
    if len(tensors) != len(layout.keys):
        raise RuntimeError(f"Gaussian node {layout.node_id!r} has {len(tensors)} tensors, expected {len(layout.keys)}")
    local_count = tensors[0].shape[0]
    if max_count is None:
        max_count = local_count
    if max_count < local_count:
        raise ValueError(f"Gaussian payload maximum count {max_count} is smaller than local count {local_count}")
    device = tensors[0].device
    if device.type != "cuda":
        raise RuntimeError("fused Gaussian payload packing requires CUDA tensors")

    flat_fields: list[torch.Tensor] = []
    for index, (tensor, width) in enumerate(zip(tensors, layout.widths, strict=True)):
        if tensor.shape[0] != local_count or tensor.numel() != local_count * width:
            raise RuntimeError(
                f"Gaussian payload field {index} has shape {tuple(tensor.shape)}, expected {local_count * width} values"
            )
        flattened = tensor.reshape(local_count, width)
        if not flattened.is_contiguous():
            flattened = flattened.contiguous()
        flat_fields.append(flattened)

    row_width = sum(layout.widths)
    packed_payload = torch.empty(max_count * row_width, dtype=layout.payload_dtype, device=device)
    gaussian_collective_payload_ops._pack_payload_trusted(
        packed_payload,
        flat_fields,
        list(layout.widths),
        local_count,
        max_count,
    )
    return packed_payload.view(max_count, row_width)


def _all_gather_padded_payload(
    world_size: int,
    padded: torch.Tensor,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Gather one same-shaped field-major payload into a rank-major buffer."""
    if world_size <= 0 or padded.ndim != 2 or not padded.is_contiguous():
        raise RuntimeError("padded Gaussian payload gathering requires a positive world size and contiguous matrix")
    gathered = padded.new_empty((world_size * padded.shape[0], padded.shape[1]))
    if gathered.numel() > 0:
        torch.distributed.all_gather_into_tensor(gathered, padded, group=process_group)
    return gathered


def _all_gather_grouped_payload(
    world_size: int,
    local_payloads: torch.Tensor,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Gather one flat rank-local buffer containing several padded layers."""
    if world_size <= 0 or local_payloads.ndim != 1 or not local_payloads.is_contiguous():
        raise RuntimeError("grouped Gaussian payload gathering requires a positive world size and contiguous vector")
    gathered = local_payloads.new_empty(world_size * local_payloads.numel())
    if gathered.numel() > 0:
        torch.distributed.all_gather_into_tensor(
            gathered,
            local_payloads,
            group=process_group,
        )
    return gathered


def _make_grouped_payload_layout(
    layouts: tuple[_LayerPayloadLayout, ...],
    counts_by_layer: tuple[tuple[int, ...], ...],
) -> _GroupedPayloadLayout:
    """Lay padded layer payloads back-to-back without adding communicated bytes."""
    if len(layouts) != len(counts_by_layer):
        raise ValueError(f"grouped payload has {len(layouts)} layouts but {len(counts_by_layer)} count vectors")
    if not layouts:
        raise ValueError("grouped payload requires at least one layer")

    layer_offsets: list[int] = []
    layer_spans: list[int] = []
    rank_stride = 0
    world_size: Optional[int] = None
    for layout, counts in zip(layouts, counts_by_layer, strict=True):
        if not counts:
            raise ValueError(f"grouped payload layer {layout.node_id!r} requires at least one rank")
        if any(count < 0 for count in counts):
            raise ValueError(f"grouped payload layer {layout.node_id!r} has negative counts {counts}")
        if world_size is None:
            world_size = len(counts)
        elif len(counts) != world_size:
            raise ValueError(f"grouped payload layer {layout.node_id!r} has {len(counts)} ranks, expected {world_size}")
        row_width = sum(layout.widths)
        layer_span = max(counts) * row_width
        layer_offsets.append(rank_stride)
        layer_spans.append(layer_span)
        rank_stride += layer_span

    return _GroupedPayloadLayout(
        layer_offsets=tuple(layer_offsets),
        layer_spans=tuple(layer_spans),
        rank_stride=rank_stride,
    )


def _can_route_node_selection(selected_nodes: Sequence[str]) -> bool:
    """Return whether one node selection can use the grouped payload route."""
    return 1 <= len(selected_nodes) <= gaussian_collective_payload_ops._MAX_GROUPED_PAYLOAD_LAYERS and len(
        set(selected_nodes)
    ) == len(selected_nodes)


def _can_group_payloads(
    layouts: tuple[_LayerPayloadLayout, ...],
    devices_by_layer: tuple[tuple[torch.device, ...], ...],
) -> bool:
    """Check rank-independent grouped pack and unpack layout compatibility."""
    if not _can_route_node_selection(tuple(layout.node_id for layout in layouts)):
        return False
    if len(devices_by_layer) != len(layouts):
        return False
    if any(len(devices) != len(layout.keys) for layout, devices in zip(layouts, devices_by_layer, strict=True)):
        return False

    # Python needs the shared grouped-layer launch capacity to choose the
    # fallback. Grouped payload segment capacities remain private to the
    # native entry points, which validate them immediately before launch; the
    # separately exported gradient capacity is validated before this point.
    if len({layout.payload_dtype for layout in layouts}) != 1:
        return False

    devices = {device for layer_devices in devices_by_layer for device in layer_devices}
    return len(devices) == 1 and next(iter(devices)).type == "cuda"


def _prepared_fields_by_layer(
    prepared: _PreparedFusedGatherInputs,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Split one frozen field tuple according to its rank-agreed layouts."""
    fields_by_layer: list[tuple[torch.Tensor, ...]] = []
    field_offset = 0
    for layout in prepared.layouts:
        next_field_offset = field_offset + len(layout.keys)
        layer_fields = prepared.local_fields[field_offset:next_field_offset]
        if len(layer_fields) != len(layout.keys):
            raise RuntimeError(f"Prepared Gaussian node {layout.node_id!r} is missing field references")
        fields_by_layer.append(layer_fields)
        field_offset = next_field_offset
    if field_offset != len(prepared.local_fields):
        raise RuntimeError(
            f"Prepared Gaussian layouts cover {field_offset} fields, found {len(prepared.local_fields)} references"
        )
    return tuple(fields_by_layer)


def _can_prefetch_grouped_payload(prepared: _PreparedFusedGatherInputs) -> bool:
    """Check static grouped-kernel eligibility before resolving dynamic counts."""
    fields_by_layer = _prepared_fields_by_layer(prepared)
    devices_by_layer = tuple(tuple(field.device for field in layer_fields) for layer_fields in fields_by_layer)
    return _can_group_payloads(prepared.layouts, devices_by_layer)


def _generation_plan_matches_prepared(
    plan: _FusedGatherGenerationPlan,
    prepared: _PreparedFusedGatherInputs,
) -> bool:
    """Return whether fresh inputs still belong to the cached generation."""
    return (
        prepared.layouts == plan.layouts
        and prepared.local_counts == plan.local_counts
        and prepared.field_needs_grad == plan.field_needs_grad
        and prepared.schema_hash == plan.schema_hash
        and prepared.device == plan.device
        and prepared.world_rank == plan.world_rank
        and prepared.world_size == plan.world_size
    )


def _prepare_fused_gather_inputs_from_generation_plan(
    gaussian_nodes: list[str],
    parameters_by_node: Mapping[str, Mapping[str, torch.Tensor]],
    plan: _FusedGatherGenerationPlan,
    *,
    process_group: Optional[torch.distributed.ProcessGroup],
) -> _PreparedFusedGatherInputs:
    """Capture current tensors while reusing one rank-agreed topology plan."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("cached Gaussian gathering requires an initialized process group")
    if process_group is None:
        raise RuntimeError("cached Gaussian gathering requires the dedicated process group")
    actual_world_rank = torch.distributed.get_rank(process_group)
    actual_world_size = torch.distributed.get_world_size(process_group)
    if (actual_world_rank, actual_world_size) != (plan.world_rank, plan.world_size):
        raise RuntimeError(
            "Cached Gaussian process-group coordinates changed between generations: "
            f"{(plan.world_rank, plan.world_size)} != {(actual_world_rank, actual_world_size)}"
        )
    if torch.distributed.get_backend(process_group) != torch.distributed.Backend.NCCL:
        raise RuntimeError("cached Gaussian gathering requires the NCCL backend")

    planned_nodes = tuple(layout.node_id for layout in plan.layouts)
    if tuple(gaussian_nodes) != planned_nodes:
        raise RuntimeError(
            f"Cached Gaussian node order changed between generations: {planned_nodes} != {tuple(gaussian_nodes)}"
        )

    fields_by_layer: list[tuple[torch.Tensor, ...]] = []
    local_fields: list[torch.Tensor] = []
    local_counts: list[int] = []
    for layout in plan.layouts:
        parameters = parameters_by_node[layout.node_id]
        if tuple(parameters) != layout.keys:
            raise RuntimeError(
                f"Cached Gaussian field order changed for {layout.node_id!r}: {layout.keys} != {tuple(parameters)}"
            )
        fields = tuple(parameters[key] for key in layout.keys)
        if not fields:
            raise RuntimeError(f"Cached Gaussian node {layout.node_id!r} has no fields")
        local_count = fields[0].shape[0]
        for key, field, trailing_shape, dtype in zip(
            layout.keys,
            fields,
            layout.trailing_shapes,
            layout.dtypes,
            strict=True,
        ):
            if field.shape[0] != local_count or tuple(field.shape[1:]) != trailing_shape:
                raise RuntimeError(
                    f"Cached Gaussian field shape changed for {layout.node_id!r}.{key}: "
                    f"{(local_count, *trailing_shape)} != {tuple(field.shape)}"
                )
            if field.dtype != dtype:
                raise RuntimeError(
                    f"Cached Gaussian field dtype changed for {layout.node_id!r}.{key}: {dtype} != {field.dtype}"
                )
            if field.device != plan.device:
                raise RuntimeError(
                    f"Cached Gaussian field device changed for {layout.node_id!r}.{key}: "
                    f"{plan.device} != {field.device}"
                )
        fields_by_layer.append(fields)
        local_fields.extend(fields)
        local_counts.append(local_count)

    frozen_fields = tuple(local_fields)
    field_needs_grad = tuple(field.requires_grad for field in frozen_fields)
    if tuple(local_counts) != plan.local_counts:
        raise RuntimeError(
            "Cached Gaussian row counts changed outside the structural boundary: "
            f"{plan.local_counts} != {tuple(local_counts)}"
        )
    if field_needs_grad != plan.field_needs_grad:
        raise RuntimeError("Cached Gaussian requires_grad schema changed outside the structural boundary")
    frozen_fields_by_layer = tuple(fields_by_layer)
    return _PreparedFusedGatherInputs(
        layouts=plan.layouts,
        local_fields=frozen_fields,
        fields_by_layer=frozen_fields_by_layer,
        local_counts=plan.local_counts,
        field_needs_grad=plan.field_needs_grad,
        schema_hash=plan.schema_hash,
        device=plan.device,
        world_rank=plan.world_rank,
        world_size=plan.world_size,
    )


def _flatten_grouped_payload_fields(
    layouts: tuple[_LayerPayloadLayout, ...],
    fields_by_layer: tuple[tuple[torch.Tensor, ...], ...],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Flatten grouped-pack inputs without changing the original autograd inputs."""
    if len(fields_by_layer) != len(layouts):
        raise RuntimeError(f"Grouped Gaussian payload has {len(fields_by_layer)} field layers, expected {len(layouts)}")

    flattened_by_layer: list[tuple[torch.Tensor, ...]] = []
    for layout, layer_fields in zip(layouts, fields_by_layer, strict=True):
        if len(layer_fields) != len(layout.widths):
            raise RuntimeError(
                f"Grouped Gaussian node {layout.node_id!r} has {len(layer_fields)} fields, "
                f"expected {len(layout.widths)}"
            )
        local_count = layer_fields[0].shape[0]
        flattened_fields: list[torch.Tensor] = []
        for field, width in zip(layer_fields, layout.widths, strict=True):
            if field.shape[0] != local_count or field.numel() != local_count * width:
                raise RuntimeError(
                    f"Grouped Gaussian node {layout.node_id!r} field has shape {tuple(field.shape)}, "
                    f"expected {local_count * width} values"
                )
            if field.ndim == 2 and field.shape[1] == width and field.is_contiguous():
                flattened = field
            else:
                flattened = field.reshape(local_count, width)
                if not flattened.is_contiguous():
                    flattened = flattened.contiguous()
            flattened_fields.append(flattened)
        flattened_by_layer.append(tuple(flattened_fields))
    return tuple(flattened_by_layer)


def _make_fused_gather_generation_plan(
    prepared: _PreparedFusedGatherInputs,
    resolved_counts: _ResolvedLayerCountGather,
    *,
    process_group: Optional[torch.distributed.ProcessGroup],
) -> _FusedGatherGenerationPlan:
    """Publish immutable metadata after one rank-agreed count/schema packet."""
    if not _can_prefetch_grouped_payload(prepared):
        raise RuntimeError("A cached Gaussian generation plan requires the grouped payload route")
    if resolved_counts.local_counts != prepared.local_counts:
        raise RuntimeError(
            "Resolved Gaussian plan counts differ from the captured generation: "
            f"{resolved_counts.local_counts} != {prepared.local_counts}"
        )
    if resolved_counts.schema_hash != prepared.schema_hash:
        raise RuntimeError("Resolved Gaussian plan schema differs from the captured generation")
    if resolved_counts.device != prepared.device:
        raise RuntimeError("Resolved Gaussian plan device differs from the captured generation")
    if (resolved_counts.world_rank, resolved_counts.world_size) != (
        prepared.world_rank,
        prepared.world_size,
    ):
        raise RuntimeError("Resolved Gaussian plan process-group coordinates differ from the captured generation")
    if len(resolved_counts.counts_by_layer) != len(prepared.layouts):
        raise RuntimeError(
            "Resolved Gaussian plan layer count differs from the captured generation: "
            f"{len(resolved_counts.counts_by_layer)} != {len(prepared.layouts)}"
        )

    _validate_fused_schema(prepared.layouts, prepared.field_needs_grad)
    _validate_payload_integer_ranges(
        prepared.layouts,
        prepared.fields_by_layer,
        process_group,
    )
    for layout, local_count, counts in zip(
        prepared.layouts,
        prepared.local_counts,
        resolved_counts.counts_by_layer,
        strict=True,
    ):
        if len(counts) != prepared.world_size:
            raise RuntimeError(
                f"Resolved Gaussian plan for {layout.node_id!r} has {len(counts)} ranks, expected {prepared.world_size}"
            )
        if counts[prepared.world_rank] != local_count:
            raise RuntimeError(
                f"Resolved Gaussian plan count mismatch for {layout.node_id!r}: "
                f"rank {prepared.world_rank} count {counts[prepared.world_rank]} != local count {local_count}"
            )

    return _FusedGatherGenerationPlan(
        layouts=prepared.layouts,
        field_needs_grad=prepared.field_needs_grad,
        resolved_counts=resolved_counts,
        grouped_payload_layout=_make_grouped_payload_layout(
            prepared.layouts,
            resolved_counts.counts_by_layer,
        ),
        integer_payload_state=_capture_integer_payload_generation_state(
            prepared.layouts,
            prepared.fields_by_layer,
        ),
    )


def _resolve_fused_gather_generation_plan(
    prepared: _PreparedFusedGatherInputs,
    *,
    process_group: Optional[torch.distributed.ProcessGroup],
) -> _FusedGatherGenerationPlan:
    """Resolve and validate one generation at the joint-gather boundary."""
    # FreePose does not receive or mutate Gaussian parameters. Revalidating
    # rank-local tensor versions here, after peers may already have entered a
    # generation payload, would turn an invariant violation into mismatched
    # distributed control flow. Exact strategy schedules and explicit public
    # mutation boundaries own topology freshness instead.
    if not _can_prefetch_grouped_payload(prepared):
        raise RuntimeError("Cached grouped Gaussian payload is not supported by the captured schema")
    with ScopedTimer("GaussiansComposite/gather_parameter_counts_generation_warmup"):
        counts_by_layer = _all_gather_layer_counts(
            prepared.world_size,
            prepared.local_counts,
            prepared.device,
            prepared.schema_hash,
            process_group=process_group,
        )
    return _make_fused_gather_generation_plan(
        prepared,
        _ResolvedLayerCountGather(
            counts_by_layer=counts_by_layer,
            local_counts=prepared.local_counts,
            schema_hash=prepared.schema_hash,
            device=prepared.device,
            world_rank=prepared.world_rank,
            world_size=prepared.world_size,
        ),
        process_group=process_group,
    )


def _start_fused_payload_gather_from_generation_plan(
    prepared: _PreparedFusedGatherInputs,
    plan: _FusedGatherGenerationPlan,
    *,
    process_group: Optional[torch.distributed.ProcessGroup],
) -> _PendingFusedPayloadGather:
    """Launch a payload snapshot with no steady-forward metadata collective."""
    if not _generation_plan_matches_prepared(plan, prepared):
        raise RuntimeError("Fresh Gaussian tensors do not match the published generation plan")
    _validate_integer_payload_generation_state(plan, prepared.fields_by_layer)

    grouped_payload_layout = plan.grouped_payload_layout
    work: Optional[torch.distributed.Work] = None
    try:
        with ScopedTimer("GaussiansComposite/launch_cached_generation_grouped_payload"):
            local_grouped_payload = torch.empty(
                grouped_payload_layout.rank_stride,
                dtype=prepared.layouts[0].payload_dtype,
                device=prepared.device,
            )
            flattened_fields_by_layer = _flatten_grouped_payload_fields(
                prepared.layouts,
                prepared.fields_by_layer,
            )
            gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                local_grouped_payload,
                flattened_fields_by_layer,
                tuple(layout.widths for layout in prepared.layouts),
                prepared.local_counts,
                tuple(max(counts) for counts in plan.resolved_counts.counts_by_layer),
            )
            grouped_gathered_payload = local_grouped_payload.new_empty(
                prepared.world_size * local_grouped_payload.numel()
            )
            if grouped_gathered_payload.numel() > 0:
                work = torch.distributed.all_gather_into_tensor(
                    grouped_gathered_payload,
                    local_grouped_payload,
                    group=process_group,
                    async_op=True,
                )
                if work is None:
                    raise RuntimeError("asynchronous cached Gaussian payload gathering did not return a work handle")

        return _PendingFusedPayloadGather(
            local_grouped_payload=local_grouped_payload,
            grouped_gathered_payload=grouped_gathered_payload,
            work=work,
        )
    except BaseException as launch_error:
        if work is not None:
            try:
                work.wait()
            except BaseException as drain_error:
                launch_error.add_note(f"Additionally failed to drain launched Gaussian payload work: {drain_error!r}")
        raise


def _finish_fused_payload_gather(pending: _PendingFusedPayloadGather) -> Optional[torch.Tensor]:
    """Wait once for a prefetched payload and transfer ownership to its resolver."""
    if pending.state != "pending":
        raise RuntimeError(f"Gaussian payload gather is already {pending.state}")
    gathered_payload = pending.grouped_gathered_payload
    try:
        if pending.work is not None:
            pending.work.wait()
        return gathered_payload
    finally:
        pending.local_grouped_payload = None
        pending.grouped_gathered_payload = None
        pending.work = None
        pending.state = "finished"


def _discard_fused_payload_gather(pending: _PendingFusedPayloadGather) -> None:
    """Drain an unused payload collective; repeated discards are no-ops."""
    if pending.state != "pending":
        return
    try:
        if pending.work is not None:
            pending.work.wait()
    finally:
        pending.local_grouped_payload = None
        pending.grouped_gathered_payload = None
        pending.work = None
        pending.state = "discarded"


# CUDA unpack metadata and field reconstruction


def _payload_unpack_segment_rows(
    counts: tuple[int, ...],
    widths: tuple[int, ...],
    max_count: int,
    *,
    field_major: bool = True,
    rank_stride: Optional[int] = None,
    layer_offset: int = 0,
) -> tuple[tuple[tuple[int, int, int, int, int], ...], int, int]:
    """Plan copies from rank-major wire slices into contiguous SoA output.

    In production, each rank's layer slice stores every field's valid values in
    schema order, with padding only after the final field. The destination
    concatenates all rank shards of one field before advancing to the next.
    ``rank_stride`` and ``layer_offset`` select the same geometry inside a
    grouped rank buffer. ``field_major=False`` remains only for legacy-layout
    tests.
    """
    if not counts:
        raise ValueError("Gaussian payload unpack requires at least one rank")
    if any(count < 0 for count in counts):
        raise ValueError(f"Gaussian payload unpack counts must be nonnegative, got {counts}")
    if max(counts) != max_count:
        raise ValueError(f"Gaussian payload unpack maximum count {max_count} does not match {counts}")
    if any(width < 0 for width in widths):
        raise ValueError(f"Gaussian payload unpack widths must be nonnegative, got {widths}")

    values_per_block = int(gaussian_collective_payload_ops._PAYLOAD_UNPACK_VALUES_PER_BLOCK)

    world_size = len(counts)
    gathered_count = sum(counts)
    row_width = sum(widths)
    layer_span = max_count * row_width
    if rank_stride is None:
        rank_stride = layer_span
    if rank_stride < 0:
        raise ValueError(f"Gaussian payload rank stride must be nonnegative, got {rank_stride}")
    if layer_offset < 0 or layer_offset > rank_stride or layer_span > rank_stride - layer_offset:
        raise ValueError(
            f"Gaussian payload layer range [{layer_offset}, {layer_offset + layer_span}) "
            f"exceeds rank stride {rank_stride}"
        )
    input_numel = world_size * rank_stride
    destination_offset = 0
    field_offset = 0
    num_blocks = 0
    segments: list[tuple[int, int, int, int, int]] = []

    for width in widths:
        field_destination_start = destination_offset
        for rank, count in enumerate(counts):
            length = count * width
            if length:
                rank_start = rank * rank_stride + layer_offset
                if field_major:
                    source_offset = rank_start + count * field_offset
                    source_end = source_offset + length
                    source_width = 1
                else:
                    source_offset = rank_start + field_offset
                    source_end = source_offset + (count - 1) * row_width + width
                    source_width = width
                if source_offset < 0 or source_end > input_numel:
                    raise ValueError(
                        f"Gaussian payload unpack source [{source_offset}, {source_end}) "
                        f"exceeds input size {input_numel}"
                    )
                num_blocks += (length + values_per_block - 1) // values_per_block
                if num_blocks > torch.iinfo(torch.int32).max:
                    raise ValueError("Gaussian payload unpack block count overflows int32")
                segments.append((destination_offset, length, source_offset, source_width, num_blocks))
            destination_offset += length

        expected_field_end = field_destination_start + gathered_count * width
        if destination_offset != expected_field_end:
            raise ValueError(
                f"Gaussian payload unpack field ends at {destination_offset}, expected {expected_field_end}"
            )
        field_offset += width

    output_numel = gathered_count * row_width
    if field_offset != row_width:
        raise ValueError(f"Gaussian payload unpack fields cover {field_offset}, expected row width {row_width}")
    if destination_offset != output_numel:
        raise ValueError(
            f"Gaussian payload unpack rows cover {destination_offset}, expected output size {output_numel}"
        )
    if (output_numel == 0) != (not segments and num_blocks == 0):
        raise ValueError("empty Gaussian SoA output, segments, and block count must occur together")
    if output_numel > 0 and (not segments or num_blocks <= 0):
        raise ValueError("nonempty Gaussian payload unpack requires positive work")
    return tuple(segments), output_numel, num_blocks


@functools.lru_cache(maxsize=16)
def _cached_payload_unpack_layout(
    counts: tuple[int, ...],
    widths: tuple[int, ...],
    max_count: int,
    rank_stride: int,
    layer_offset: int,
    device: torch.device,
    stream_id: int,
) -> _PayloadUnpackLayout:
    """Upload exact field-major copy metadata once per topology and CUDA stream.

    The raw ``cudaStream_t`` is part of the key. Externally managed streams
    must not be destroyed and recreated while cached entries remain live,
    because CUDA may recycle that integer handle.
    """
    if device.type != "cuda":
        raise RuntimeError("Gaussian payload metadata must be cached on CUDA")
    del stream_id
    with ScopedTimer("GaussiansComposite/prepare_payload_unpack_metadata"):
        segments, output_numel, num_blocks = _payload_unpack_segment_rows(
            counts,
            widths,
            max_count,
            field_major=True,
            rank_stride=rank_stride,
            layer_offset=layer_offset,
        )
        segment_tensor = (
            torch.tensor(segments, dtype=torch.int64, device=device)
            if segments
            else torch.empty((0, 5), dtype=torch.int64, device=device)
        )
        return _PayloadUnpackLayout(
            segments=segment_tensor,
            output_numel=output_numel,
            num_blocks=num_blocks,
        )


def _grouped_payload_unpack_segment_rows(
    counts_by_layer: tuple[tuple[int, ...], ...],
    widths_by_layer: tuple[tuple[int, ...], ...],
    grouped_layout: _GroupedPayloadLayout,
) -> tuple[tuple[tuple[int, int, int, int, int], ...], tuple[int, ...], int]:
    """Build one rank-scalable copy table for a grouped payload.

    Rows are ``[output, destination, length, source, cumulative block end]``.
    Rank-dependent geometry lives in this cached device table rather than the
    CUDA launch argument block, so its size can grow through NVL72 without
    changing the kernel ABI.
    """
    if not counts_by_layer:
        raise ValueError("grouped Gaussian payload unpack requires at least one layer")
    if len(counts_by_layer) != len(widths_by_layer):
        raise ValueError(
            f"grouped Gaussian payload has {len(counts_by_layer)} count vectors "
            f"but {len(widths_by_layer)} width vectors"
        )
    if len(grouped_layout.layer_offsets) != len(counts_by_layer) or len(grouped_layout.layer_spans) != len(
        counts_by_layer
    ):
        raise ValueError("grouped Gaussian payload offsets and spans must match the layer count")

    world_size = len(counts_by_layer[0])
    if world_size <= 0:
        raise ValueError("grouped Gaussian payload unpack requires at least one rank")
    if grouped_layout.rank_stride < 0:
        raise ValueError(f"grouped Gaussian payload rank stride must be nonnegative, got {grouped_layout.rank_stride}")
    input_numel = world_size * grouped_layout.rank_stride
    values_per_block = int(gaussian_collective_payload_ops._PAYLOAD_UNPACK_VALUES_PER_BLOCK)

    segments: list[tuple[int, int, int, int, int]] = []
    output_numels: list[int] = []
    num_blocks = 0
    expected_layer_offset = 0
    for output_index, (counts, widths, layer_offset, layer_span) in enumerate(
        zip(
            counts_by_layer,
            widths_by_layer,
            grouped_layout.layer_offsets,
            grouped_layout.layer_spans,
            strict=True,
        )
    ):
        if len(counts) != world_size:
            raise ValueError(
                f"grouped Gaussian payload layer {output_index} has counts for {len(counts)} ranks, "
                f"expected {world_size}"
            )
        if any(count < 0 for count in counts):
            raise ValueError(f"grouped Gaussian payload layer {output_index} has negative counts {counts}")
        if any(width < 0 for width in widths):
            raise ValueError(f"grouped Gaussian payload layer {output_index} has negative widths {widths}")
        if layer_offset != expected_layer_offset:
            raise ValueError(
                f"grouped Gaussian payload layer {output_index} starts at {layer_offset}, "
                f"expected {expected_layer_offset}"
            )

        row_width = sum(widths)
        expected_layer_span = max(counts) * row_width
        if layer_span != expected_layer_span:
            raise ValueError(
                f"grouped Gaussian payload layer {output_index} has span {layer_span}, expected {expected_layer_span}"
            )
        expected_layer_offset += layer_span

        gathered_count = sum(counts)
        output_numel = gathered_count * row_width
        output_numels.append(output_numel)
        destination_offset = 0
        field_offset = 0
        for width in widths:
            field_destination_start = destination_offset
            for rank, count in enumerate(counts):
                length = count * width
                if length:
                    rank_start = rank * grouped_layout.rank_stride + layer_offset
                    source_offset = rank_start + count * field_offset
                    source_end = source_offset + length
                    if source_offset < 0 or source_end > input_numel:
                        raise ValueError(
                            f"grouped Gaussian payload source [{source_offset}, {source_end}) "
                            f"exceeds input size {input_numel}"
                        )
                    num_blocks += (length + values_per_block - 1) // values_per_block
                    if num_blocks > torch.iinfo(torch.int32).max:
                        raise ValueError("grouped Gaussian payload unpack block count overflows int32")
                    segments.append((output_index, destination_offset, length, source_offset, num_blocks))
                destination_offset += length

            expected_field_end = field_destination_start + gathered_count * width
            if destination_offset != expected_field_end:
                raise ValueError(
                    f"grouped Gaussian payload layer {output_index} field ends at {destination_offset}, "
                    f"expected {expected_field_end}"
                )
            field_offset += width

        if field_offset != row_width:
            raise ValueError(
                f"grouped Gaussian payload layer {output_index} fields cover {field_offset}, "
                f"expected row width {row_width}"
            )
        if destination_offset != output_numel:
            raise ValueError(
                f"grouped Gaussian payload layer {output_index} rows cover {destination_offset}, "
                f"expected output size {output_numel}"
            )

    if expected_layer_offset != grouped_layout.rank_stride:
        raise ValueError(
            f"grouped Gaussian payload layers cover rank stride {expected_layer_offset}, "
            f"expected {grouped_layout.rank_stride}"
        )
    total_output_numel = sum(output_numels)
    if (total_output_numel == 0) != (not segments and num_blocks == 0):
        raise ValueError("empty grouped Gaussian outputs, segments, and block count must occur together")
    if total_output_numel > 0 and (not segments or num_blocks <= 0):
        raise ValueError("nonempty grouped Gaussian payload unpack requires positive work")
    return tuple(segments), tuple(output_numels), num_blocks


@functools.lru_cache(maxsize=16)
def _cached_grouped_payload_unpack_layout(
    counts_by_layer: tuple[tuple[int, ...], ...],
    widths_by_layer: tuple[tuple[int, ...], ...],
    grouped_layout: _GroupedPayloadLayout,
    device: torch.device,
    stream_id: int,
) -> _GroupedPayloadUnpackLayout:
    """Upload scalable grouped-unpack metadata once per topology and stream.

    The raw stream handle is stable for PyTorch-owned streams; callers that
    destroy external streams must clear this cache before CUDA can reuse it.
    """
    if device.type != "cuda":
        raise RuntimeError("grouped Gaussian payload metadata must be cached on CUDA")
    del stream_id
    with ScopedTimer("GaussiansComposite/prepare_grouped_payload_unpack_metadata"):
        segments, output_numels, num_blocks = _grouped_payload_unpack_segment_rows(
            counts_by_layer,
            widths_by_layer,
            grouped_layout,
        )
        segment_tensor = (
            torch.tensor(segments, dtype=torch.int64, device=device)
            if segments
            else torch.empty((0, 5), dtype=torch.int64, device=device)
        )
        return _GroupedPayloadUnpackLayout(
            segments=segment_tensor,
            output_numels=output_numels,
            num_blocks=num_blocks,
        )


def _unpack_layer_payload(
    layout: _LayerPayloadLayout,
    gathered: torch.Tensor,
    counts: tuple[int, ...],
    max_count: int,
    *,
    rank_stride: Optional[int] = None,
    layer_offset: int = 0,
) -> dict[str, torch.Tensor]:
    """Unpack one layer from a padded rank-major payload into contiguous SoA."""
    if gathered.device.type != "cuda":
        raise RuntimeError("fused Gaussian payload unpack requires CUDA tensors")
    if not gathered.is_contiguous():
        raise RuntimeError("fused Gaussian gathered payload must be contiguous")
    layer_span = max_count * sum(layout.widths)
    if rank_stride is None:
        rank_stride = layer_span
    if rank_stride < 0:
        raise ValueError(f"Gaussian payload rank stride must be nonnegative, got {rank_stride}")
    if layer_offset < 0 or layer_offset > rank_stride or layer_span > rank_stride - layer_offset:
        raise ValueError(
            f"Gaussian payload layer range [{layer_offset}, {layer_offset + layer_span}) "
            f"exceeds rank stride {rank_stride}"
        )
    if gathered.numel() != len(counts) * rank_stride:
        raise ValueError(
            f"Gaussian gathered payload has {gathered.numel()} values, expected {len(counts) * rank_stride}"
        )

    output_numel = sum(counts) * sum(layout.widths)
    unpacked_payload = gathered.new_empty(output_numel)
    unpack_layout = _cached_payload_unpack_layout(
        counts,
        layout.widths,
        max_count,
        rank_stride,
        layer_offset,
        gathered.device,
        torch.cuda.current_stream(gathered.device).cuda_stream,
    )
    if unpack_layout.output_numel != output_numel:
        raise RuntimeError(
            f"Gaussian payload metadata output has {unpack_layout.output_numel} values, expected {output_numel}"
        )
    contiguous_source = gathered.reshape(gathered.numel(), 1)
    gaussian_collective_payload_ops._unpack_payload_trusted(
        unpacked_payload,
        contiguous_source,
        unpack_layout.segments,
        unpack_layout.num_blocks,
        1,
    )

    return _split_unpacked_layer_payload(layout, unpacked_payload, counts)


def _split_unpacked_layer_payload(
    layout: _LayerPayloadLayout,
    unpacked_payload: torch.Tensor,
    counts: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Restore one contiguous SoA payload to its named field tensors."""
    gathered_count = sum(counts)
    output_numel = gathered_count * sum(layout.widths)
    if unpacked_payload.numel() != output_numel:
        raise ValueError(f"Gaussian unpacked payload has {unpacked_payload.numel()} values, expected {output_numel}")
    outputs: dict[str, torch.Tensor] = {}
    field_offset = 0
    for key, trailing_shape, dtype, width in zip(
        layout.keys,
        layout.trailing_shapes,
        layout.dtypes,
        layout.widths,
        strict=True,
    ):
        field_numel = gathered_count * width
        field = unpacked_payload.narrow(0, field_offset, field_numel)
        outputs[key] = field.reshape(gathered_count, *trailing_shape).to(dtype)
        field_offset += field_numel
    if field_offset != output_numel:
        raise RuntimeError(f"Gaussian unpacked fields cover {field_offset} values, expected {output_numel}")
    return outputs


def _unpack_grouped_layer_payloads(
    layouts: tuple[_LayerPayloadLayout, ...],
    gathered: torch.Tensor,
    counts_by_layer: tuple[tuple[int, ...], ...],
    grouped_layout: _GroupedPayloadLayout,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Unpack a production grouped payload with one CUDA launch."""
    if not layouts:
        raise ValueError("grouped Gaussian payload unpack requires at least one layer")
    if len(counts_by_layer) != len(layouts):
        raise ValueError(
            f"grouped Gaussian payload has {len(layouts)} layouts but {len(counts_by_layer)} count vectors"
        )

    if not counts_by_layer[0]:
        raise ValueError("grouped Gaussian payload unpack requires at least one rank")
    world_size = len(counts_by_layer[0])
    if any(len(counts) != world_size for counts in counts_by_layer):
        raise ValueError("grouped Gaussian payload count vectors must have one shared rank count")
    expected_input_numel = world_size * grouped_layout.rank_stride
    if gathered.numel() != expected_input_numel:
        raise ValueError(
            f"grouped Gaussian gathered payload has {gathered.numel()} values, expected {expected_input_numel}"
        )

    unpacked_payloads = tuple(
        gathered.new_empty(sum(counts) * sum(layout.widths))
        for layout, counts in zip(layouts, counts_by_layer, strict=True)
    )
    widths_by_layer = tuple(layout.widths for layout in layouts)
    unpack_layout = _cached_grouped_payload_unpack_layout(
        counts_by_layer,
        widths_by_layer,
        grouped_layout,
        gathered.device,
        torch.cuda.current_stream(gathered.device).cuda_stream,
    )
    actual_output_numels = tuple(output.numel() for output in unpacked_payloads)
    if actual_output_numels != unpack_layout.output_numels:
        raise RuntimeError(
            f"grouped Gaussian payload outputs have sizes {actual_output_numels}, "
            f"expected {unpack_layout.output_numels}"
        )
    gaussian_collective_payload_ops._unpack_grouped_payload_trusted(
        unpacked_payloads,
        gathered,
        unpack_layout.segments,
        unpack_layout.num_blocks,
    )
    return tuple(
        _split_unpacked_layer_payload(layout, unpacked, counts)
        for layout, unpacked, counts in zip(layouts, unpacked_payloads, counts_by_layer, strict=True)
    )


# Backward gradient packing


def _validate_gradient_pack_input_capacity(active_fields: tuple[int, ...]) -> None:
    """Keep Python plans within the extension's by-value descriptor ABI."""
    capacity = int(gaussian_collective_payload_ops._MAX_GRADIENT_PACK_INPUTS)
    if len(active_fields) > capacity:
        raise RuntimeError(
            "fused Gaussian gradient packing supports at most "
            f"{capacity} trainable input tensors, got {len(active_fields)}"
        )


def _rank_gradient_sizes(
    counts_by_layer: tuple[tuple[int, ...], ...],
    field_layers: tuple[int, ...],
    field_widths: tuple[int, ...],
    active_fields: tuple[int, ...],
) -> tuple[int, ...]:
    """Return each destination rank's unpadded gradient payload size."""
    _validate_gradient_pack_input_capacity(active_fields)
    world_size = len(counts_by_layer[0])
    if not all(len(counts) == world_size for counts in counts_by_layer):
        raise RuntimeError("Gaussian gradient count vectors must share one world size")
    return tuple(
        sum(counts_by_layer[field_layers[field]][rank] * field_widths[field] for field in active_fields)
        for rank in range(world_size)
    )


def _gradient_pack_plan_rows(
    counts_by_layer: tuple[tuple[int, ...], ...],
    field_layers: tuple[int, ...],
    field_widths: tuple[int, ...],
    active_fields: tuple[int, ...],
    sizes_by_rank: tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int, int, int], ...], int, int]:
    """Build an exact-cover plan for the padded ReduceScatter input.

    The output is destination-rank-major. Each equal-width rank chunk
    concatenates that rank's slice of every active field in forward field order,
    then zero-pads to ``max(sizes_by_rank)``. Copy rows cover the output exactly;
    ``input_index == -1`` denotes padding.
    """
    _validate_gradient_pack_input_capacity(active_fields)
    if not sizes_by_rank:
        raise ValueError("gradient packing requires at least one rank")

    values_per_block = int(gaussian_collective_payload_ops._GRADIENT_PACK_VALUES_PER_BLOCK)

    world_size = len(sizes_by_rank)
    max_size = max(sizes_by_rank)
    output_numel = world_size * max_size
    if output_numel <= 0:
        raise ValueError("nonempty gradient packing requires a positive output size")

    expected_source_sizes: list[int] = []
    for field in active_fields:
        layer = field_layers[field]
        counts = counts_by_layer[layer]
        if len(counts) != world_size:
            raise ValueError(f"gradient field {field} has counts for {len(counts)} ranks, expected {world_size}")
        expected_source_sizes.append(sum(counts) * field_widths[field])

    source_offsets = [0] * len(active_fields)
    copy_rows: list[tuple[int, int, int, int, int]] = []
    destination_offset = 0
    num_blocks = 0

    for rank, rank_size in enumerate(sizes_by_rank):
        rank_start = destination_offset
        if rank_size < 0 or rank_size > max_size:
            raise ValueError(f"gradient rank {rank} has invalid packed size {rank_size} for maximum {max_size}")
        for input_index, field in enumerate(active_fields):
            layer = field_layers[field]
            length = counts_by_layer[layer][rank] * field_widths[field]
            if length < 0:
                raise ValueError(f"gradient field {field} has negative packed length {length} on rank {rank}")

            source_offset = source_offsets[input_index]
            source_end = source_offset + length
            if source_end > expected_source_sizes[input_index]:
                raise ValueError(
                    f"gradient field {field} source ends at {source_end}, "
                    f"beyond expected size {expected_source_sizes[input_index]}"
                )
            if length:
                num_blocks += (length + values_per_block - 1) // values_per_block
                if num_blocks > torch.iinfo(torch.int32).max:
                    raise ValueError("gradient packing block count overflows int32")
                copy_rows.append(
                    (
                        destination_offset,
                        length,
                        input_index,
                        source_offset,
                        num_blocks,
                    )
                )
            destination_offset += length
            source_offsets[input_index] = source_end

        packed_rank_size = destination_offset - rank_start
        if packed_rank_size != rank_size:
            raise ValueError(f"gradient rank {rank} rows contain {packed_rank_size} values, expected {rank_size}")
        padding_size = max_size - rank_size
        if padding_size:
            num_blocks += (padding_size + values_per_block - 1) // values_per_block
            if num_blocks > torch.iinfo(torch.int32).max:
                raise ValueError("gradient packing block count overflows int32")
            copy_rows.append((destination_offset, padding_size, -1, 0, num_blocks))
            destination_offset += padding_size
        expected_destination = (rank + 1) * max_size
        if destination_offset != expected_destination:
            raise ValueError(
                f"gradient rank {rank} rows end at {destination_offset}, expected exact cover at {expected_destination}"
            )

    if source_offsets != expected_source_sizes:
        raise ValueError(f"gradient source rows cover {source_offsets}, expected {expected_source_sizes}")
    if destination_offset != output_numel:
        raise ValueError(f"gradient rows cover {destination_offset} values, expected output size {output_numel}")
    if num_blocks <= 0 or not copy_rows:
        raise ValueError("nonempty gradient packing requires positive blocks and copy rows")
    return tuple(copy_rows), output_numel, num_blocks


@functools.lru_cache(maxsize=16)
def _cached_gradient_pack_plan(
    counts_by_layer: tuple[tuple[int, ...], ...],
    field_layers: tuple[int, ...],
    field_widths: tuple[int, ...],
    active_fields: tuple[int, ...],
    device: torch.device,
    stream_id: int,
) -> _GradientPackPlan:
    """Upload and cache a gradient-pack plan for one topology and CUDA stream.

    The key must include every property that affects element offsets or launch
    geometry. If packing becomes dtype-, stride-, or alignment-dependent, add
    those properties to the key before reusing this cache.

    The upload occurs on the forward stream, and backward currently relies on
    PyTorch's same-stream autograd semantics. A future side-stream consumer must
    add an explicit CUDA event or wait: ``recordStream`` protects allocation
    lifetime, not execution ordering.

    The key contains the raw ``cudaStream_t``. Callers that destroy externally
    owned streams must clear this cache before CUDA can recycle the handle.
    """
    _validate_gradient_pack_input_capacity(active_fields)
    if device.type != "cuda":
        raise RuntimeError("Gaussian gradient metadata must be cached on CUDA")
    del stream_id
    with ScopedTimer("GaussiansComposite/prepare_gradient_pack_metadata"):
        sizes_by_rank = _rank_gradient_sizes(
            counts_by_layer,
            field_layers,
            field_widths,
            active_fields,
        )
        copy_rows, output_numel, num_blocks = _gradient_pack_plan_rows(
            counts_by_layer,
            field_layers,
            field_widths,
            active_fields,
            sizes_by_rank,
        )
        return _GradientPackPlan(
            copy_rows=torch.tensor(copy_rows, dtype=torch.int64, device=device),
            output_numel=output_numel,
            num_blocks=num_blocks,
        )


# Autograd orchestration and composite entry point


class _FusedGaussianParameterGather(torch.autograd.Function):
    """Own the rank-symmetric forward and backward collective sequence.

    ``forward`` first completes schema/count agreement, then chooses grouped or
    per-layer payload packing and gathering. It stores only topology needed to
    construct local gradients. ``backward`` packs all active fields into one
    destination-major buffer and performs exactly one ReduceScatter when the
    global gradient payload is nonempty.
    """

    @staticmethod
    def forward(
        ctx,
        layouts: tuple[_LayerPayloadLayout, ...],
        integer_range_validation_state: _PayloadIntegerRangeValidationState,
        world_rank: int,
        world_size: int,
        process_group: torch.distributed.ProcessGroup,
        prefetched_gather: Optional[_PrefetchedFusedGather],
        *local_fields: torch.Tensor,
    ) -> tuple[tuple[tuple[int, ...], ...] | torch.Tensor, ...]:
        assert 0 <= world_rank < world_size
        ctx.set_materialize_grads(False)

        # Reconstruct layer boundaries in the flat autograd input list. The
        # caller creates ``layouts`` and ``local_fields`` from the same ordered
        # dictionaries, so these slices are the authoritative field mapping.
        field_needs_grad = tuple(ctx.needs_input_grad[6:])
        prefetched_payload: Optional[torch.Tensor] = None
        generation_plan: Optional[_FusedGatherGenerationPlan] = None
        pending_payload: Optional[_PendingFusedPayloadGather] = None
        if isinstance(prefetched_gather, _PrefetchedFusedGather):
            generation_plan = prefetched_gather.plan
            pending_payload = prefetched_gather.payload
            with ScopedTimer("GaussiansComposite/wait_prefetched_grouped_payload"):
                prefetched_payload = _finish_fused_payload_gather(pending_payload)
            prefetched_counts: Optional[_ResolvedLayerCountGather] = generation_plan.resolved_counts
        else:
            prefetched_counts = None
        active_fields = tuple(field for field, needs_grad in enumerate(field_needs_grad) if needs_grad)
        schema_hash = (
            generation_plan.schema_hash
            if generation_plan is not None
            else _communication_schema_hash(layouts, field_needs_grad)
        )
        gathered_fields: list[torch.Tensor] = []
        per_layer_gathers_to_unpack: list[tuple[_LayerPayloadLayout, torch.Tensor, tuple[int, ...], int, int, int]] = []
        grouped_unpack_layouts: tuple[_LayerPayloadLayout, ...] = ()
        grouped_unpack_counts: tuple[tuple[int, ...], ...] = ()
        grouped_unpack_payload_layout: Optional[_GroupedPayloadLayout] = None
        grouped_gathered_payload: Optional[torch.Tensor] = None
        field_layers: list[int] = []
        field_widths: list[int] = []
        field_trailing_shapes: list[tuple[int, ...]] = []
        field_dtypes: list[torch.dtype] = []
        fields_by_layer: list[tuple[torch.Tensor, ...]] = []
        field_offset = 0
        for layout in layouts:
            next_field_offset = field_offset + len(layout.keys)
            layer_fields = local_fields[field_offset:next_field_offset]
            assert len(layer_fields) == len(layout.keys)
            fields_by_layer.append(layer_fields)
            field_offset = next_field_offset
        assert field_offset == len(local_fields)
        if not fields_by_layer:
            raise RuntimeError("coalesced Gaussian count gathering requires at least one layer")
        fields_by_layer_tuple = tuple(fields_by_layer)
        if generation_plan is None:
            integer_range_validation_state.observe(
                layouts,
                fields_by_layer_tuple,
                schema_hash,
                world_rank,
                world_size,
                process_group,
            )

        # Control plane: establish identical collective topology everywhere
        # before any rank allocates or launches a variable-sized payload.
        current_local_counts = tuple(layer_fields[0].shape[0] for layer_fields in fields_by_layer)
        current_device = fields_by_layer[0][0].device
        if prefetched_counts is None:
            with ScopedTimer("GaussiansComposite/gather_parameter_counts_fused_layers"):
                counts_by_layer = _all_gather_layer_counts(
                    world_size,
                    current_local_counts,
                    current_device,
                    schema_hash,
                    integer_range_validation_state,
                    process_group,
                )
        else:
            if (prefetched_counts.world_rank, prefetched_counts.world_size) != (
                world_rank,
                world_size,
            ):
                raise RuntimeError(
                    "Prefetched Gaussian process-group coordinates do not match the payload gather: "
                    f"{(prefetched_counts.world_rank, prefetched_counts.world_size)} != "
                    f"{(world_rank, world_size)}"
                )
            if prefetched_counts.local_counts != current_local_counts:
                raise RuntimeError(
                    "Gaussian row counts changed between cached generation resolution and payload gather: "
                    f"{prefetched_counts.local_counts} != {current_local_counts}"
                )
            if prefetched_counts.device != current_device:
                raise RuntimeError(
                    "Prefetched Gaussian count device does not match the payload device: "
                    f"{prefetched_counts.device} != {current_device}"
                )
            counts_by_layer = prefetched_counts.counts_by_layer
            if len(counts_by_layer) != len(layouts) or any(len(counts) != world_size for counts in counts_by_layer):
                raise RuntimeError("Prefetched Gaussian counts do not match the payload gather topology")
        if generation_plan is None:
            _validate_fused_schema(layouts, field_needs_grad)

        if generation_plan is None:
            for layout, layer_fields, counts in zip(layouts, fields_by_layer_tuple, counts_by_layer, strict=True):
                local_count = layer_fields[0].shape[0]
                if counts[world_rank] != local_count:
                    raise RuntimeError(
                        f"Gathered count mismatch for {layout.node_id}: rank {world_rank} count "
                        f"{counts[world_rank]} != local count {local_count}"
                    )

        for layer, layout in enumerate(layouts):
            field_layers.extend([layer] * len(layout.keys))
            field_widths.extend(layout.widths)
            field_trailing_shapes.extend(layout.trailing_shapes)
            field_dtypes.extend(layout.dtypes)

        # Data plane: compatible layers share one grouped AllGather. The
        # fallback keeps one AllGather per layer when payload dtypes/devices or
        # the CUDA-exported layer capacity prevent grouping.
        can_group_payloads = generation_plan is not None
        if generation_plan is None:
            devices_by_layer = tuple(
                tuple(field.device for field in layer_fields) for layer_fields in fields_by_layer_tuple
            )
            can_group_payloads = _can_group_payloads(layouts, devices_by_layer)
            # Integer fields share each layer's floating wire dtype. The fixed
            # control packet reports whether every rank has validated this exact
            # weak identity + version generation. A miss anywhere makes all
            # direct ranks run the value-dependent check before payload packing.
            # Published generation plans use their own structural validation and
            # never touch this direct-route cache after their payload is launched.
            if not integer_range_validation_state.all_ranks_cached:
                _validate_payload_integer_ranges(
                    layouts,
                    fields_by_layer_tuple,
                    process_group,
                )
                integer_range_validation_state.mark_validated()
        if can_group_payloads:
            grouped_layouts = layouts
            grouped_fields = fields_by_layer_tuple
            grouped_counts = counts_by_layer
            grouped_payload_layout = (
                generation_plan.grouped_payload_layout
                if generation_plan is not None
                else _make_grouped_payload_layout(grouped_layouts, grouped_counts)
            )
            if pending_payload is not None:
                if prefetched_payload is None:
                    raise RuntimeError("Prefetched grouped Gaussian payload has no gathered storage")
                grouped_gathered_payload = prefetched_payload
            else:
                with ScopedTimer("GaussiansComposite/gather_parameters[grouped_layers]"):
                    local_grouped_payload = torch.empty(
                        grouped_payload_layout.rank_stride,
                        dtype=grouped_layouts[0].payload_dtype,
                        device=grouped_fields[0][0].device,
                    )
                    flattened_grouped_fields = _flatten_grouped_payload_fields(grouped_layouts, grouped_fields)
                    gaussian_collective_payload_ops._pack_grouped_payload_trusted(
                        local_grouped_payload,
                        flattened_grouped_fields,
                        tuple(layout.widths for layout in grouped_layouts),
                        tuple(layer_fields[0].shape[0] for layer_fields in grouped_fields),
                        tuple(max(counts) for counts in grouped_counts),
                    )
                    grouped_gathered_payload = _all_gather_grouped_payload(
                        world_size,
                        local_grouped_payload,
                        process_group,
                    )
                    del local_grouped_payload

            grouped_unpack_layouts = grouped_layouts
            grouped_unpack_counts = grouped_counts
            grouped_unpack_payload_layout = grouped_payload_layout
        else:
            for layer, (layout, layer_fields) in enumerate(zip(layouts, fields_by_layer_tuple, strict=True)):
                with ScopedTimer(f"GaussiansComposite/gather_parameters[{layout.node_id}]"):
                    counts = counts_by_layer[layer]
                    max_count = max(counts)
                    payload = _pack_layer_payload(layout, layer_fields, max_count)
                    assert payload.shape == (max_count, sum(layout.widths))
                    gathered = _all_gather_padded_payload(
                        world_size,
                        payload,
                        process_group,
                    )
                    del payload
                    per_layer_gathers_to_unpack.append(
                        (
                            layout,
                            gathered,
                            counts,
                            max_count,
                            max_count * sum(layout.widths),
                            0,
                        )
                    )

        # Keep the established count/payload sequence free of unpack
        # kernels. Unpack only after every payload gather, at the existing
        # gather-to-collector boundary, and share these contiguous fields
        # between camera and lidar.
        while per_layer_gathers_to_unpack:
            layout, gathered, counts, max_count, rank_stride, layer_offset = per_layer_gathers_to_unpack.pop(0)
            unpacked = _unpack_layer_payload(
                layout,
                gathered,
                counts,
                max_count,
                rank_stride=rank_stride,
                layer_offset=layer_offset,
            )
            gathered_fields.extend(unpacked[key] for key in layout.keys)
            del gathered

        if grouped_gathered_payload is not None:
            assert grouped_unpack_payload_layout is not None
            unpacked_grouped_layers = _unpack_grouped_layer_payloads(
                grouped_unpack_layouts,
                grouped_gathered_payload,
                grouped_unpack_counts,
                grouped_unpack_payload_layout,
            )
            for layout, unpacked in zip(grouped_unpack_layouts, unpacked_grouped_layers, strict=True):
                gathered_fields.extend(unpacked[key] for key in layout.keys)
            # The grouped extension records the shared input allocation once;
            # release the final Python reference immediately after launch.
            del grouped_gathered_payload
        assert len(gathered_fields) == len(local_fields)

        ctx.active_fields = active_fields
        ctx.counts_by_layer = counts_by_layer
        ctx.field_layers = tuple(field_layers)
        ctx.field_widths = tuple(field_widths)
        ctx.field_trailing_shapes = tuple(field_trailing_shapes)
        ctx.field_dtypes = tuple(field_dtypes)
        ctx.world_rank = world_rank
        ctx.process_group = process_group
        # Prepare immutable, stream-specific CUDA copy metadata during forward
        # so backward only supplies the gradients produced by autograd.
        if active_fields:
            ctx.gradient_pack_plan = None
            first_active_field = active_fields[0]
            first_active_layer = ctx.field_layers[first_active_field]
            ctx.gradient_dtype = layouts[first_active_layer].payload_dtype
            ctx.gradient_device = local_fields[first_active_field].device
            ctx.sizes_by_rank = _rank_gradient_sizes(
                ctx.counts_by_layer,
                ctx.field_layers,
                ctx.field_widths,
                active_fields,
            )
            if max(ctx.sizes_by_rank) > 0:
                if ctx.gradient_device.type != "cuda":
                    raise RuntimeError("fused Gaussian gradient packing requires CUDA tensors")
                if ctx.gradient_dtype not in _CUDA_FUSED_GATHER_DTYPES:
                    raise RuntimeError(f"fused Gaussian gradient packing does not support {ctx.gradient_dtype}")
                ctx.gradient_pack_plan = _cached_gradient_pack_plan(
                    ctx.counts_by_layer,
                    ctx.field_layers,
                    ctx.field_widths,
                    active_fields,
                    ctx.gradient_device,
                    torch.cuda.current_stream(ctx.gradient_device).cuda_stream,
                )

        # Concatenating a layer's fields makes every floating split output
        # differentiable when any field requires gradients. Consuming a frozen
        # sibling must therefore return explicit zeros for trainable fields.
        active_layers = {field_layers[field] for field in active_fields}
        non_differentiable = [
            output
            for index, output in enumerate(gathered_fields)
            if field_layers[index] not in active_layers or not output.dtype.is_floating_point
        ]
        if non_differentiable:
            ctx.mark_non_differentiable(*non_differentiable)
        return (counts_by_layer, *gathered_fields)

    @staticmethod
    def backward(ctx, *all_grad_outputs: Any):
        assert all_grad_outputs[0] is None
        grad_outputs = cast(tuple[Optional[torch.Tensor], ...], all_grad_outputs[1:])
        input_count = len(ctx.field_layers)
        assert len(grad_outputs) == input_count
        input_gradients: list[Optional[torch.Tensor]] = [None] * input_count
        active_fields: tuple[int, ...] = ctx.active_fields
        if not active_fields:
            return None, None, None, None, None, None, *input_gradients

        # The CUDA out-parameter packer intentionally has no autograd formula.
        # Reject higher-order differentiation instead of silently switching to
        # a Python implementation with a different performance contract.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "fused Gaussian gradient packing supports first-order backward only; create_graph=True is not supported"
            )

        # A layer absent from the local autograd graph returns None gradients;
        # an unused field in an active layer returns explicit zeros. Preserve
        # that distinction while packing every field into one collective.
        used_layers = {ctx.field_layers[field] for field, gradient in enumerate(grad_outputs) if gradient is not None}
        sizes_by_rank: tuple[int, ...] = ctx.sizes_by_rank
        max_size = max(sizes_by_rank)
        if max_size == 0:
            for field in active_fields:
                layer = ctx.field_layers[field]
                if layer in used_layers:
                    shape = (ctx.counts_by_layer[layer][ctx.world_rank], *ctx.field_trailing_shapes[field])
                    input_gradients[field] = torch.empty(
                        shape,
                        dtype=ctx.field_dtypes[field],
                        device=ctx.gradient_device,
                    )
            return None, None, None, None, None, None, *input_gradients

        with ScopedTimer("GaussiansComposite/pack_gradients_fused_layers"):
            plan = ctx.gradient_pack_plan
            if not isinstance(plan, _GradientPackPlan):
                raise RuntimeError("fused Gaussian CUDA gradient-pack plan is unavailable")
            with ScopedTimer("GaussiansComposite/flatten_gradient_inputs"):
                flat_gradients: list[Optional[torch.Tensor]] = []
                for field in active_fields:
                    gradient = grad_outputs[field]
                    if gradient is None:
                        flat_gradients.append(None)
                        continue
                    layer = ctx.field_layers[field]
                    expected_numel = sum(ctx.counts_by_layer[layer]) * ctx.field_widths[field]
                    if gradient.numel() != expected_numel:
                        raise RuntimeError(
                            f"Gradient field {field} has {gradient.numel()} values, expected {expected_numel}"
                        )
                    # The CUDA packer consumes each input as a flat contiguous
                    # span and already uses ``numel`` rather than tensor
                    # dimensions.  Production gradients normally have the
                    # common payload dtype and contiguous row-major storage,
                    # so pass those tensors through directly instead of
                    # creating a flattened view and dispatching two no-op
                    # conversions for every field on every backward.
                    if gradient.dtype == ctx.gradient_dtype and gradient.is_contiguous():
                        flat_gradients.append(gradient)
                    else:
                        flat_gradients.append(gradient.reshape(-1).to(ctx.gradient_dtype).contiguous())
            with ScopedTimer("GaussiansComposite/allocate_packed_gradients"):
                packed_gradients = torch.empty(
                    plan.output_numel,
                    dtype=ctx.gradient_dtype,
                    device=ctx.gradient_device,
                )
            with ScopedTimer("GaussiansComposite/pack_gradients_cuda"):
                gaussian_collective_payload_ops._pack_gradients_trusted(
                    packed_gradients,
                    flat_gradients,
                    plan.copy_rows,
                    plan.num_blocks,
                )

        with ScopedTimer("GaussiansComposite/reduce_scatter_gradients_fused_layers"):
            # The packed buffer is destination-major: chunk ``r`` contains the
            # contributions rank ``r`` must receive for all active fields.
            local_padded_gradient = packed_gradients.new_empty(max_size)
            torch.distributed.reduce_scatter_tensor(
                local_padded_gradient,
                packed_gradients,
                op=torch.distributed.ReduceOp.SUM,
                group=ctx.process_group,
            )

        offset = 0
        for field in active_fields:
            layer = ctx.field_layers[field]
            local_count = ctx.counts_by_layer[layer][ctx.world_rank]
            numel = local_count * ctx.field_widths[field]
            if layer in used_layers:
                input_gradients[field] = (
                    local_padded_gradient.narrow(0, offset, numel)
                    .reshape(local_count, *ctx.field_trailing_shapes[field])
                    .to(ctx.field_dtypes[field])
                )
            offset += numel
        assert offset == sizes_by_rank[ctx.world_rank]
        return None, None, None, None, None, None, *input_gradients


def _fused_all_gather_gaussian_parameters(
    gaussian_nodes: list[str],
    parameters_by_node: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    process_group: Optional[torch.distributed.ProcessGroup],
    integer_range_validation_state: Optional[_PayloadIntegerRangeValidationState] = None,
    prepared_inputs: Optional[_PreparedFusedGatherInputs] = None,
    generation_plan: Optional[_FusedGatherGenerationPlan] = None,
    prefetched_payload_gather: Optional[_PendingFusedPayloadGather] = None,
) -> _FusedGatherResult:
    """Fuse production forward payloads and CUDA/NCCL backward.

    This is a private optimization for ``GaussiansComposite``'s collector
    path, not a general distributed gather API. Unsupported distributed
    schemas raise instead of silently selecting a Python path. A
    single exact reduction requires every trainable layer payload to share a
    floating dtype and CUDA device. As with DDP itself, field order, dtype,
    device, and ``requires_grad`` must describe the same replicated model
    schema on every rank. Whole-layer participation in the autograd graph must
    also be rank-symmetric, matching the established per-layer gather path;
    field use may differ when a sibling keeps the layer active on every rank.
    A fixed-size metadata packet validates layer arity, carries a hash of the
    communication schema, and includes common-case layer counts. Every rank
    must run the same binary and fixed packet width; the packet version does
    not make mixed-width implementations interoperable. Layer
    selections above the inline capacity use a second full count-vector
    AllGather. Divergent layer/field order, shapes, dtypes, or trainability fail
    before payload gathering.
    Direct callers validate a fresh fixed metadata packet at the collector
    boundary. The exact built-in training route may instead provide a
    generation plan refreshed after structural mutations plus an already
    launched payload. That fast path still captures current tensor references
    and validates them against the published schema before launch on every
    forward. It deliberately omits post-launch mutation-version revalidation because a rank-local error
    after peers enter NCCL could strand the process group. It performs no
    steady-forward metadata or validity collective.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError("fused Gaussian gathering requires an initialized multi-rank process group")
    if process_group is None:
        raise RuntimeError("fused Gaussian gathering requires the dedicated process group")
    world_size = torch.distributed.get_world_size(process_group)
    if world_size <= 1:
        raise RuntimeError("fused Gaussian gathering requires an initialized multi-rank process group")
    if not gaussian_nodes or len(set(gaussian_nodes)) != len(gaussian_nodes):
        raise RuntimeError("fused Gaussian gathering requires at least one distinct Gaussian node")

    if prepared_inputs is None:
        prepared = _prepare_fused_gather_inputs(
            gaussian_nodes,
            parameters_by_node,
            process_group=process_group,
        )
    else:
        prepared = prepared_inputs
    prepared_node_order = tuple(layout.node_id for layout in prepared.layouts)
    if prepared_node_order != tuple(gaussian_nodes):
        if prefetched_payload_gather is not None:
            _discard_fused_payload_gather(prefetched_payload_gather)
        raise RuntimeError(
            "Prepared Gaussian node order does not match the requested gather: "
            f"{prepared_node_order} != {tuple(gaussian_nodes)}"
        )

    gradient_device = prepared.device
    if gradient_device.type != "cuda":
        raise RuntimeError("fused Gaussian gradient packing requires CUDA tensors")
    if torch.distributed.get_backend(process_group) != torch.distributed.Backend.NCCL:
        raise RuntimeError("fused Gaussian gradient packing requires the NCCL backend")

    if integer_range_validation_state is None:
        # Direct callers retain the historical self-contained behavior.  The
        # production composite supplies a persistent per-instance state so
        # unchanged topology generations take the scan-free warm path.
        integer_range_validation_state = _PayloadIntegerRangeValidationState()
    if (generation_plan is None) != (prefetched_payload_gather is None):
        if prefetched_payload_gather is not None:
            _discard_fused_payload_gather(prefetched_payload_gather)
        raise RuntimeError("A prefetched Gaussian payload and generation plan must be supplied together")
    if generation_plan is not None:
        if prepared_inputs is None:
            _discard_fused_payload_gather(cast(_PendingFusedPayloadGather, prefetched_payload_gather))
            raise RuntimeError("A Gaussian generation plan requires explicitly captured fresh inputs")
        if not _generation_plan_matches_prepared(generation_plan, prepared):
            _discard_fused_payload_gather(cast(_PendingFusedPayloadGather, prefetched_payload_gather))
            raise RuntimeError("Fresh Gaussian tensors do not match the published generation plan")
        # FreePose has no Gaussian reference and cannot mutate this snapshot.
        # Do not add a post-launch rank-local validation branch: under the
        # exact schedule, topology is owned by generation refresh/invalidation;
        # an out-of-contract mismatch must terminate the distributed run.

    prefetched_gather: Optional[_PrefetchedFusedGather] = None
    if generation_plan is not None:
        prefetched_gather = _PrefetchedFusedGather(
            plan=generation_plan,
            payload=cast(_PendingFusedPayloadGather, prefetched_payload_gather),
        )

    forward_outputs = _FusedGaussianParameterGather.apply(
        prepared.layouts,
        integer_range_validation_state,
        prepared.world_rank,
        prepared.world_size,
        process_group,
        prefetched_gather,
        *prepared.local_fields,
    )
    counts_by_layer = cast(tuple[tuple[int, ...], ...], forward_outputs[0])
    gathered_fields = cast(tuple[torch.Tensor, ...], forward_outputs[1:])
    assert len(gathered_fields) == len(prepared.local_fields)
    result: dict[str, dict[str, torch.Tensor]] = {}
    counts_by_node: dict[str, tuple[int, ...]] = {}
    field_offset = 0
    for layer, layout in enumerate(prepared.layouts):
        next_field_offset = field_offset + len(layout.keys)
        result[layout.node_id] = dict(zip(layout.keys, gathered_fields[field_offset:next_field_offset], strict=True))
        counts_by_node[layout.node_id] = counts_by_layer[layer]
        field_offset = next_field_offset
    assert field_offset == len(gathered_fields)
    return _FusedGatherResult(parameters_by_node=result, counts_by_node=counts_by_node)
