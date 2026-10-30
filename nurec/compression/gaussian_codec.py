# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pack and unpack compact Gaussian scene tensors.

Use pack() on a model state dict and unpack() with the returned metadata to
restore it. State keys must use the "model." prefix. unpack() restores skipped
tensors as zeros. Non-tensor state must be JSON-serializable.

    >>> tensors, metadata = pack(state_dict)
    >>> restored = unpack(tensors, metadata)
"""

import json
import logging
import math
import numbers
import typing

import torch

from nurec.utils.morton import VOXEL_SIZE, morton_sort_order


__all__ = ["EXTRA_SIGNAL_TENSOR_NAMES", "FORMAT_VALUE", "pack", "unpack"]

_logger = logging.getLogger(__name__)

_PagePaddingPlan = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_PositionMode = typing.Literal["fp16_offset", "fp32_sorted"]


class _Level(typing.NamedTuple):
    sh_bits: int | None
    albedo_bits: int | None
    rotation_bits: int
    density_bits: int
    scale_bits: int
    position_mode: _PositionMode


# Bump this when key names, sidecars, SH layout, or quantizers change.
# unpack rejects any other value.
_FORMAT_KEY = "compact_format"
FORMAT_VALUE = "nurec.gaussians.compact/v1"

_METADATA_SKIPPED_TENSORS = "_skipped_tensors"
_METADATA_NON_TENSOR_STATE = "_non_tensor_state"
_METADATA_PRESERVED_EXTRA_SIGNAL_TENSORS = "_preserved_extra_signal_tensors"

_MODEL_PREFIX = "model."


_LAYER_PREFIXES = (
    "gaussians_nodes.background",
    "gaussians_nodes.road",
    "gaussians_nodes.dynamic_rigids",
    "gaussians_nodes.dynamic_deformables",
)

_DEFORMABLE_PREFIXES = (
    "gaussians_nodes.dynamic_rigids",
    "gaussians_nodes.dynamic_deformables",
)

# Skip training-only tensors. They do not belong in storage artifacts.
_SKIP_PATTERNS = (
    "invisible_steps",
    "texture_grads",
    "n_grad_updates",
)

EXTRA_SIGNAL_TENSOR_NAMES = (
    "extra_signal",
    "camera_extra_signal",
    "lidar_extra_signal",
)

# Extra-signal values are training/debug outputs. Compact artifacts keep
# their shapes only; unpack() zero-fills them so model loading remains valid.
_DROPPED_SIGNAL_TENSOR_NAMES = frozenset(EXTRA_SIGNAL_TENSOR_NAMES)

# Match whole path segments. Substring matches would catch unrelated keys.
_FP16_PATTERNS = frozenset(
    ("textures", "bilateral_grid", "features_albedo", "deform_network"),
)


# The three stored quaternion components are bounded by 1/sqrt(2).
_SQRT2_INV = 1.0 / 2.0**0.5

# Use the symmetric [-127, 127] range so zero stays centered and negation is safe.
_INT8_SYMMETRIC_MAX = 127.0

# Use the symmetric [-32767, 32767] range so zero stays centered and negation is safe.
_INT16_SYMMETRIC_MAX = 32767.0

# Max uint8 value for affine-normalized columns.
_UINT8_AFFINE_MAX = 255.0

# Max uint16 value for affine-normalized columns.
_UINT16_AFFINE_MAX = 65535.0

# Map smallest-three rotations into the symmetric int8 range.
_ROT_QUANT_SCALE = _INT8_SYMMETRIC_MAX / _SQRT2_INV

# Map smallest-three rotations into the symmetric int16 range.
_ROT16_QUANT_SCALE = _INT16_SYMMETRIC_MAX / _SQRT2_INV

# Per-column fit-sample cap. Paged Lloyd sorts `(page_len, page_count * cols)`;
# page_len dominates sort cost. 2053 samples are plenty for k=256 centroids.
# A prime-ish cap slightly reduces the chance that strided sampling repeatedly
# hits the same phase of periodic data.
_LLOYD_FIT_BUDGET = 2_053

# Constant columns map to 0 instead of NaN.
_EPS = 1e-8

# features_specular has 45 columns: (3 + 5 + 7 SH coeffs) * RGB.
# Joint PCA beat per-band PCA by 2 to 14 dB at the same byte count.
_SH_TOTAL_COLS = 45

_LAYER_TENSOR_SHAPES: dict[str, tuple[object, ...]] = {
    "positions": (3,),
    "densities": (1,),
    "scales": (3,),
    "rotations": (4,),
    "features_specular": (_SH_TOTAL_COLS,),
    "features_albedo": (..., 3),
}
# Add new row-aligned Gaussian tensors here, or culling and Morton sorting will
# not keep them aligned with positions.
_PER_GAUSSIAN_TENSOR_NAMES = (
    frozenset(_LAYER_TENSOR_SHAPES)
    | _DROPPED_SIGNAL_TENSOR_NAMES
    | {
        "gaussian_cuboid_ids",
    }
)

# Levels follow the measured PSNR, SSIM, LPIPS Pareto curve. SH is bits per
# Gaussian. Albedo is bits per RGB triple. None means fp16 passthrough.
_LEVELS: dict[int, _Level] = {
    0: _Level(None, None, 16, 16, 16, "fp32_sorted"),
    1: _Level(360, None, 16, 16, 16, "fp32_sorted"),
    2: _Level(236, None, 16, 16, 16, "fp32_sorted"),
    3: _Level(236, 24, 16, 16, 16, "fp16_offset"),
    4: _Level(168, 24, 16, 16, 16, "fp16_offset"),
    5: _Level(140, 24, 16, 16, 16, "fp16_offset"),
    6: _Level(92, 24, 16, 16, 16, "fp16_offset"),
    7: _Level(92, 16, 16, 16, 16, "fp16_offset"),
    8: _Level(60, 16, 16, 8, 8, "fp16_offset"),
    9: _Level(36, 16, 8, 8, 8, "fp16_offset"),
}

# Tuned per-layer page policy. Static layers preserve quality with more
# leaves; dynamic layers use larger pages to keep sidecars small.
_DEFAULT_PAGE_MIN_ELEMENTS: dict[str, int] = {
    "background": 256,
    "road": 256,
    "dynamic_rigids": 2048,
    "dynamic_deformables": 2048,
}
_DEFAULT_PAGE_MAX_COUNT: dict[str, int] = {
    "background": 200,
    "road": 200,
    "dynamic_rigids": 100,
    "dynamic_deformables": 100,
}


def pack(
    state_dict: dict[str, typing.Any],
    opacity_threshold: float = 0.005,
    compression_level: int = 9,
    preserve_extra_signal_tensors: typing.Sequence[str] = (),
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Compact a Gaussian scene state dict.

    Known Gaussian rows are Morton-sorted. unpack() restores a render-ready
    state in that compact row order, not the original checkpoint order.

    Args:
        state_dict: Model state keyed by "model."-prefixed names. Tensor values
            are cloned before modification. Non-tensor values must be
            JSON-serializable.
        opacity_threshold: Drop Gaussians with sigmoid density below this value.
            Must be in [0, 1). Defaults to 0.005.
        compression_level: Storage level in [0, 9]. Defaults to 9. Higher
            levels use smaller position, appearance, and geometry budgets.
        preserve_extra_signal_tensors: Per-Gaussian `*_extra_signal` tensor leaf
            names to keep. Values are whole tensor leaves such as
            "lidar_extra_signal", not individual configured signal names such as
            "intensity" or "raydrop". By default all extra-signal values are
            dropped and zero-filled during unpack.

    Returns:
        Packed tensors and metadata required by unpack().

    Raises:
        ValueError: If an argument or state-dict entry violates the codec
            contract.
    """
    if type(compression_level) is not int or compression_level not in _LEVELS:
        raise ValueError(f"compression_level must be an int in [0, 9], got {compression_level!r}")
    level = _LEVELS[compression_level]
    preserved_signals = _validated_preserved_extra_signal_tensors(preserve_extra_signal_tensors)

    return _pack(
        state_dict,
        opacity_threshold=opacity_threshold,
        level=level,
        preserve_extra_signal_tensors=preserved_signals,
    )


def unpack(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> dict[str, typing.Any]:
    """Restore a state dict from compact tensors.

    Returned floating-point tensors use the input tensors' device. Metadata
    records skipped tensors, which unpack() restores as zeros. Known Gaussian
    rows stay in Morton order.

    Args:
        tensors: Packed tensors returned by pack(). The input mapping is copied
            before it is consumed.
        metadata: Metadata returned by pack().

    Returns:
        Restored state dict keyed by "model."-prefixed names.

    Raises:
        ValueError: If the format marker is missing, unsupported, corrupt, or
            if reconstructed keys collide.
    """
    got = metadata.get(_FORMAT_KEY)
    if got != FORMAT_VALUE:
        raise ValueError(
            f"unsupported compact format: {got!r}, expected {FORMAT_VALUE!r}",
        )

    tensors = dict(tensors)

    _unmorton_layers(tensors)
    _dequantize_geometry(tensors)
    _unpack_rotations(tensors)
    _unpack_albedo(tensors)
    _unpack_sh(tensors)

    # Pop tensors before writing fp32 replacements to lower peak memory.
    device = next((v.device for v in tensors.values()), torch.device("cpu"))
    state_dict: dict[str, typing.Any] = {}
    for k in list(tensors):
        v = tensors.pop(k)
        _put_state(state_dict, k, v.float() if v.is_floating_point() else v)

    # Forward paths reference some skipped tensors.
    # Zero-fill them so state loading remains valid.
    for key, (shape, dtype_str) in json.loads(
        metadata.get(_METADATA_SKIPPED_TENSORS, "{}"),
    ).items():
        dtype = getattr(torch, dtype_str, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unknown dtype in metadata for {key!r}: {dtype_str!r}")
        _put_state(state_dict, key, torch.zeros(shape, dtype=dtype, device=device))

    for key, value in json.loads(
        metadata.get(_METADATA_NON_TENSOR_STATE, "{}"),
    ).items():
        _put_state(state_dict, key, value)

    return state_dict


def _pack(
    state_dict: dict[str, typing.Any],
    *,
    opacity_threshold: float = 0.005,
    level: _Level,
    preserve_extra_signal_tensors: frozenset[str] = frozenset(),
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Compact using one measured compression level."""
    if isinstance(opacity_threshold, bool) or not isinstance(opacity_threshold, numbers.Real):
        raise ValueError(f"opacity_threshold must be a real number in [0, 1), got {opacity_threshold!r}")
    if not 0.0 <= opacity_threshold < 1.0:
        raise ValueError(f"opacity_threshold must be in [0, 1), got {opacity_threshold}")

    tensors, skipped_tensors, non_tensor_state = _strip_state_dict(
        state_dict,
        preserve_extra_signal_tensors=preserve_extra_signal_tensors,
    )
    table_keys = sorted(key for key in (*tensors, *skipped_tensors, *non_tensor_state) if ".deformation_table." in key)
    if table_keys:
        raise ValueError(
            "compact Gaussian export does not support interpolated deformation tables; "
            f"found table state such as {table_keys[0]!r}"
        )
    _validate_layer_contract(tensors, skipped_tensors)
    kept_bytes = sum(t.nbytes for t in tensors.values())

    if opacity_threshold > 0.0:
        total_culled = sum(
            _cull_opacity(tensors, skipped_tensors, prefix, opacity_threshold) for prefix in _LAYER_PREFIXES
        )
        if total_culled:
            culled_bytes = kept_bytes - sum(t.nbytes for t in tensors.values())
            _logger.info(
                "  culled %d Gaussians total (%.1f MiB)",
                total_culled,
                culled_bytes / 2**20,
            )

    for prefix in _DEFORMABLE_PREFIXES:
        pos_key = f"{prefix}.positions"
        has_layer_keys = _layer_has_key(prefix, tensors, skipped_tensors)
        has_real_positions = pos_key in tensors and tensors[pos_key].shape[0] > 0

        if has_real_positions or not has_layer_keys:
            continue
        # Empty positions mean a fully culled layer.
        # Missing positions are invalid; row alignment is unknown.
        if pos_key not in tensors and pos_key not in skipped_tensors:
            raise ValueError(f"malformed input: {prefix!r} has layer tensors but no {pos_key!r}")
        _strip_deform_network(prefix, tensors, skipped_tensors, non_tensor_state)

    # Empty layers cannot pass through codecs that reduce over dim 0.
    # Store their shapes in metadata and zero-fill on unpack.
    for prefix in _LAYER_PREFIXES:
        _drop_empty_layer(tensors, skipped_tensors, prefix)

    for prefix in _LAYER_PREFIXES:
        _morton_sort_layer(tensors, prefix, position_mode=level.position_mode)

    _fp16_auxiliary(tensors, skip_albedo=level.albedo_bits is not None)
    _quantize_geometry(tensors, density_bits=level.density_bits, scale_bits=level.scale_bits)
    _pack_rotations(tensors, rotation_bits=level.rotation_bits)

    if level.sh_bits is not None:
        _quantize_sh(tensors, level.sh_bits)
    else:
        _fp16_specular(tensors)

    if level.albedo_bits is not None:
        _quantize_albedo(tensors, level.albedo_bits)

    metadata = {
        _FORMAT_KEY: FORMAT_VALUE,
        _METADATA_SKIPPED_TENSORS: json.dumps(skipped_tensors),
        _METADATA_NON_TENSOR_STATE: json.dumps(non_tensor_state),
        _METADATA_PRESERVED_EXTRA_SIGNAL_TENSORS: json.dumps(sorted(preserve_extra_signal_tensors)),
    }

    return tensors, metadata


def _validated_preserved_extra_signal_tensors(signal_tensors: typing.Sequence[str]) -> frozenset[str]:
    """Return validated signal tensor leaf names."""
    out = frozenset(signal_tensors)
    unknown = sorted(out - _DROPPED_SIGNAL_TENSOR_NAMES)
    if unknown:
        raise ValueError(
            f"preserve_extra_signal_tensors entries must be one of {sorted(_DROPPED_SIGNAL_TENSOR_NAMES)}, got {unknown}"
        )
    return out


def _put_state(
    state_dict: dict[str, typing.Any],
    key: str,
    value: typing.Any,
) -> None:
    """Insert one stripped state key. Raises if the full key already exists."""
    full_key = _MODEL_PREFIX + key
    if full_key in state_dict:
        raise ValueError(f"duplicate reconstructed state key: {full_key!r}")
    state_dict[full_key] = value


def _validate_geometry_bits(name: str, bits: int) -> None:
    """Validate one fixed-width geometry setting."""
    if type(bits) is not int or bits not in (8, 16):
        raise ValueError(f"{name} must be 8 or 16, got {bits!r}")


def _validate_position_mode(mode: str) -> None:
    """Validate one position storage mode."""
    if mode not in typing.get_args(_PositionMode):
        raise ValueError(f"position_mode must be 'fp16_offset' or 'fp32_sorted', got {mode!r}")


def _strip_state_dict(
    state_dict: dict[str, typing.Any],
    *,
    preserve_extra_signal_tensors: frozenset[str] = frozenset(),
) -> tuple[dict[str, torch.Tensor], dict[str, list], dict[str, typing.Any]]:
    """Split model state into packable tensors and metadata records.

    Returned keys have the "model." prefix stripped. Skipped tensors become
    [shape, dtype] records. JSON round-tripping validates non-tensor values and
    detaches references.

    Args:
        state_dict: State dict keyed by "model."-prefixed names.

    Returns:
        Packable tensors, skipped tensor records, and non-tensor state.

    Raises:
        ValueError: If a key lacks the "model." prefix or non-tensor state is
            not JSON-serializable.
    """
    tensors: dict[str, torch.Tensor] = {}
    skipped_tensors: dict[str, list] = {}
    non_tensor_state: dict[str, typing.Any] = {}
    skipped_bytes = 0
    skipped_counts: dict[str, int] = {}

    for key, value in state_dict.items():
        if not key.startswith(_MODEL_PREFIX):
            raise ValueError(f"expected {_MODEL_PREFIX!r}-prefixed key, got {key!r}")
        short_key = key.removeprefix(_MODEL_PREFIX)

        if not isinstance(value, torch.Tensor):
            try:
                non_tensor_state[short_key] = json.loads(json.dumps(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-JSON non-tensor state at {key!r}: {exc}") from exc
            continue

        reason = _should_skip(short_key, value, preserve_extra_signal_tensors=preserve_extra_signal_tensors)
        if reason:
            skipped_bytes += value.nbytes
            skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
            skipped_tensors[short_key] = [
                list(value.shape),
                _dtype_str(value.dtype),
            ]
            continue

        tensors[short_key] = value.clone()

    kept_mib = sum(t.nbytes for t in tensors.values()) / 2**20
    _logger.info("checkpoint: kept %d tensors (%.1f MiB)", len(tensors), kept_mib)
    for reason, count in sorted(skipped_counts.items()):
        _logger.debug("  skipped %d tensors matching '%s'", count, reason)
    _logger.info("  skipped total: %.1f MiB", skipped_bytes / 2**20)
    if non_tensor_state:
        _logger.info("  preserved %d non-tensor state entries", len(non_tensor_state))

    return tensors, skipped_tensors, non_tensor_state


def _should_skip(
    key: str,
    tensor: torch.Tensor,
    *,
    preserve_extra_signal_tensors: frozenset[str] = frozenset(),
) -> str | None:
    """Return a skip reason, or None when the tensor is packable."""
    leaf = key.rsplit(".", 1)[-1]
    if leaf in _DROPPED_SIGNAL_TENSOR_NAMES and leaf not in preserve_extra_signal_tensors:
        return leaf
    for pattern in _SKIP_PATTERNS:
        if pattern in key:
            return pattern
    if tensor.nelement() == 0:
        return "empty"
    return None


def _dtype_str(dtype: torch.dtype) -> str:
    """Format a torch dtype without the `torch.` prefix."""
    return str(dtype).removeprefix("torch.")


def _layer_has_key(
    prefix: str,
    tensors: dict[str, torch.Tensor],
    skipped_tensors: dict[str, list],
) -> bool:
    """Return True when a layer has any packable or skipped tensor entry."""
    dotted = prefix + "."
    return any(k.startswith(dotted) for k in tensors) or any(k.startswith(dotted) for k in skipped_tensors)


def _validate_layer_contract(
    tensors: dict[str, torch.Tensor],
    skipped_tensors: dict[str, list],
) -> None:
    """Validate known layer shapes and row counts."""
    for prefix in _LAYER_PREFIXES:
        pos_key = f"{prefix}.positions"
        pos_shape = _state_shape(pos_key, tensors, skipped_tensors)
        n = pos_shape[0] if pos_shape is not None else None
        saw_known_key = False

        for name, expected_shape in _LAYER_TENSOR_SHAPES.items():
            key = f"{prefix}.{name}"
            shape = _state_shape(key, tensors, skipped_tensors)
            if shape is None:
                continue
            saw_known_key = True
            _validate_layer_tensor_shape(key, shape, expected_shape)
            if key in tensors and not tensors[key].is_floating_point():
                raise ValueError(f"{key!r} must be floating-point, got {tensors[key].dtype}")
            if n is not None and shape[0] != n:
                raise ValueError(f"{key!r} has {shape[0]} rows, expected {n} to match {pos_key!r}")

        for name in _PER_GAUSSIAN_TENSOR_NAMES - frozenset(_LAYER_TENSOR_SHAPES):
            key = f"{prefix}.{name}"
            shape = _state_shape(key, tensors, skipped_tensors)
            if shape is None:
                continue
            saw_known_key = True
            if n is not None and shape[0] != n:
                raise ValueError(f"{key!r} has {shape[0]} rows, expected {n} to match {pos_key!r}")

        if n is not None:
            dotted = prefix + "."
            for key, tensor in tensors.items():
                if not key.startswith(dotted) or tensor.ndim < 1 or tensor.shape[0] != n:
                    continue
                leaf = key.removeprefix(dotted)
                if "." not in leaf and leaf not in _PER_GAUSSIAN_TENSOR_NAMES:
                    raise ValueError(
                        f"{key!r} looks row-aligned with {pos_key!r}; add it to the per-Gaussian tensor set"
                    )

        if saw_known_key and pos_shape is None:
            raise ValueError(f"malformed input: {prefix!r} has layer tensors but no {pos_key!r}")


def _state_shape(
    key: str,
    tensors: dict[str, torch.Tensor],
    skipped_tensors: dict[str, list],
) -> tuple[int, ...] | None:
    """Return the packable or skipped tensor shape, or None when absent."""
    if key in tensors:
        return tuple(tensors[key].shape)
    if key in skipped_tensors:
        shape, _ = skipped_tensors[key]
        return tuple(shape)
    return None


def _validate_layer_tensor_shape(
    key: str,
    shape: tuple[int, ...],
    expected_shape: tuple[object, ...],
) -> None:
    """Validate one layer tensor shape.

    Ellipsis accepts one optional middle dimension: (..., 3) matches (N, 3)
    and (N, M, 3).
    """
    if expected_shape and expected_shape[0] is ...:
        trailing = expected_shape[1:]
        ok = len(shape) in (1 + len(trailing), 2 + len(trailing)) and shape[-len(trailing) :] == trailing
        tail = ", ".join(map(str, trailing))
        expected = f"(N, {tail}) or (N, M, {tail})"
    else:
        ok = len(shape) == 1 + len(expected_shape) and shape[1:] == expected_shape
        expected = f"(N, {', '.join(map(str, expected_shape))})"
    if not ok:
        raise ValueError(f"{key!r} must have shape {expected}, got {shape}")


def _cull_opacity(
    tensors: dict[str, torch.Tensor],
    skipped_tensors: dict[str, list],
    prefix: str,
    threshold: float,
) -> int:
    """Cull one layer by opacity and return the removed row count.

    Missing densities are ignored. Per-Gaussian skipped tensors stay row-aligned
    so unpack() can zero-fill the surviving row count.
    """
    density_key = f"{prefix}.densities"
    if density_key not in tensors:
        return 0

    # sigmoid is monotonic: sigmoid(d) >= t iff d >= logit(t).
    # Compare logits to avoid a sigmoid over all rows.
    logit_threshold = math.log(threshold / (1.0 - threshold))
    densities = tensors[density_key].squeeze(-1)
    keep = densities >= logit_threshold
    n_before = len(densities)
    n_after = int(keep.sum())
    if n_after == n_before:
        return 0

    for key in _per_gaussian_tensor_keys(tensors, prefix, n_before):
        tensors[key] = tensors[key][keep]

    # Skipped tensors with the layer row count are assumed Gaussian-row-aligned.
    for key, (shape, _) in skipped_tensors.items():
        if key.startswith(prefix + ".") and shape and shape[0] == n_before:
            shape[0] = n_after

    culled = n_before - n_after
    pct = culled / n_before * 100
    _logger.debug(
        "  %s: %d -> %d (culled %d, %.1f%%)",
        prefix,
        n_before,
        n_after,
        culled,
        pct,
    )
    return culled


def _per_gaussian_tensor_keys(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    n: int,
) -> list[str]:
    """Return explicit per-Gaussian tensor keys under one layer."""
    dotted = prefix + "."
    out = []
    for key, tensor in tensors.items():
        if not key.startswith(dotted) or tensor.ndim < 1 or tensor.shape[0] != n:
            continue
        if key.removeprefix(dotted) in _PER_GAUSSIAN_TENSOR_NAMES:
            out.append(key)
    return out


def _strip_deform_network(
    prefix: str,
    tensors: dict[str, torch.Tensor],
    skipped_tensors: dict[str, list],
    non_tensor_state: dict[str, typing.Any],
) -> None:
    """Disable a fully culled deform network.

    No-op if the layer has no deform-network tensors. unpack() restores stashed
    weights as zeros, so the layer's extra state must be updated to avoid using
    them.

    Args:
        prefix: Deformable layer prefix without the "model." prefix.
        tensors: Packable stripped tensor mapping.
        skipped_tensors: Skipped tensor records to update.
        non_tensor_state: Non-tensor state records.

    Raises:
        ValueError: If the layer has deform-network tensors but no dict-valued
            extra state where use_deform_network can be disabled.
    """
    deform_keys = [k for k in tensors if k.startswith(f"{prefix}.deform_network")]
    if not deform_keys:
        return

    saved = 0
    for key in deform_keys:
        t = tensors.pop(key)
        saved += t.nbytes
        skipped_tensors[key] = [list(t.shape), _dtype_str(t.dtype)]

    # Without this flag, unpack would build an active network with zero weights.
    extra_key = f"{prefix}._extra_state"
    extra = non_tensor_state.get(extra_key)
    if not isinstance(extra, dict):
        raise ValueError(
            f"{prefix}: cannot disable deform network; "
            f"{extra_key!r} missing or not a dict (got {type(extra).__name__}).",
        )
    extra["use_deform_network"] = False

    _logger.debug(
        "  %s: stripped deform_network (%d tensors, %.1f MiB)",
        prefix,
        len(deform_keys),
        saved / 2**20,
    )


def _drop_empty_layer(
    tensors: dict[str, torch.Tensor],
    skipped_tensors: dict[str, list],
    prefix: str,
) -> None:
    """Stash a fully culled layer so codecs never see zero rows."""
    positions = tensors.get(f"{prefix}.positions")
    if positions is None or positions.shape[0] > 0:
        return

    dotted = prefix + "."
    moved = 0
    for key in [k for k in tensors if k.startswith(dotted)]:
        t = tensors.pop(key)
        skipped_tensors[key] = [list(t.shape), _dtype_str(t.dtype)]
        moved += 1
    _logger.debug("  %s: empty after cull; stashed %d tensors", prefix, moved)


def _fp16_auxiliary(
    tensors: dict[str, torch.Tensor],
    skip_albedo: bool = False,
) -> None:
    """Downcast auxiliary fp32 tensors.

    skip_albedo leaves features_albedo in fp32 for the albedo PCA path.
    """
    saved = 0
    for key in list(tensors):
        t = tensors[key]
        segments = key.split(".")
        if t.dtype != torch.float32 or _FP16_PATTERNS.isdisjoint(segments):
            continue
        if key.endswith(".features_albedo") and (skip_albedo or not _is_known_layer_tensor(key, "features_albedo")):
            continue
        tensors[key] = t.half()
        saved += t.nbytes // 2
    if saved:
        _logger.info("  fp16 auxiliary: saved %.1f MiB", saved / 2**20)


def _is_known_layer_tensor(key: str, name: str) -> bool:
    """Return True for an exact tensor name under a known Gaussian layer."""
    return any(key == f"{prefix}.{name}" for prefix in _LAYER_PREFIXES)


def _fp16_specular(tensors: dict[str, torch.Tensor]) -> None:
    """Downcast known-layer `features_specular` tensors to fp16 in place."""
    saved = 0
    for key in _known_layer_keys(tensors, "features_specular"):
        t = tensors[key]
        if t.dtype == torch.float32:
            tensors[key] = t.half()
            saved += t.nbytes // 2
    if saved:
        _logger.info("  fp16 specular: saved %.1f MiB", saved / 2**20)


def _morton_sort_layer(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    position_mode: _PositionMode = "fp16_offset",
    voxel_size: float = VOXEL_SIZE,
) -> None:
    """Sort a layer by Morton code and optionally encode local positions.

    "fp16_offset" adds centroid and range sidecars used by unpack(). "fp32_sorted"
    keeps absolute fp32 positions after sorting. Missing or empty positions are
    ignored. Output is deterministic on CUDA.

    Args:
        tensors: Packable stripped tensor mapping.
        prefix: Layer prefix without the "model." prefix.
        position_mode: Position storage mode after sorting.
        voxel_size: Voxel edge length in meters. Must be positive.

    Raises:
        ValueError: If positions are malformed, non-finite, or exceed the
            two-segment Morton range.
    """
    _validate_position_mode(position_mode)

    pos_key = f"{prefix}.positions"
    if pos_key not in tensors:
        return

    positions = tensors[pos_key]
    n = positions.shape[0]
    if n == 0:
        return
    device = positions.device

    order, morton_low, morton_high = morton_sort_order(positions, voxel_size, label=pos_key)

    for key in _per_gaussian_tensor_keys(tensors, prefix, n):
        tensors[key] = tensors[key][order]

    if position_mode == "fp32_sorted":
        tensors[pos_key] = tensors[pos_key].float()
        _logger.debug("  %s: %d points (Morton-sorted, fp32 positions)", prefix, n)
        return

    counts = _morton_run_counts(morton_low, order, morton_high)
    ranges = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            counts.cumsum(0).int(),
        ]
    )

    # segment_reduce is deterministic on CUDA. scatter_add_ is not.
    # float cumsum also lacks a deterministic CUDA kernel.
    sorted_pos = tensors[pos_key].float()
    sums = torch.segment_reduce(sorted_pos, reduce="sum", lengths=counts)
    centroids = sums / counts.unsqueeze(1).float()

    voxel_idx = torch.repeat_interleave(counts)
    tensors[pos_key] = (sorted_pos - centroids[voxel_idx]).half()
    tensors[f"{prefix}._voxel_centroids"] = centroids
    tensors[f"{prefix}._voxel_ranges"] = ranges

    _logger.debug(
        "  %s: %d points -> %d voxels (Morton-sorted, fp16 offsets)",
        prefix,
        n,
        len(counts),
    )


def _morton_run_counts(
    low: torch.Tensor,
    order: torch.Tensor,
    high: torch.Tensor | None,
) -> torch.Tensor:
    """Return sorted voxel run lengths for one- or two-segment Morton keys."""
    sorted_low = low[order]
    if high is None:
        return sorted_low.unique_consecutive(return_counts=True)[1]

    sorted_high = high[order]
    starts = torch.ones(len(order), dtype=torch.bool, device=order.device)
    starts[1:] = (sorted_low[1:] != sorted_low[:-1]) | (sorted_high[1:] != sorted_high[:-1])
    start_indices = starts.nonzero().flatten()
    return torch.diff(torch.cat([start_indices, start_indices.new_tensor([len(order)])]))


def _unmorton_layers(tensors: dict[str, torch.Tensor]) -> None:
    """Restore absolute fp32 positions from voxel sidecars."""
    for key in [k for k in tensors if k.endswith("._voxel_centroids")]:
        prefix = key.removesuffix("._voxel_centroids")
        centroids = tensors.pop(key)
        range_key = f"{prefix}._voxel_ranges"
        pos_key = f"{prefix}.positions"
        if range_key not in tensors:
            raise ValueError(f"{key!r} has voxel sidecar but no {range_key!r}")
        if pos_key not in tensors:
            raise ValueError(f"{key!r} has voxel sidecar but no {pos_key!r}")
        ranges = tensors.pop(range_key)
        offsets = tensors[pos_key]
        counts = ranges[1:] - ranges[:-1]
        voxel_idx = torch.repeat_interleave(counts)
        tensors[pos_key] = centroids[voxel_idx] + offsets


def _quantize_geometry(
    tensors: dict[str, torch.Tensor],
    *,
    density_bits: int,
    scale_bits: int,
) -> None:
    """Quantize densities and scales in place."""
    _validate_geometry_bits("density_bits", density_bits)
    _validate_geometry_bits("scale_bits", scale_bits)
    saved = 0

    for key in [k for k in _known_layer_keys(tensors, "densities") if tensors[k].is_floating_point()]:
        t = tensors[key]
        flat = t.float().reshape(t.shape[0], -1)
        quantized, col_absmax = (
            _quantize_band_symmetric_int16(flat) if density_bits == 16 else _quantize_band_symmetric(flat)
        )
        tensors[key] = quantized.reshape(t.shape)
        tensors[f"{key}_absmax"] = col_absmax.reshape(t.shape[1:])
        saved += t.nbytes - quantized.nbytes - col_absmax.nbytes

    for key in [k for k in _known_layer_keys(tensors, "scales") if tensors[k].is_floating_point()]:
        t = tensors[key]
        flat = t.float().reshape(t.shape[0], -1)
        quantized, col_min, col_scale = (
            _quantize_affine_uint16(flat) if scale_bits == 16 else _quantize_affine_uint8(flat)
        )
        tensors[key] = quantized.reshape(t.shape)
        tensors[f"{key}_min"] = col_min.reshape(t.shape[1:])
        tensors[f"{key}_scale"] = col_scale.reshape(t.shape[1:])
        saved += t.nbytes - quantized.nbytes - col_min.nbytes - col_scale.nbytes

    if saved:
        _logger.info("  geometry: saved %.1f MiB", saved / 2**20)


def _dequantize_geometry(tensors: dict[str, torch.Tensor]) -> None:
    """Restore quantized densities and scales in place."""
    for key in _sidecar_bases(tensors, "densities", "_absmax"):
        if tensors[key].dtype not in (torch.int8, torch.int16):
            raise ValueError(f"{key!r} has density sidecar but dtype {tensors[key].dtype}")
        col_absmax = tensors.pop(f"{key}_absmax")
        quantized = tensors[key]
        shape = quantized.shape
        flat = quantized.reshape(shape[0], -1)
        restored = (
            _dequant_band_symmetric_int16(flat, col_absmax)
            if quantized.dtype == torch.int16
            else _dequant_band_symmetric(flat, col_absmax)
        )
        tensors[key] = restored.reshape(shape)

    scale_keys = set(_sidecar_bases(tensors, "scales", "_min"))
    scale_keys.update(_sidecar_bases(tensors, "scales", "_scale"))
    for key in sorted(scale_keys):
        if tensors[key].dtype not in (torch.uint8, torch.int16):
            raise ValueError(f"{key!r} has scale sidecar but dtype {tensors[key].dtype}")
        min_key = f"{key}_min"
        scale_key = f"{key}_scale"
        if min_key not in tensors:
            raise ValueError(f"{key!r} has scale sidecar but no {min_key!r}")
        if scale_key not in tensors:
            raise ValueError(f"{key!r} has scale sidecar but no {scale_key!r}")
        col_min = tensors.pop(min_key)
        col_scale = tensors.pop(scale_key)
        quantized = tensors[key]
        shape = quantized.shape
        flat = quantized.reshape(shape[0], -1)
        restored = (
            _dequant_affine_uint16(flat, col_min, col_scale)
            if quantized.dtype == torch.int16
            else _dequant_affine_uint8(flat, col_min, col_scale)
        )
        tensors[key] = restored.reshape(shape)


def _gather_complement_cols(
    largest_idx: torch.Tensor,
    total: int = 4,
) -> torch.Tensor:
    """Return `range(total)` minus each row's `largest_idx`, ascending; shape `(N, total - 1)`."""
    positions = torch.arange(total - 1, device=largest_idx.device).unsqueeze(0)
    return torch.where(positions < largest_idx.unsqueeze(1), positions, positions + 1)


def _pack_rotations(
    tensors: dict[str, torch.Tensor],
    *,
    rotation_bits: int,
) -> None:
    """Encode rotations as smallest-three quaternions.

    Inputs are renormalized first; callers do not need to pre-normalize them.

    Args:
        tensors: Packable stripped tensor mapping to mutate.
        rotation_bits: Stored component bits. Must be 8 or 16.
    """
    _validate_geometry_bits("rotation_bits", rotation_bits)
    saved = 0
    quant_max = _INT16_SYMMETRIC_MAX if rotation_bits == 16 else _INT8_SYMMETRIC_MAX
    quant_scale = _ROT16_QUANT_SCALE if rotation_bits == 16 else _ROT_QUANT_SCALE
    quant_dtype = torch.int16 if rotation_bits == 16 else torch.int8
    for key in _known_layer_keys(tensors, "rotations"):
        rot = tensors[key]
        q = torch.nn.functional.normalize(rot.float(), dim=1)

        # A zero sign would zero the quaternion. Use +1 for degenerate rows.
        largest_idx = q.abs().argmax(dim=1)
        sign = q.gather(1, largest_idx.unsqueeze(1)).sign()
        sign = sign.where(sign != 0, torch.ones_like(sign))
        q.mul_(sign)

        small_cols = _gather_complement_cols(largest_idx)
        small = q.gather(1, small_cols)

        scaled = (small * quant_scale).round()
        quantized = scaled.clamp(-quant_max, quant_max).to(quant_dtype)
        packed_idx = _pack_crumbs(largest_idx.to(torch.uint8).unsqueeze(0)).squeeze(0)

        original_bytes = tensors[key].nbytes
        tensors[key] = quantized
        tensors[f"{key}_idx"] = packed_idx
        saved += original_bytes - quantized.nbytes - packed_idx.nbytes

    if saved:
        _logger.info("  smallest-three rotations (%d-bit): saved %.1f MiB", rotation_bits, saved / 2**20)


def _unpack_rotations(tensors: dict[str, torch.Tensor]) -> None:
    """Restore smallest-three rotations to fp32 quaternions."""
    for key in _sidecar_bases(tensors, "rotations", "_idx"):
        if tensors[key].dtype not in (torch.int8, torch.int16):
            raise ValueError(f"{key!r} has rotation sidecar but dtype {tensors[key].dtype}")
        quant_max = _INT16_SYMMETRIC_MAX if tensors[key].dtype == torch.int16 else _INT8_SYMMETRIC_MAX
        small = tensors[key].float() * (_SQRT2_INV / quant_max)
        n = small.shape[0]
        device = small.device

        largest_idx = _unpack_crumbs(tensors.pop(f"{key}_idx"), n).long()

        q = torch.zeros(n, 4, device=device)
        small_cols = _gather_complement_cols(largest_idx)
        q.scatter_(1, small_cols, small)

        # Round-trip error can make the radicand slightly negative.
        largest = (1.0 - small.pow(2).sum(dim=1)).clamp(min=0.0).sqrt()
        q.scatter_(1, largest_idx.unsqueeze(1), largest.unsqueeze(1))

        tensors[key] = q


def _quantize_sh(
    tensors: dict[str, torch.Tensor],
    sh_total_bits: int,
) -> None:
    """Quantize known-layer SH features.

    Args:
        tensors: Packable stripped tensor mapping to mutate.
        sh_total_bits: Total PCA bit budget per Gaussian.

    Raises:
        ValueError: If a compressed SH tensor is not float32.
    """
    total_saved = 0
    for key in _known_layer_keys(tensors, "features_specular"):
        specular = tensors.pop(key)
        if specular.dtype != torch.float32:
            raise ValueError(f"{key!r} must be float32 for SH quantization, got {specular.dtype}")

        min_elements = _page_setting_for_key(key, _DEFAULT_PAGE_MIN_ELEMENTS)
        max_pages = _page_setting_for_key(key, _DEFAULT_PAGE_MAX_COUNT)
        page_ids, page_count = _layer_pages(specular, min_elements=min_elements, max_pages=max_pages)
        original_bytes = specular.nbytes
        new_bytes = _quantize_paged_pca(
            tensors,
            key,
            specular,
            sh_total_bits,
            page_ids,
            page_count,
        )
        total_saved += original_bytes - new_bytes

    if total_saved:
        _logger.info("  SH features: saved %.1f MiB", total_saved / 2**20)


def _known_layer_keys(
    tensors: dict[str, torch.Tensor],
    name: str,
) -> list[str]:
    """Return present tensor keys for known Gaussian layers."""
    return [f"{prefix}.{name}" for prefix in _LAYER_PREFIXES if f"{prefix}.{name}" in tensors]


def _sidecar_bases(
    tensors: dict[str, torch.Tensor],
    tensor_name: str,
    sidecar_suffix: str,
) -> list[str]:
    """Return base tensor keys discovered from compact sidecars."""
    marker = f".{tensor_name}{sidecar_suffix}"
    keys = []
    for sidecar_key in sorted(k for k in tensors if k.endswith(marker)):
        key = sidecar_key[: -len(sidecar_suffix)]
        if key not in tensors:
            raise ValueError(f"{sidecar_key!r} has sidecar but no base tensor {key!r}")
        keys.append(key)
    return keys


def _unpack_paged(
    tensors: dict[str, torch.Tensor],
    feature_suffix: str,
    reshape: typing.Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    """Restore paged PCA features and reshape the decoded matrix."""
    marker = f".{feature_suffix}_leaf_basis"
    keys = [k[: -len("_leaf_basis")] for k in tensors if k.endswith(marker)]
    for key in keys:
        prefix = key[: -len(f".{feature_suffix}")]
        pos_key = f"{prefix}.positions"
        if pos_key not in tensors:
            raise ValueError(f"{key!r} has PCA sidecars but no {pos_key!r} to recover row count")
        n = tensors[pos_key].shape[0]
        basis = tensors[f"{key}_leaf_basis"]
        page_ids = _derive_page_ids(tensors, key, n)
        flat = _dequantize_paged_pca(
            tensors,
            key,
            n,
            basis.shape[-1],
            basis.device,
            page_ids,
        )
        tensors[key] = reshape(key, flat, basis)


def _unpack_sh(tensors: dict[str, torch.Tensor]) -> None:
    """Restore compressed features_specular tensors."""

    def _reshape(key, flat, basis):
        if basis.shape[-1] != _SH_TOTAL_COLS:
            raise ValueError(f"SH leaf basis has {basis.shape[-1]} cols, expected {_SH_TOTAL_COLS}")
        return flat

    _unpack_paged(tensors, "features_specular", _reshape)


def _pad_pages(
    data: torch.Tensor,
    page_ids: torch.Tensor,
    page_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad monotone pages to a dense batch.

    Page ids must be in [0, page_count). Short pages duplicate their last valid
    row. Returns padded rows, counts, and a valid-row mask.
    """
    n = data.shape[0]
    plan = _page_padding_plan(page_ids, page_count, n)
    return _pad_pages_with_plan(data, plan)


def _page_padding_plan(
    page_ids: torch.Tensor,
    page_count: int,
    n_rows: int,
) -> _PagePaddingPlan:
    """Return counts, gather indices, and valid rows for monotone pages."""
    if page_ids.numel() != n_rows:
        raise ValueError(f"page_ids length {page_ids.numel()} != data rows {n_rows}")
    counts = torch.bincount(page_ids, minlength=page_count)
    if counts.numel() > page_count:
        raise ValueError(f"page_ids has values outside [0, {page_count})")

    device = page_ids.device
    offsets = torch.cumsum(counts, 0) - counts
    n_p_max = int(counts.max())

    arange = torch.arange(n_p_max, device=device)
    clamped = torch.minimum(
        arange.unsqueeze(0),
        (counts - 1).clamp(min=0).unsqueeze(1),
    )
    gather_idx = (offsets.unsqueeze(1) + clamped).clamp(max=n_rows - 1)
    valid_mask = arange.unsqueeze(0) < counts.unsqueeze(1)
    return counts, gather_idx, valid_mask


def _pad_pages_with_plan(
    data: torch.Tensor,
    plan: _PagePaddingPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad monotone pages using a precomputed page plan."""
    counts, gather_idx, valid_mask = plan
    data_3d = data[gather_idx]
    return data_3d, counts, valid_mask


def _layer_pages(
    data: torch.Tensor,
    min_elements: int,
    max_pages: int,
) -> tuple[torch.Tensor, int]:
    """Return even row-index page ids and page count."""
    n_rows = data.shape[0]
    device = data.device
    page_count = max(1, min(n_rows // min_elements, max_pages))
    if page_count == 1:
        return torch.zeros(n_rows, dtype=torch.int64, device=device), 1
    ids = torch.arange(n_rows, dtype=torch.int64, device=device) * page_count // n_rows
    return ids.clamp_max(page_count - 1), page_count


def _page_setting_for_key(
    key: str,
    settings: dict[str, int],
) -> int:
    """Return a page setting for a known layer feature key."""
    for prefix in _LAYER_PREFIXES:
        if key.startswith(prefix + "."):
            layer = prefix.removeprefix("gaussians_nodes.")
            return settings[layer]
    raise ValueError(f"cannot infer layer for {key!r}")


def _derive_page_ids(
    tensors: dict[str, torch.Tensor],
    feature_key: str,
    n_rows: int,
) -> torch.Tensor:
    """Return row-index page ids from the stored page count."""
    page_count = int(tensors[f"{feature_key}_page_count"].item())
    device = tensors[f"{feature_key}_leaf_basis"].device
    if page_count == 1:
        return torch.zeros(n_rows, dtype=torch.int64, device=device)
    return (torch.arange(n_rows, dtype=torch.int64, device=device) * page_count // n_rows).clamp_max(page_count - 1)


def _quantize_paged_pca(
    tensors: dict[str, torch.Tensor],
    key: str,
    data: torch.Tensor,
    total_bits: int,
    page_ids: torch.Tensor,
    page_count: int,
) -> int:
    """Compress one feature matrix with paged PCA.

    Each page gets its own basis and mean. The bit schedule is shared across
    pages so a layer has one compact set of PCA buckets.

    Args:
        tensors: Tensor mapping to receive sidecars.
        key: Feature key prefix for emitted sidecars.
        data: Feature matrix with shape (N, D).
        total_bits: Total PCA bit budget per row.
        page_ids: Page id per row.
        page_count: Number of pages.

    Returns:
        Number of bytes written for payloads and sidecars.
    """
    n, d = data.shape
    device = data.device
    dtype = data.dtype

    padding_plan = _page_padding_plan(page_ids, page_count, n)
    data_3d, counts, valid_mask = _pad_pages_with_plan(data, padding_plan)
    counts_f = counts.to(dtype).clamp(min=1)
    # Mean / cov ignore padded duplicates by zeroing them after centering.
    leaf_mean = (data_3d * valid_mask.unsqueeze(-1)).sum(dim=1) / counts_f.unsqueeze(-1)
    centered = (data_3d - leaf_mean.unsqueeze(1)) * valid_mask.unsqueeze(-1)
    cov = centered.transpose(1, 2) @ centered / counts_f.view(page_count, 1, 1)
    cov = cov + torch.eye(d, dtype=dtype, device=device) * _EPS
    eigvals_asc, U_asc = torch.linalg.eigh(cov)
    eigvals = eigvals_asc.flip(-1).contiguous()
    leaf_basis = U_asc.flip(-1).contiguous()
    leaf_mean_h = leaf_mean.to(torch.float16)
    leaf_basis_h = leaf_basis.to(torch.float16)

    # Re-center with stored fp16 sidecars so encode and decode use the same basis.
    centered_h = (data_3d - leaf_mean_h.to(dtype).unsqueeze(1)) * valid_mask.unsqueeze(-1)
    proj_3d = centered_h @ leaf_basis_h.to(dtype)
    proj = proj_3d[valid_mask]

    blended_eigvals = (counts_f.unsqueeze(1) * eigvals).sum(dim=0) / n
    bits_per_axis = _waterfill_bits(blended_eigvals, total_bits)

    written = 0
    for b in (16, 8, 4, 2):
        axis_idx = (bits_per_axis == b).nonzero(as_tuple=True)[0]
        if axis_idx.numel() == 0:
            continue
        block = proj.index_select(1, axis_idx)
        written += _store_paged_pca_bucket(tensors, f"{key}_pca_b{b}", block, b, padding_plan)

    tensors[f"{key}_page_count"] = torch.tensor(page_count, dtype=torch.int32, device=device)
    tensors[f"{key}_leaf_basis"] = leaf_basis_h
    tensors[f"{key}_leaf_mean"] = leaf_mean_h
    tensors[f"{key}_pca_bits"] = bits_per_axis.to(torch.uint8)
    written += (
        tensors[f"{key}_page_count"].nbytes
        + tensors[f"{key}_leaf_basis"].nbytes
        + tensors[f"{key}_leaf_mean"].nbytes
        + tensors[f"{key}_pca_bits"].nbytes
    )
    return written


def _store_paged_pca_bucket(
    tensors: dict[str, torch.Tensor],
    key: str,
    block: torch.Tensor,
    bits: int,
    padding_plan: _PagePaddingPlan,
) -> int:
    """Store one PCA bit bucket and return bytes written."""
    if bits == 16:
        tensors[key] = block.to(torch.float16)
        return tensors[key].nbytes
    if bits not in (8, 4, 2):
        raise ValueError(f"PCA bucket bit width must be 16, 8, 4, or 2, got {bits}")

    _, k_cols = block.shape
    block_3d, counts, valid_mask = _pad_pages_with_plan(block, padding_plan)
    page_count = counts.shape[0]
    c_min = block_3d.amin(dim=1)
    c_max = block_3d.amax(dim=1)
    c_scale = (c_max - c_min).clamp(min=_EPS)
    normalized = (block_3d - c_min.unsqueeze(1)) / c_scale.unsqueeze(1)
    min_k = c_min.to(torch.float16)
    scale_k = c_scale.to(torch.float16)

    if bits == 8:
        packed = (normalized * _UINT8_AFFINE_MAX).round().clamp(0, _UINT8_AFFINE_MAX).to(torch.uint8)[valid_mask]
        tensors[key] = packed
        tensors[f"{key}_min"] = min_k
        tensors[f"{key}_scale"] = scale_k
        return packed.nbytes + min_k.nbytes + scale_k.nbytes

    cb_size = 1 << bits
    pack_fn = {4: _pack_nibbles, 2: _pack_crumbs}[bits]

    n_p = normalized.shape[1]
    norm_2d = normalized.permute(1, 0, 2).reshape(n_p, page_count * k_cols)
    indices_2d, codebooks_2d = _lloyd_quantize_axis(norm_2d, cb_size)
    indices_3d = indices_2d.reshape(n_p, page_count, k_cols).permute(1, 0, 2)
    indices = indices_3d[valid_mask]
    codebooks = codebooks_2d.reshape(page_count, k_cols, cb_size).to(torch.float16)
    packed = pack_fn(indices)
    tensors[key] = packed
    tensors[f"{key}_min"] = min_k
    tensors[f"{key}_scale"] = scale_k
    tensors[f"{key}_codebooks"] = codebooks
    return packed.nbytes + min_k.nbytes + scale_k.nbytes + codebooks.nbytes


def _dequant_paged_pca_bucket(
    tensors: dict[str, torch.Tensor],
    key: str,
    k_cols: int,
    bits: int,
    page_ids: torch.Tensor,
) -> torch.Tensor:
    """Restore one PCA bit bucket as an fp32 matrix."""
    if bits == 16:
        return tensors.pop(key).float()
    quantized = tensors.pop(key)
    min_k = tensors.pop(f"{key}_min").float()
    scale_k = tensors.pop(f"{key}_scale").float()

    if bits == 8:
        return quantized.float() * (scale_k[page_ids] / _UINT8_AFFINE_MAX) + min_k[page_ids]

    codebooks = tensors.pop(f"{key}_codebooks").float()

    unpack_fn = {4: _unpack_nibbles, 2: _unpack_crumbs}[bits]
    indices = unpack_fn(quantized, k_cols).long()
    cb_size = codebooks.shape[-1]
    flat_cb = codebooks.reshape(-1)
    col_offset = torch.arange(k_cols, device=indices.device) * cb_size
    flat_idx = (page_ids.unsqueeze(-1) * (k_cols * cb_size)) + col_offset + indices
    values = flat_cb[flat_idx]
    return values * scale_k[page_ids] + min_k[page_ids]


def _dequantize_paged_pca(
    tensors: dict[str, torch.Tensor],
    key: str,
    n_rows: int,
    d: int,
    device: torch.device,
    page_ids: torch.Tensor,
) -> torch.Tensor:
    """Restore one paged PCA feature matrix."""
    page_count = int(tensors.pop(f"{key}_page_count").item())
    leaf_basis = tensors.pop(f"{key}_leaf_basis").float()
    leaf_mean = tensors.pop(f"{key}_leaf_mean").float()
    bits_per_axis = tensors.pop(f"{key}_pca_bits")

    proj = torch.zeros(n_rows, d, dtype=torch.float32, device=device)
    for b in (16, 8, 4, 2):
        axis_idx = (bits_per_axis == b).nonzero(as_tuple=True)[0]
        if axis_idx.numel() == 0:
            continue
        block = _dequant_paged_pca_bucket(tensors, f"{key}_pca_b{b}", axis_idx.numel(), b, page_ids)
        proj.index_copy_(1, axis_idx.to(device), block.to(device))

    proj_3d, _, valid_mask = _pad_pages(proj, page_ids, page_count)
    out_3d = proj_3d @ leaf_basis.transpose(-1, -2) + leaf_mean.unsqueeze(1)
    return out_3d[valid_mask]


def _quantize_albedo(
    tensors: dict[str, torch.Tensor],
    albedo_total_bits: int,
) -> None:
    """Quantize known-layer albedo features.

    Albedo may be shaped (N, 3) or (N, M, 3). The budget is per RGB triple.

    Args:
        tensors: Packable stripped tensor mapping to mutate.
        albedo_total_bits: PCA bit budget per RGB triple.

    Raises:
        ValueError: If a compressed albedo tensor is not float32 or does not
            end in RGB channels.
    """
    total_saved = 0
    for key in _known_layer_keys(tensors, "features_albedo"):
        albedo = tensors.pop(key)
        if albedo.dtype != torch.float32:
            raise ValueError(f"{key!r} must be float32 for PCA albedo quantization, got {albedo.dtype}")
        if albedo.shape[-1] != 3:
            raise ValueError(f"{key!r} last-dim is {albedo.shape[-1]}, expected 3 (RGB) for PCA albedo")
        n = albedo.shape[0]
        m = albedo.numel() // (n * 3)
        flat = albedo.reshape(n, m * 3)
        min_elements = _page_setting_for_key(key, _DEFAULT_PAGE_MIN_ELEMENTS)
        max_pages = _page_setting_for_key(key, _DEFAULT_PAGE_MAX_COUNT)
        page_ids, page_count = _layer_pages(flat, min_elements=min_elements, max_pages=max_pages)
        original_bytes = albedo.nbytes
        new_bytes = _quantize_paged_pca(
            tensors,
            key,
            flat,
            albedo_total_bits * m,
            page_ids,
            page_count,
        )
        tensors[f"{key}_shape"] = torch.tensor(albedo.shape[1:], dtype=torch.int32, device=albedo.device)
        new_bytes += tensors[f"{key}_shape"].nbytes
        total_saved += original_bytes - new_bytes

    if total_saved:
        _logger.info(
            "  paged PCA albedo (%d bits/triple): saved %.1f MiB",
            albedo_total_bits,
            total_saved / 2**20,
        )


def _unpack_albedo(tensors: dict[str, torch.Tensor]) -> None:
    """Restore compressed features_albedo tensors."""

    def _reshape(key, flat, basis):
        d = basis.shape[-1]
        shape_key = f"{key}_shape"
        if shape_key in tensors:
            tail = tuple(int(x) for x in tensors.pop(shape_key).tolist())
            if not tail or tail[-1] != 3 or math.prod(tail) != d:
                raise ValueError(f"{shape_key!r} is incompatible with decoded width {d}")
            return flat.reshape(flat.shape[0], *tail)
        if d % 3 != 0:
            raise ValueError(f"albedo leaf basis has {d} cols, expected multiple of 3 (RGB)")
        n = flat.shape[0]
        m = d // 3
        return flat.reshape(n, 3) if m == 1 else flat.reshape(n, m, 3)

    _unpack_paged(tensors, "features_albedo", _reshape)


def _affine_normalize(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize columns to [0, 1] and return data, minima, and scales."""
    col_min, col_max = torch.aminmax(tensor, dim=0)
    col_scale = (col_max - col_min).clamp(min=_EPS)
    return (tensor - col_min) / col_scale, col_min, col_scale


def _quantize_affine_uint8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize columns to affine uint8 and return decode sidecars."""
    normalized, col_min, col_scale = _affine_normalize(tensor)
    q = (normalized * _UINT8_AFFINE_MAX).round().clamp(0, _UINT8_AFFINE_MAX).to(torch.uint8)
    return q, col_min, col_scale


def _dequant_affine_uint8(
    quantized: torch.Tensor,
    col_min: torch.Tensor,
    col_scale: torch.Tensor,
) -> torch.Tensor:
    """Restore affine uint8 columns to fp32."""
    return quantized.float() * (col_scale / _UINT8_AFFINE_MAX) + col_min


def _quantize_affine_uint16(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize columns to affine uint16 codes and return decode sidecars.

    The payload is stored as biased int16 because safetensors does not support
    uint16 in the Bazel runtime.
    """
    normalized, col_min, col_scale = _affine_normalize(tensor)
    q = (normalized * _UINT16_AFFINE_MAX).round().clamp(0, _UINT16_AFFINE_MAX)
    q = (q.to(torch.int32) - 32768).to(torch.int16)
    return q, col_min, col_scale


def _dequant_affine_uint16(
    quantized: torch.Tensor,
    col_min: torch.Tensor,
    col_scale: torch.Tensor,
) -> torch.Tensor:
    """Restore affine uint16-code columns to fp32."""
    unsigned = quantized.to(torch.int32) + 32768
    return unsigned.float() * (col_scale / _UINT16_AFFINE_MAX) + col_min


def _quantize_band_symmetric(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize centered columns to symmetric int8 and return absmax sidecars."""
    col_absmax = tensor.abs().amax(dim=0).clamp(min=_EPS)
    scale = _INT8_SYMMETRIC_MAX / col_absmax
    scaled = (tensor * scale).round()
    quantized = scaled.clamp(-_INT8_SYMMETRIC_MAX, _INT8_SYMMETRIC_MAX).to(torch.int8)
    return quantized, col_absmax


def _dequant_band_symmetric(
    quantized: torch.Tensor,
    col_absmax: torch.Tensor,
) -> torch.Tensor:
    """Restore symmetric int8 columns to fp32."""
    return quantized.float() * (col_absmax / _INT8_SYMMETRIC_MAX)


def _quantize_band_symmetric_int16(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize centered columns to symmetric int16 and return absmax sidecars."""
    col_absmax = tensor.abs().amax(dim=0).clamp(min=_EPS)
    scale = _INT16_SYMMETRIC_MAX / col_absmax
    scaled = (tensor * scale).round()
    quantized = scaled.clamp(-_INT16_SYMMETRIC_MAX, _INT16_SYMMETRIC_MAX).to(torch.int16)
    return quantized, col_absmax


def _dequant_band_symmetric_int16(
    quantized: torch.Tensor,
    col_absmax: torch.Tensor,
) -> torch.Tensor:
    """Restore symmetric int16 columns to fp32."""
    return quantized.float() * (col_absmax / _INT16_SYMMETRIC_MAX)


def _lloyd_max_axis(
    data: torch.Tensor,
    k: int,
    n_iter: int = 20,
) -> torch.Tensor:
    """Fit ascending Lloyd-Max centroids per column.

    Fits on a stride subsample. Linspace initialization covers the full value
    range, including fat-tailed outliers.
    """
    n_full, c = data.shape
    stride = max(1, (n_full + _LLOYD_FIT_BUDGET - 1) // _LLOYD_FIT_BUDGET)

    fit_sorted = data[::stride].sort(dim=0).values
    fit_t = fit_sorted.transpose(0, 1).contiguous()
    n = fit_t.shape[1]
    cumsum = torch.cat([fit_t.new_zeros(c, 1), fit_t.cumsum(dim=1)], dim=1)

    centroids = torch.linspace(0.0, 1.0, k, dtype=fit_t.dtype, device=fit_t.device)
    centroids = centroids.unsqueeze(0).expand(c, k).clone()
    centroids.mul_(fit_t[:, -1:] - fit_t[:, :1]).add_(fit_t[:, :1])
    pad_lo = torch.zeros(c, 1, dtype=torch.int64, device=fit_t.device)
    pad_hi = torch.full((c, 1), n, dtype=torch.int64, device=fit_t.device)

    for _ in range(n_iter):
        boundaries = (centroids[:, :-1] + centroids[:, 1:]) / 2
        idx = torch.searchsorted(fit_t, boundaries)
        pos = torch.cat([pad_lo, idx, pad_hi], dim=1)
        counts = pos[:, 1:] - pos[:, :-1]
        sums = cumsum.gather(1, pos[:, 1:]) - cumsum.gather(1, pos[:, :-1])
        new_centroids = torch.where(
            counts > 0,
            sums / counts.to(fit_t.dtype).clamp(min=1),
            centroids,
        )
        centroids = new_centroids
    # `_lloyd_quantize_axis` calls searchsorted on implied boundaries,
    # which requires sorted codebooks. Sorting hardens degenerate cases
    # where Lloyd-Max output is not monotone.
    return centroids.sort(dim=-1).values


def _lloyd_quantize_axis(
    normalized: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign normalized columns to Lloyd-Max codebooks."""
    codebook = _lloyd_max_axis(normalized, k)
    boundaries = (codebook[:, :-1] + codebook[:, 1:]) / 2
    norm_t = normalized.transpose(0, 1).contiguous()
    indices_t = torch.searchsorted(boundaries, norm_t)
    return indices_t.transpose(0, 1).contiguous().to(torch.uint8), codebook


def _waterfill_bits(
    eigvals: torch.Tensor,
    total_bits: int,
    levels: tuple[int, ...] = (0, 2, 4, 8, 16),
) -> torch.Tensor:
    """Allocate PCA bits across axes under a total budget."""
    next_lvl = dict(zip(levels, levels[1:]))
    eigs = [float(e) for e in eigvals]
    bits = [0] * len(eigs)
    spent = 0
    while spent < total_bits:
        best, best_pri = -1, 0.0
        for i, cur in enumerate(bits):
            nxt = next_lvl.get(cur)
            if nxt is None or spent + nxt - cur > total_bits:
                continue
            pri = eigs[i] / 4.0**cur
            if pri > best_pri:
                best, best_pri = i, pri
        if best < 0:
            break
        cur = bits[best]
        bits[best] = next_lvl[cur]
        spent += bits[best] - cur
    return torch.tensor(bits, dtype=torch.int32, device=eigvals.device)


def _pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into uint8. Pads the column count to even."""
    _, c = indices.shape
    pad = (-c) % 2
    if pad:
        indices = torch.nn.functional.pad(indices, (0, pad))
    return indices[:, 0::2] | (indices[:, 1::2] << 4)


def _pack_crumbs(indices: torch.Tensor) -> torch.Tensor:
    """Pack quadruples of 2-bit indices into uint8. Pads the column count to a multiple of 4."""
    _, c = indices.shape
    pad = (-c) % 4
    if pad:
        indices = torch.nn.functional.pad(indices, (0, pad))
    return indices[:, 0::4] | (indices[:, 1::4] << 2) | (indices[:, 2::4] << 4) | (indices[:, 3::4] << 6)


def _unpack_crumbs(packed: torch.Tensor, n_codes: int) -> torch.Tensor:
    """Unpack 2-bit codes and drop padding."""
    shifts = torch.arange(0, 8, 2, dtype=torch.uint8, device=packed.device)
    return ((packed.unsqueeze(-1) >> shifts) & 0x03).reshape(*packed.shape[:-1], -1)[..., :n_codes]


def _unpack_nibbles(packed: torch.Tensor, n_cols: int) -> torch.Tensor:
    """Unpack 4-bit codes and drop padding."""
    shifts = torch.arange(0, 8, 4, dtype=torch.uint8, device=packed.device)
    n = packed.shape[0]
    return ((packed.unsqueeze(-1) >> shifts) & 0x0F).reshape(n, -1)[:, :n_cols]
