# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np
import pytest
import safetensors.torch
import torch

from nurec.compression import gaussian_codec as compact
from nurec.utils.morton import VOXEL_SIZE


_NUM_GAUSSIANS = 1024


def _round_trip_atol(key: str, orig: torch.Tensor) -> float:
    """Return the max-abs tolerance for one state key."""
    if key.endswith(".densities"):
        return orig.abs().amax(0).max().item() / 32767.0 * 1.05
    if key.endswith(".scales"):
        return (orig.amax(0) - orig.amin(0)).max().item() / 65535.0 * 1.05
    if key.endswith((".positions", ".features_albedo")):
        return 1e-2
    if key.endswith(".features_specular"):
        return 1e-3
    raise KeyError(f"no round-trip tolerance for {key!r}")


def _layer(prefix: str, n: int, rng: torch.Generator) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.positions": torch.randn(n, 3, generator=rng) * 50.0,
        f"{prefix}.densities": torch.randn(n, 1, generator=rng),
        f"{prefix}.scales": torch.rand(n, 3, generator=rng) + 0.01,
        f"{prefix}.rotations": torch.nn.functional.normalize(torch.randn(n, 4, generator=rng), dim=1),
        f"{prefix}.features_specular": torch.randn(n, 45, generator=rng) * 0.1,
        f"{prefix}.features_albedo": torch.rand(n, 3, generator=rng),
    }


@pytest.fixture()
def rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture()
def random_quaternions(rng):
    q = torch.randn(_NUM_GAUSSIANS, 4, generator=rng)
    return torch.nn.functional.normalize(q, dim=1)


@pytest.fixture()
def sample_state_dict(rng):
    return _layer("model.gaussians_nodes.background", _NUM_GAUSSIANS, rng)


def test_affine_uint8_round_trip(rng):
    data = torch.randn(_NUM_GAUSSIANS, 8, generator=rng)
    quantized, col_min, col_scale = compact._quantize_affine_uint8(data)
    reconstructed = compact._dequant_affine_uint8(quantized, col_min, col_scale)

    col_range = (data.amax(dim=0) - data.amin(dim=0)).max().item()
    max_err = (data - reconstructed).abs().max().item()
    assert max_err < col_range / 255.0 + 1e-6


def test_symmetric_int8_round_trip(rng):
    data = torch.randn(_NUM_GAUSSIANS, 8, generator=rng)
    quantized, col_absmax = compact._quantize_band_symmetric(data)
    reconstructed = compact._dequant_band_symmetric(quantized, col_absmax)

    max_err = (data - reconstructed).abs().max().item()
    expected_step = col_absmax.max().item() / 127.0
    assert max_err < expected_step + 1e-6


def test_symmetric_int16_round_trip(rng):
    data = torch.randn(_NUM_GAUSSIANS, 8, generator=rng)
    quantized, col_absmax = compact._quantize_band_symmetric_int16(data)
    reconstructed = compact._dequant_band_symmetric_int16(quantized, col_absmax)

    assert quantized.dtype == torch.int16
    max_err = (data - reconstructed).abs().max().item()
    expected_step = col_absmax.max().item() / 32767.0
    assert max_err < expected_step + 1e-6


def test_affine_uint16_round_trip_and_serializes(rng):
    data = torch.randn(_NUM_GAUSSIANS, 8, generator=rng)
    quantized, col_min, col_scale = compact._quantize_affine_uint16(data)
    reconstructed = compact._dequant_affine_uint16(quantized, col_min, col_scale)

    assert quantized.dtype == torch.int16
    safetensors.torch.save({"x": quantized}, metadata={})
    col_range = (data.amax(dim=0) - data.amin(dim=0)).max().item()
    max_err = (data - reconstructed).abs().max().item()
    assert max_err < col_range / 65535.0 + 1e-6


@pytest.mark.parametrize("rotation_bits", [8, 16])
def test_smallest_three_quaternion_round_trip(random_quaternions, rotation_bits):
    prefix = "gaussians_nodes.background"
    tensors = {f"{prefix}.rotations": random_quaternions.clone()}
    compact._pack_rotations(tensors, rotation_bits=rotation_bits)

    dtype = torch.int16 if rotation_bits == 16 else torch.int8
    assert tensors[f"{prefix}.rotations"].dtype == dtype
    assert tensors[f"{prefix}.rotations"].shape == (_NUM_GAUSSIANS, 3)

    restored = compact.unpack(tensors, {compact._FORMAT_KEY: compact.FORMAT_VALUE})
    q = restored[f"model.{prefix}.rotations"]

    dots = (q * random_quaternions).sum(dim=1).abs()
    assert dots.min().item() > 0.999


def test_pack_unpack_round_trip(sample_state_dict):
    tensors, metadata = compact.pack(
        sample_state_dict,
        opacity_threshold=0.0,
        compression_level=0,
    )
    restored = compact.unpack(tensors, metadata)

    assert set(restored) == set(sample_state_dict)
    for key, original in sample_state_dict.items():
        assert key in restored, f"missing key: {key}"
        assert restored[key].shape == original.shape, f"shape mismatch for {key}"

    pos_key = "model.gaussians_nodes.background.positions"
    orig_order = np.lexsort(sample_state_dict[pos_key].numpy().T[::-1])
    rec_order = np.lexsort(restored[pos_key].numpy().T[::-1])

    for key, original in sample_state_dict.items():
        if not original.is_floating_point():
            continue
        orig_aligned = original[orig_order]
        rec_aligned = restored[key][rec_order]
        if key.endswith(".rotations"):
            dots = (orig_aligned * rec_aligned).sum(dim=1).abs()
            assert dots.min().item() > 0.999, f"{key}: min dot {dots.min():.4f}"
            continue
        tol = _round_trip_atol(key, orig_aligned)
        err = (orig_aligned - rec_aligned).abs().max().item()
        assert err < tol, f"{key}: max err {err:.4f} (tol {tol:.4f})"


def test_pack_preserves_camera_calib_embeds(sample_state_dict, rng):
    key = "model.calib.camera_view_geometry.embeds"
    state = {
        **sample_state_dict,
        key: torch.randn(6, 2, 7, generator=rng),
    }

    tensors, metadata = compact.pack(
        state,
        opacity_threshold=0.0,
        compression_level=9,
    )
    restored = compact.unpack(tensors, metadata)
    skipped = json.loads(metadata[compact._METADATA_SKIPPED_TENSORS])

    assert key.removeprefix("model.") not in skipped
    assert torch.equal(restored[key], state[key])


def test_pack_preserves_lidar_calib_embeds(sample_state_dict, rng):
    key = "model.calib.lidar_view_geometry.embeds"
    state = {
        **sample_state_dict,
        key: torch.randn(6, 2, 7, generator=rng),
    }

    tensors, metadata = compact.pack(
        state,
        opacity_threshold=0.0,
        compression_level=9,
    )
    restored = compact.unpack(tensors, metadata)
    skipped = json.loads(metadata[compact._METADATA_SKIPPED_TENSORS])

    assert key.removeprefix("model.") not in skipped
    assert torch.equal(restored[key], state[key])


def test_pack_unpack_round_trip_sh_compressed(sample_state_dict, rng):
    # L9 drops most PCA axes. This bound catches all-zero output and amplitude
    # blow-up without depending on a specific basis.
    tensors, metadata = compact.pack(
        sample_state_dict,
        opacity_threshold=0.0,
        compression_level=9,
    )
    restored = compact.unpack(tensors, metadata)

    for key, original in sample_state_dict.items():
        assert key in restored, f"missing key: {key}"
        assert restored[key].shape == original.shape, f"shape mismatch for {key}"

    sh_key = "model.gaussians_nodes.background.features_specular"
    pos_key = "model.gaussians_nodes.background.positions"
    orig_order = np.lexsort(sample_state_dict[pos_key].numpy().T[::-1])
    rec_order = np.lexsort(restored[pos_key].numpy().T[::-1])
    sh_orig = sample_state_dict[sh_key][orig_order]
    sh_rec = restored[sh_key][rec_order]
    err = (sh_rec - sh_orig).abs().max().item()
    assert err < sh_orig.abs().amax().item(), f"L9 SH error {err:.3f} exceeds bound"


def test_unpack_discovers_encoded_tensors_from_sidecars(sample_state_dict, monkeypatch):
    tensors, metadata = compact.pack(
        sample_state_dict,
        opacity_threshold=0.0,
        compression_level=9,
    )
    monkeypatch.setattr(compact, "_LAYER_PREFIXES", ("renamed.layer",))

    restored = compact.unpack(tensors, metadata)

    assert set(restored) == set(sample_state_dict)
    for key, original in sample_state_dict.items():
        assert restored[key].shape == original.shape
        if original.is_floating_point():
            assert restored[key].is_floating_point(), f"{key} stayed encoded as {restored[key].dtype}"


@pytest.mark.parametrize("level", [1, 8])
def test_pack_unpack_sh_levels_smoke(sample_state_dict, rng, level):
    # Cover mid-level SH paths. Shape and finite values are the contract.
    tensors, metadata = compact.pack(
        sample_state_dict,
        opacity_threshold=0.0,
        compression_level=level,
    )
    restored = compact.unpack(tensors, metadata)
    sh = restored["model.gaussians_nodes.background.features_specular"]
    assert sh.shape == sample_state_dict["model.gaussians_nodes.background.features_specular"].shape
    assert torch.isfinite(sh).all()


# pack strips `model.` from tensor keys. unpack restores it.
_ALBEDO_KEY_FULL = "model.gaussians_nodes.background.features_albedo"
_ALBEDO_KEY_SHORT = _ALBEDO_KEY_FULL.removeprefix("model.")


@pytest.mark.parametrize("level", [0, 2, 3])
def test_albedo_levels_restore_finite_shape(sample_state_dict, level):
    tensors, metadata = compact.pack(
        sample_state_dict,
        opacity_threshold=0.0,
        compression_level=level,
    )
    restored = compact.unpack(tensors, metadata)
    assert restored[_ALBEDO_KEY_FULL].shape == sample_state_dict[_ALBEDO_KEY_FULL].shape
    assert torch.isfinite(restored[_ALBEDO_KEY_FULL]).all()


def test_level_endpoints_use_expected_storage_modes(sample_state_dict):
    prefix = "gaussians_nodes.background"
    l0, _ = compact.pack(
        sample_state_dict,
        opacity_threshold=0.0,
        compression_level=0,
    )
    l9, _ = compact.pack(
        sample_state_dict,
        opacity_threshold=0.0,
        compression_level=9,
    )

    assert l0[f"{prefix}.positions"].dtype == torch.float32
    assert f"{prefix}._voxel_centroids" not in l0
    assert l0[f"{prefix}.rotations"].dtype == torch.int16
    assert l0[f"{prefix}.densities"].dtype == torch.int16
    assert l0[f"{prefix}.scales"].dtype == torch.int16
    assert l0[_ALBEDO_KEY_SHORT].dtype == torch.float16

    assert l9[f"{prefix}.positions"].dtype == torch.float16
    assert f"{prefix}._voxel_centroids" in l9
    assert f"{prefix}._voxel_ranges" in l9
    assert l9[f"{prefix}.rotations"].dtype == torch.int8
    assert l9[f"{prefix}.densities"].dtype == torch.int8
    assert l9[f"{prefix}.scales"].dtype == torch.uint8
    assert _ALBEDO_KEY_SHORT not in l9


def test_aggressive_level_is_smaller_than_conservative_level(sample_state_dict):
    l0, _ = compact.pack(sample_state_dict, opacity_threshold=0.0, compression_level=0)
    l9, _ = compact.pack(sample_state_dict, opacity_threshold=0.0, compression_level=9)

    assert sum(t.nbytes for t in l9.values()) < sum(t.nbytes for t in l0.values())


@pytest.mark.parametrize("albedo_shape", [(_NUM_GAUSSIANS, 1, 3), (_NUM_GAUSSIANS, 5, 3)])
def test_albedo_with_middle_dims_round_trips(sample_state_dict, rng, albedo_shape):
    # Dynamic layers store albedo as (N, M, 3) Fourier coefficients; this
    # test checks shape preservation and per-triple error tolerance.
    state = dict(sample_state_dict)
    state[_ALBEDO_KEY_FULL] = torch.rand(albedo_shape, generator=rng)

    tensors, metadata = compact.pack(
        state,
        opacity_threshold=0.0,
        compression_level=3,
    )

    restored = compact.unpack(tensors, metadata)
    assert restored[_ALBEDO_KEY_FULL].shape == albedo_shape

    pos_key = "model.gaussians_nodes.background.positions"
    orig_order = np.lexsort(state[pos_key].numpy().T[::-1])
    rec_order = np.lexsort(restored[pos_key].numpy().T[::-1])
    orig = state[_ALBEDO_KEY_FULL][orig_order]
    rec = restored[_ALBEDO_KEY_FULL][rec_order]
    assert (orig - rec).abs().max().item() < _round_trip_atol(_ALBEDO_KEY_FULL, orig)


def test_page_dynamic_round_trip_preserves_cuboid_ids(rng):
    # Dynamic cuboid ids pack through row-index pages and round-trip with the
    # same Morton row ordering.
    prefix = "model.gaussians_nodes.dynamic_rigids"
    short_prefix = prefix.removeprefix("model.")
    id_key = f"{prefix}.gaussian_cuboid_ids"
    short_id_key = id_key.removeprefix("model.")
    n = 2048
    state = {
        f"{prefix}.positions": torch.randn(n, 3, generator=rng),
        f"{prefix}.densities": torch.randn(n, 1, generator=rng),
        f"{prefix}.scales": torch.rand(n, 3, generator=rng) + 0.01,
        f"{prefix}.rotations": torch.nn.functional.normalize(torch.randn(n, 4, generator=rng), dim=1),
        f"{prefix}.features_specular": torch.randn(n, 45, generator=rng) * 0.1,
        f"{prefix}.features_albedo": torch.rand(n, 3, generator=rng),
        id_key: torch.arange(n, dtype=torch.int64),
    }
    expected = {
        f"{short_prefix}.positions": state[f"{prefix}.positions"].clone(),
        short_id_key: state[id_key].clone(),
    }
    compact._morton_sort_layer(expected, short_prefix)

    tensors, metadata = compact.pack(state, opacity_threshold=0.0, compression_level=5)
    assert torch.equal(tensors[short_id_key], expected[short_id_key])

    restored = compact.unpack(tensors, metadata)
    assert torch.equal(restored[id_key], expected[short_id_key])


def test_pack_is_deterministic(sample_state_dict):
    def _pack():
        return compact.pack(
            sample_state_dict,
            opacity_threshold=0.0,
            compression_level=9,
        )

    tensors_a, metadata_a = _pack()
    tensors_b, metadata_b = _pack()

    assert metadata_a == metadata_b
    assert tensors_a.keys() == tensors_b.keys()
    for key in tensors_a:
        assert torch.equal(tensors_a[key], tensors_b[key]), f"non-deterministic: {key}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"opacity_threshold": -0.1},
        {"opacity_threshold": False},
        {"opacity_threshold": 1.0},  # exclusive upper bound
        {"compression_level": 10},  # above 9
        {"compression_level": 2.5},  # not an int
        {"preserve_extra_signal_tensors": ("not_a_signal",)},
    ],
)
def test_pack_invalid_args(sample_state_dict, kwargs):
    with pytest.raises(ValueError):
        compact.pack(sample_state_dict, **kwargs)


def test_pack_rejects_interpolated_deformation_table_state(sample_state_dict):
    """Compact export must not reorder Gaussians without reordering table rows."""

    state = dict(sample_state_dict)
    state["model.gaussians_nodes.dynamic_deformables.deformation_table.global_row_ids"] = torch.arange(4)

    with pytest.raises(ValueError, match="does not support interpolated deformation tables"):
        compact.pack(state, opacity_threshold=0.0, compression_level=0)


def test_morton_sort_rejects_nonpositive_voxel_size():
    tensors = {"layer.positions": torch.zeros(1, 3)}
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        compact._morton_sort_layer(tensors, "layer", voxel_size=0.0)


def test_morton_sort_rejects_nonfinite_positions():
    tensors = {"layer.positions": torch.tensor([[0.0, 0.0, float("nan")]])}
    with pytest.raises(ValueError, match="non-finite"):
        compact._morton_sort_layer(tensors, "layer")


def test_morton_sort_uses_two_segments_for_large_voxel_coordinates():
    extent = float((1 << 21) * VOXEL_SIZE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [extent, 0.0, 0.0],
            [VOXEL_SIZE, 0.0, 0.0],
            [extent + VOXEL_SIZE, 0.0, 0.0],
        ]
    )
    tensors = {
        "layer.positions": positions.clone(),
        "layer.gaussian_cuboid_ids": torch.arange(len(positions)),
    }

    compact._morton_sort_layer(tensors, "layer")

    assert tensors["layer.gaussian_cuboid_ids"].tolist() == [0, 2, 1, 3]
    assert tensors["layer._voxel_ranges"].tolist() == [0, 1, 2, 3, 4]
    compact._unmorton_layers(tensors)
    assert torch.equal(tensors["layer.positions"], positions[[0, 2, 1, 3]])


def test_morton_sort_rejects_coordinate_above_two_segment_limit():
    extent = float((1 << 42) * VOXEL_SIZE)
    tensors = {"layer.positions": torch.tensor([[0.0, 0.0, 0.0], [extent, 0.0, 0.0]])}
    with pytest.raises(ValueError, match="exceeds 42-bit Morton limit"):
        compact._morton_sort_layer(tensors, "layer")


def test_pack_can_morton_sort_fp32_positions(sample_state_dict):
    short_prefix = "gaussians_nodes.background"
    pos_key = f"{short_prefix}.positions"
    expected = {k.removeprefix("model."): v.clone() for k, v in sample_state_dict.items()}
    compact._morton_sort_layer(expected, short_prefix, position_mode="fp32_sorted")

    tensors, metadata = compact.pack(
        sample_state_dict,
        opacity_threshold=0.0,
        compression_level=0,
    )

    assert tensors[pos_key].dtype == torch.float32
    assert f"{short_prefix}._voxel_centroids" not in tensors
    assert f"{short_prefix}._voxel_ranges" not in tensors
    assert torch.equal(tensors[pos_key], expected[pos_key])

    restored = compact.unpack(tensors, metadata)
    assert torch.equal(restored[f"model.{pos_key}"], expected[pos_key])


def test_pack_does_not_mutate_input(sample_state_dict):
    sample_state_dict["model.cfg"] = {"layers": [1, 2, 3]}
    snapshot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample_state_dict.items()}
    snapshot_keys = list(sample_state_dict.keys())

    compact.pack(sample_state_dict, opacity_threshold=0.0, compression_level=0)

    assert list(sample_state_dict.keys()) == snapshot_keys
    for key, original in snapshot.items():
        if isinstance(original, torch.Tensor):
            assert torch.equal(sample_state_dict[key], original), f"pack mutated {key}"
        else:
            assert sample_state_dict[key] == original, f"pack mutated non-tensor {key}"


def test_pack_culls_low_opacity_gaussians(rng):
    n_keep = _NUM_GAUSSIANS // 2
    prefix = "model.gaussians_nodes.background"
    state = _layer(prefix, _NUM_GAUSSIANS, rng) | {
        f"{prefix}.densities": torch.cat(
            [
                torch.full((n_keep, 1), 10.0),
                torch.full((_NUM_GAUSSIANS - n_keep, 1), -10.0),
            ]
        ),
    }

    tensors, metadata = compact.pack(state, opacity_threshold=0.5, compression_level=0)
    restored = compact.unpack(tensors, metadata)

    rec_pos = restored[f"{prefix}.positions"]
    rec_density = restored[f"{prefix}.densities"]
    assert rec_pos.shape == (n_keep, 3)
    assert rec_density.shape == (n_keep, 1)
    assert (torch.sigmoid(rec_density) >= 0.5).all()


def test_pack_culls_low_opacity_with_skipped_buffer(rng):
    # Skipped per-Gaussian tensors remain row-aligned after culling.
    n_keep = _NUM_GAUSSIANS // 2
    prefix = "model.gaussians_nodes.background"
    state = _layer(prefix, _NUM_GAUSSIANS, rng) | {
        f"{prefix}.densities": torch.cat(
            [
                torch.full((n_keep, 1), 10.0),
                torch.full((_NUM_GAUSSIANS - n_keep, 1), -10.0),
            ]
        ),
        f"{prefix}.texture_grads": torch.zeros(_NUM_GAUSSIANS, 4),
    }

    tensors, metadata = compact.pack(state, opacity_threshold=0.5, compression_level=0)
    restored = compact.unpack(tensors, metadata)

    rec_tex = restored[f"{prefix}.texture_grads"]
    assert rec_tex.shape == (n_keep, 4), f"skipped buffer shape drifted: {rec_tex.shape}"


def test_pack_culls_with_scalar_skipped_tensor(rng):
    prefix = "model.gaussians_nodes.background"
    state = _layer(prefix, _NUM_GAUSSIANS, rng) | {
        f"{prefix}.densities": torch.cat(
            [
                torch.full((_NUM_GAUSSIANS // 2, 1), 10.0),
                torch.full((_NUM_GAUSSIANS // 2, 1), -10.0),
            ]
        ),
        f"{prefix}.n_grad_updates": torch.tensor(40001),
    }

    tensors, metadata = compact.pack(state, opacity_threshold=0.5, compression_level=0)
    restored = compact.unpack(tensors, metadata)

    rec = restored[f"{prefix}.n_grad_updates"]
    assert rec.shape == ()
    assert rec.dtype == torch.int64
    assert rec.item() == 0


def test_pack_drops_extra_signals_and_restores_zero_shapes(rng):
    n_keep = _NUM_GAUSSIANS // 2
    prefix = "model.gaussians_nodes.background"
    short_prefix = prefix.removeprefix("model.")
    keep_density = torch.full((n_keep, 1), 10.0)
    drop_density = torch.full((_NUM_GAUSSIANS - n_keep, 1), -10.0)
    state = _layer(prefix, _NUM_GAUSSIANS, rng) | {
        f"{prefix}.densities": torch.cat([keep_density, drop_density]),
        f"{prefix}.extra_signal": torch.arange(_NUM_GAUSSIANS, dtype=torch.float32).unsqueeze(1),
        f"{prefix}.camera_extra_signal": torch.arange(_NUM_GAUSSIANS * 2, dtype=torch.float32).reshape(
            _NUM_GAUSSIANS, 2
        ),
        f"{prefix}.lidar_extra_signal": torch.arange(_NUM_GAUSSIANS * 3, dtype=torch.float32).reshape(
            _NUM_GAUSSIANS, 3
        ),
    }

    tensors, metadata = compact.pack(state, opacity_threshold=0.5, compression_level=0)
    restored = compact.unpack(tensors, metadata)
    skipped = json.loads(metadata[compact._METADATA_SKIPPED_TENSORS])

    for suffix, width in (
        ("extra_signal", 1),
        ("camera_extra_signal", 2),
        ("lidar_extra_signal", 3),
    ):
        short_key = f"{short_prefix}.{suffix}"
        full_key = f"model.{short_key}"
        assert short_key not in tensors
        assert skipped[short_key] == [[n_keep, width], "float32"]
        assert restored[full_key].shape == (n_keep, width)
        assert restored[full_key].dtype == torch.float32
        assert not restored[full_key].any()


def test_pack_can_preserve_lidar_extra_signal(rng):
    n = 4
    n_keep = 2
    prefix = "model.gaussians_nodes.background"
    short_prefix = prefix.removeprefix("model.")
    lidar_signal = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    state = _layer(prefix, n, rng) | {
        f"{prefix}.positions": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        f"{prefix}.densities": torch.tensor([[10.0], [10.0], [-10.0], [-10.0]]),
        f"{prefix}.extra_signal": torch.ones(n, 1),
        f"{prefix}.camera_extra_signal": torch.ones(n, 2),
        f"{prefix}.lidar_extra_signal": lidar_signal,
    }

    tensors, metadata = compact.pack(
        state,
        opacity_threshold=0.5,
        compression_level=0,
        preserve_extra_signal_tensors=("lidar_extra_signal",),
    )
    restored = compact.unpack(tensors, metadata)
    skipped = json.loads(metadata[compact._METADATA_SKIPPED_TENSORS])

    assert json.loads(metadata[compact._METADATA_PRESERVED_EXTRA_SIGNAL_TENSORS]) == ["lidar_extra_signal"]
    assert f"{short_prefix}.lidar_extra_signal" not in skipped
    assert f"{short_prefix}.extra_signal" in skipped
    assert f"{short_prefix}.camera_extra_signal" in skipped
    assert torch.equal(restored[f"{prefix}.lidar_extra_signal"], lidar_signal[:n_keep])


def test_pack_drops_fully_culled_layer(rng):
    prefix = "model.gaussians_nodes.background"
    state = _layer(prefix, _NUM_GAUSSIANS, rng) | {
        f"{prefix}.densities": torch.full((_NUM_GAUSSIANS, 1), -10.0),
    }

    tensors, metadata = compact.pack(state, opacity_threshold=0.5, compression_level=0)
    skipped = json.loads(metadata[compact._METADATA_SKIPPED_TENSORS])

    short_prefix = prefix.removeprefix("model.")
    layer_keys = [k for k in skipped if k.startswith(short_prefix + ".")]
    assert layer_keys, "fully-culled layer was not stashed in skipped_tensors"
    assert not any(k.startswith(short_prefix) for k in tensors), (
        f"culled layer leaked tensors: {[k for k in tensors if k.startswith(short_prefix)]}"
    )

    restored = compact.unpack(tensors, metadata)
    rec_pos = restored[f"{prefix}.positions"]
    assert rec_pos.shape[0] == 0
    assert rec_pos.abs().sum().item() == 0.0


def test_pack_unpack_multi_layer(rng):
    n_per_layer = 256
    prefixes = (
        "model.gaussians_nodes.background",
        "model.gaussians_nodes.road",
        "model.gaussians_nodes.dynamic_rigids",
        "model.gaussians_nodes.dynamic_deformables",
    )
    state = {}
    for prefix in prefixes:
        state.update(_layer(prefix, n_per_layer, rng))

    tensors, metadata = compact.pack(state, opacity_threshold=0.0, compression_level=0)
    restored = compact.unpack(tensors, metadata)

    for prefix in prefixes:
        for key in (".positions", ".densities", ".scales", ".rotations"):
            full = prefix + key
            assert restored[full].shape[0] == n_per_layer, f"{full} row count drifted"


def test_pack_unpack_tuple_becomes_list():
    # Non-tensor state JSON-round-trips. Tuples therefore come back as lists.
    state = {"model.cfg": {"layers": ("a", "b")}}
    tensors, metadata = compact.pack(state, opacity_threshold=0.0, compression_level=0)
    restored = compact.unpack(tensors, metadata)
    assert restored["model.cfg"] == {"layers": ["a", "b"]}


def test_pack_rotations_handles_axis_aligned(rng):
    # Axis-aligned quaternions hit the sign-of-zero edge in smallest-three.
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    prefix = "model.gaussians_nodes.background"
    state = _layer(prefix, quats.shape[0], rng)
    state[f"{prefix}.rotations"] = quats

    tensors, metadata = compact.pack(state, opacity_threshold=0.0, compression_level=0)
    restored = compact.unpack(tensors, metadata)

    pos_key = f"{prefix}.positions"
    orig_order = np.lexsort(state[pos_key].numpy().T[::-1])
    rec_order = np.lexsort(restored[pos_key].numpy().T[::-1])
    q = restored[f"{prefix}.rotations"][rec_order]
    dots = (q * quats[orig_order]).sum(dim=1).abs()
    assert dots.min().item() > 0.99, f"axis-aligned dot dropped to {dots.min():.4f}"


def test_quantize_constant_column_returns_zero():
    # Symmetric: a zero column stays zero and never becomes NaN.
    zeros = torch.zeros(10, 3)
    q_sym, absmax = compact._quantize_band_symmetric(zeros)
    rec_sym = compact._dequant_band_symmetric(q_sym, absmax)
    assert rec_sym.abs().max().item() == 0.0
    assert not torch.isnan(rec_sym).any()

    # Affine: a constant nonzero column round-trips through col_min.
    constant = torch.full((10, 3), 5.0)
    q_aff, col_min, col_scale = compact._quantize_affine_uint8(constant)
    rec_aff = compact._dequant_affine_uint8(q_aff, col_min, col_scale)
    assert torch.allclose(rec_aff, constant, atol=1e-5)


def test_pack_strips_deform_network_when_dynamic_layer_culled(rng):
    # Force a fully culled dynamic layer so pack must strip the deform network.
    prefix = "model.gaussians_nodes.dynamic_rigids"
    weight = torch.randn(8, 8, generator=rng)
    state = _layer(prefix, _NUM_GAUSSIANS, rng) | {
        f"{prefix}.densities": torch.full((_NUM_GAUSSIANS, 1), -10.0),
        f"{prefix}.deform_network.weight": weight,
        f"{prefix}._extra_state": {"use_deform_network": True},
    }

    _, metadata = compact.pack(state, opacity_threshold=0.5, compression_level=0)

    extra = json.loads(metadata[compact._METADATA_NON_TENSOR_STATE])["gaussians_nodes.dynamic_rigids._extra_state"]
    assert extra["use_deform_network"] is False
    skipped = json.loads(metadata[compact._METADATA_SKIPPED_TENSORS])
    assert skipped["gaussians_nodes.dynamic_rigids.deform_network.weight"] == [
        list(weight.shape),
        "float32",
    ]

    restored = compact.unpack({}, metadata)
    rec_weight = restored[f"{prefix}.deform_network.weight"]
    assert rec_weight.shape == weight.shape
    assert rec_weight.abs().sum().item() == 0.0


@pytest.mark.parametrize("extra_state", [None, [1, 2, 3], "not_a_dict"])
def test_pack_raises_when_dynamic_layer_culled_without_extra_state(rng, extra_state):
    # The flag must be recorded when deform weights are stripped.
    # If the slot is missing, raising is safer than shipping broken state.
    prefix = "model.gaussians_nodes.dynamic_rigids"
    state = _layer(prefix, _NUM_GAUSSIANS, rng) | {
        f"{prefix}.densities": torch.full((_NUM_GAUSSIANS, 1), -10.0),
        f"{prefix}.deform_network.weight": torch.zeros(2, 2),
    }
    if extra_state is not None:
        state[f"{prefix}._extra_state"] = extra_state

    with pytest.raises(ValueError, match=r"cannot disable deform network"):
        compact.pack(state, opacity_threshold=0.5, compression_level=0)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {compact._FORMAT_KEY: "nurec.gaussians.compact/v999"},
        {compact._FORMAT_KEY: "some.other.codec/v1"},
    ],
    ids=["missing", "wrong_version", "wrong_producer"],
)
def test_unpack_rejects_unknown_format(metadata):
    with pytest.raises(ValueError, match=r"unsupported compact format"):
        compact.unpack({}, metadata)


def test_unpack_rejects_corrupt_metadata():
    metadata = {
        compact._FORMAT_KEY: compact.FORMAT_VALUE,
        compact._METADATA_SKIPPED_TENSORS: json.dumps({"foo": [[1], "not_a_dtype"]}),
    }
    with pytest.raises(ValueError, match=r"unknown dtype in metadata for"):
        compact.unpack({}, metadata)


def test_unpack_rejects_density_sidecar_without_base():
    tensors = {"foo.densities_absmax": torch.ones(1)}
    metadata = {compact._FORMAT_KEY: compact.FORMAT_VALUE}

    with pytest.raises(ValueError, match=r"densities_absmax.*base tensor"):
        compact.unpack(tensors, metadata)


def test_unpack_rejects_scale_sidecar_without_companion():
    tensors = {
        "foo.scales": torch.zeros(1, 3, dtype=torch.uint8),
        "foo.scales_min": torch.zeros(3),
    }
    metadata = {compact._FORMAT_KEY: compact.FORMAT_VALUE}

    with pytest.raises(ValueError, match=r"scales.*scales_scale"):
        compact.unpack(tensors, metadata)


def test_unpack_rejects_rotation_sidecar_without_base():
    tensors = {"foo.rotations_idx": torch.zeros(1, dtype=torch.uint8)}
    metadata = {compact._FORMAT_KEY: compact.FORMAT_VALUE}

    with pytest.raises(ValueError, match=r"rotations_idx.*base tensor"):
        compact.unpack(tensors, metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        {
            compact._FORMAT_KEY: compact.FORMAT_VALUE,
            compact._METADATA_SKIPPED_TENSORS: json.dumps({"foo": [[1], "float32"]}),
        },
        {
            compact._FORMAT_KEY: compact.FORMAT_VALUE,
            compact._METADATA_NON_TENSOR_STATE: json.dumps({"foo": "bar"}),
        },
    ],
    ids=["skipped_tensor", "non_tensor"],
)
def test_unpack_rejects_duplicate_reconstructed_key(metadata):
    with pytest.raises(ValueError, match=r"duplicate.*foo"):
        compact.unpack({"foo": torch.ones(1)}, metadata)


_PREFIX = "model.gaussians_nodes.background"


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda s: {**s, f"{_PREFIX}.features_specular": s[f"{_PREFIX}.features_specular"][:, :44]},
            r"features_specular",
        ),
        (
            lambda s: {**s, f"{_PREFIX}.features_specular": s[f"{_PREFIX}.features_specular"].half()},
            r"features_specular.*float32",
        ),
        (
            lambda s: {**s, f"{_PREFIX}.features_specular": torch.zeros(45)},
            r"features_specular",
        ),
        (
            lambda s: {**s, f"{_PREFIX}.densities": torch.zeros(s[f"{_PREFIX}.densities"].shape[0], 2)},
            r"densities",
        ),
        (
            lambda s: {
                **s,
                f"{_PREFIX}.densities": torch.zeros(s[f"{_PREFIX}.densities"].shape, dtype=torch.int16),
            },
            r"densities.*floating-point",
        ),
        (
            lambda s: {**s, f"{_PREFIX}.rotations": s[f"{_PREFIX}.rotations"][:, :3]},
            r"rotations",
        ),
        (
            lambda s: {**s, f"{_PREFIX}.rotations": torch.zeros(4)},
            r"rotations",
        ),
        (
            lambda s: {
                **s,
                f"{_PREFIX}.features_albedo": torch.zeros(s[f"{_PREFIX}.features_albedo"].shape[0], 4),
            },
            r"features_albedo",
        ),
    ],
    ids=[
        "sh_width",
        "sh_dtype",
        "sh_rank",
        "density_width",
        "density_dtype",
        "quaternion_width",
        "quaternion_rank",
        "albedo_channels",
    ],
)
def test_pack_rejects_malformed_schema(sample_state_dict, mutate, match):
    state = mutate(sample_state_dict)
    with pytest.raises(ValueError, match=match):
        compact.pack(state, opacity_threshold=0.0, compression_level=9)


def test_pack_rejects_known_layer_row_mismatch(sample_state_dict):
    state = {
        **sample_state_dict,
        f"{_PREFIX}.features_albedo": torch.zeros(_NUM_GAUSSIANS + 1, 3),
    }
    with pytest.raises(ValueError, match=r"features_albedo"):
        compact.pack(state, compression_level=0)


def test_pack_rejects_unknown_direct_per_gaussian_layer_tensor(sample_state_dict):
    state = {
        **sample_state_dict,
        f"{_PREFIX}.new_signal": torch.arange(_NUM_GAUSSIANS),
    }
    with pytest.raises(ValueError, match=r"new_signal.*per-Gaussian tensor set"):
        compact.pack(state, compression_level=0)


def test_pack_malformed_dynamic_layer_missing_positions(sample_state_dict):
    prefix = "model.gaussians_nodes.dynamic_rigids"
    state = dict(sample_state_dict)
    state[f"{prefix}.deform_network.weight"] = torch.zeros(4, 4)
    with pytest.raises(ValueError, match=r"malformed input.*dynamic_rigids.*positions"):
        compact.pack(state, opacity_threshold=0.0, compression_level=0)


def test_pack_malformed_empty_dynamic_layer_missing_positions():
    prefix = "model.gaussians_nodes.dynamic_rigids"
    state = {f"{prefix}.densities": torch.empty(0, 1)}
    with pytest.raises(ValueError, match=r"malformed input.*dynamic_rigids.*positions"):
        compact.pack(state, compression_level=0)


def test_pack_absent_dynamic_layer_is_noop(sample_state_dict):
    tensors, _ = compact.pack(sample_state_dict, opacity_threshold=0.0, compression_level=0)
    assert not any("dynamic_rigids" in k for k in tensors)
    assert not any("dynamic_deformables" in k for k in tensors)


def test_pack_preserves_unknown_feature_like_tensor():
    state = {"model.aux.features_albedo": torch.rand(8, 3)}

    tensors, metadata = compact.pack(state, opacity_threshold=0.0, compression_level=1)

    assert "aux.features_albedo" in tensors
    assert not any(k.startswith("aux.features_albedo_") for k in tensors)

    restored = compact.unpack(tensors, metadata)
    assert torch.equal(restored["model.aux.features_albedo"], state["model.aux.features_albedo"])


def test_pack_preserves_unknown_suffix_tensors(rng):
    state = {
        "model.aux.densities": torch.randn(8, 1, generator=rng),
        "model.aux.scales": torch.rand(8, 3, generator=rng),
        "model.aux.rotations": torch.nn.functional.normalize(torch.randn(8, 4, generator=rng), dim=1),
        "model.aux.features_specular": torch.randn(8, 45, generator=rng),
    }

    tensors, metadata = compact.pack(state, opacity_threshold=0.0, compression_level=0)
    restored = compact.unpack(tensors, metadata)

    for key, original in state.items():
        short_key = key.removeprefix("model.")
        assert torch.equal(tensors[short_key], original)
        assert torch.equal(restored[key], original)


def test_pack_preserves_gaussian_cuboid_ids_passenger(sample_state_dict):
    # gaussian_cuboid_ids preserve row alignment, dtype, and values.
    short = f"{_PREFIX.removeprefix('model.')}.gaussian_cuboid_ids"
    state = {**sample_state_dict, f"{_PREFIX}.gaussian_cuboid_ids": torch.arange(_NUM_GAUSSIANS, dtype=torch.int32)}
    expected = {
        f"{_PREFIX.removeprefix('model.')}.positions": state[f"{_PREFIX}.positions"].clone(),
        short: state[f"{_PREFIX}.gaussian_cuboid_ids"].clone(),
    }
    compact._morton_sort_layer(expected, _PREFIX.removeprefix("model."))

    tensors, metadata = compact.pack(state, opacity_threshold=0.0, compression_level=0)

    assert tensors[short].dtype == state[f"{_PREFIX}.gaussian_cuboid_ids"].dtype
    assert tensors[short].shape == state[f"{_PREFIX}.gaussian_cuboid_ids"].shape
    assert torch.equal(tensors[short], expected[short])

    restored = compact.unpack(tensors, metadata)
    assert torch.equal(restored[f"model.{short}"], expected[short])


def test_pack_does_not_cull_or_sort_non_gaussian_layer_tensor(rng):
    n_keep = _NUM_GAUSSIANS // 2
    prefix = "model.gaussians_nodes.dynamic_rigids"
    short_weight = f"{prefix.removeprefix('model.')}.deform_network.weight"
    weight = torch.arange(_NUM_GAUSSIANS * 2, dtype=torch.float32).reshape(_NUM_GAUSSIANS, 2)
    state = _layer(prefix, _NUM_GAUSSIANS, rng) | {
        f"{prefix}.densities": torch.cat(
            [
                torch.full((n_keep, 1), 10.0),
                torch.full((_NUM_GAUSSIANS - n_keep, 1), -10.0),
            ]
        ),
        f"{prefix}.deform_network.weight": weight,
        f"{prefix}._extra_state": {"use_deform_network": True},
    }

    tensors, _ = compact.pack(state, opacity_threshold=0.5, compression_level=0)

    assert tensors[short_weight].shape == weight.shape
    assert torch.equal(tensors[short_weight].float(), weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pack_round_trip_on_cuda(rng):
    prefix = "model.gaussians_nodes.background"
    state = {k: v.cuda() for k, v in _layer(prefix, _NUM_GAUSSIANS, rng).items()}

    def _pack():
        return compact.pack(
            state,
            opacity_threshold=0.0,
            compression_level=9,
        )

    tensors_a, _ = _pack()
    tensors_b, metadata = _pack()

    assert tensors_a.keys() == tensors_b.keys()
    for key in tensors_a:
        assert torch.equal(tensors_a[key], tensors_b[key]), f"non-deterministic on CUDA: {key}"

    restored = compact.unpack(tensors_b, metadata)
    for key, tensor in restored.items():
        assert tensor.device.type == "cuda", f"{key} fell off CUDA: device={tensor.device}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pack_deterministic_under_strict_mode(rng):
    # CUBLAS_WORKSPACE_CONFIG must be set before Python starts. The Bazel test
    # target provides it; setting it here would be too late for cuBLAS.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":16:8", ":4096:8"}:
        pytest.skip("CUBLAS_WORKSPACE_CONFIG must be set before Python starts")
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        prefix = "model.gaussians_nodes.background"
        state = {k: v.cuda() for k, v in _layer(prefix, _NUM_GAUSSIANS, rng).items()}

        def _pack():
            return compact.pack(
                state,
                opacity_threshold=0.0,
                compression_level=9,
            )

        tensors_a, _ = _pack()
        tensors_b, _ = _pack()
        for key in tensors_a:
            assert torch.equal(tensors_a[key], tensors_b[key]), f"nondeterministic: {key}"
    finally:
        torch.use_deterministic_algorithms(False)
