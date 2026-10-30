# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nurec.instant_nurec.config.predict import PrimitivePLYExportConfig
from nurec.instant_nurec.predict.export_ply import (
    build_ply_gaussian_loader,
    export_kelvin_ply,
    export_ply,
    get_gaussian_shape_pruning_mask,
)
from nurec.instant_nurec.primitives.celsius_primitive import CelsiusInstantNuRecPrimitive
from nurec.instant_nurec.primitives.kelvin_primitive import (
    KelvinInstantNuRecPrimitive,
    KelvinSemanticClass,
    KelvinStaticLayer,
)
from nurec.models.gaussians.utils import RGB2SH, PLYGaussianLoader


def _make_default_export_config(**overrides: Any) -> PrimitivePLYExportConfig:
    """Create a PrimitivePLYExportConfig with permissive defaults for testing."""
    defaults: dict[str, Any] = dict(
        enabled=True,
        density_activation="sigmoid",
        scale_activation="exp",
        color_mode="sh",
        apply_affine_mtx=False,
        falloff_sigma_timestamp_us=None,
        minimum_density=None,
        minimum_scale=None,
        minimum_surface_area=None,
        maximum_velocity=None,
        maximum_sky_mask=None,
    )
    defaults.update(overrides)
    return PrimitivePLYExportConfig(**defaults)


class TestGetGaussianShapePruningMask:
    """Tests for get_gaussian_shape_pruning_mask."""

    def test_no_filters(self):
        """With all thresholds None, all finite gaussians pass."""
        config = _make_default_export_config()
        densities = torch.tensor([[0.5], [0.8], [0.1]])
        scales = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.01, 0.01, 0.01]])
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask.shape == (3, 1)
        assert mask.all()

    def test_removes_non_finite_densities(self):
        """Non-finite (NaN, inf) densities are always filtered."""
        config = _make_default_export_config()
        densities = torch.tensor([[0.5], [float("nan")], [float("inf")]])
        scales = torch.tensor([[0.1, 0.1, 0.1]] * 3)
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask[:, 0].tolist() == [True, False, False]

    def test_minimum_density_filter(self):
        """Gaussians below minimum_density are filtered."""
        config = _make_default_export_config(minimum_density=0.5)
        densities = torch.tensor([[0.3], [0.5], [0.8]])
        scales = torch.tensor([[0.1, 0.1, 0.1]] * 3)
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask[:, 0].tolist() == [False, True, True]

    def test_minimum_scale_filter(self):
        """Gaussians with any scale dimension below threshold are filtered."""
        config = _make_default_export_config(minimum_scale=0.05)
        densities = torch.tensor([[1.0], [1.0]])
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.01, 0.1]])
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask[:, 0].tolist() == [True, False]

    def test_minimum_surface_area_filter(self):
        """Gaussians with surface area below threshold are filtered."""
        config = _make_default_export_config(minimum_surface_area=0.1)
        densities = torch.tensor([[1.0], [1.0]])
        # surface_area = x^2 + y^2 + z^2: [0.03, 0.12]
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        assert mask[:, 0].tolist() == [False, True]

    def test_combined_filters(self):
        """Multiple filters are applied together (AND logic)."""
        config = _make_default_export_config(minimum_density=0.5, minimum_scale=0.05)
        densities = torch.tensor([[0.8], [0.3], [0.8]])
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.01, 0.1, 0.1]])
        mask = get_gaussian_shape_pruning_mask(config, densities, scales)
        # Only first passes both filters
        assert mask[:, 0].tolist() == [True, False, False]


class TestInMemoryPLYGaussianPayload:
    @staticmethod
    def _make_celsius_primitive() -> CelsiusInstantNuRecPrimitive:
        n = 5
        return CelsiusInstantNuRecPrimitive(
            positions=torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 2.0], [9.0, 9.0, 9.0]]
            ),
            rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
            scales=torch.tensor([[0.1, 0.2, 0.4], [0.2, 0.3, 0.5], [0.25, 0.5, 1.0], [0.3, 0.6, 1.2], [0.4, 0.8, 1.6]]),
            densities=torch.tensor([[0.8], [0.2], [0.7], [0.6], [float("nan")]]),
            rgb=torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8], [0.8, 0.9, 1.0]]),
            timestamps_us=torch.arange(n, dtype=torch.int64).to(torch.uint64).unsqueeze(-1),
            forward_speed_mps=None,
            falloff_sigma=None,
            dynamic_bbox_mask=torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0]]),
            road_mask=torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0]]),
            sky_mask=torch.tensor([[0.0], [0.0], [0.75], [0.0], [0.0]]),
            sky_token=None,
            sky_head=None,
            sky_rotation=None,
            affine_matrix=None,
            affine_bias=None,
            gaussians_renderer=None,
        )

    @staticmethod
    def _world_transform() -> torch.Tensor:
        return torch.tensor(
            [
                [0.0, -1.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

    def test_build_payload_matches_disk_reload_and_exact_conversion(self):
        primitive = self._make_celsius_primitive()
        config = _make_default_export_config(minimum_density=0.5)
        rig = SimpleNamespace(T_world_base=self._world_transform())

        in_memory = build_ply_gaussian_loader(config, primitive, rig, device="cpu")  # type: ignore[arg-type]

        expected_positions = torch.tensor([[8.0, 21.0, 33.0], [9.0, 20.0, 30.0]])
        expected_rotation = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]]).repeat(2, 1)
        expected_scales = torch.log(primitive.scales[[0, 2]])
        expected_densities = torch.logit(primitive.densities[[0, 2]])
        expected_features_albedo = RGB2SH(primitive.rgb[[0, 2]])

        assert in_memory.quaternion_format == "wxyz"
        torch.testing.assert_close(in_memory.positions, expected_positions)
        torch.testing.assert_close(in_memory.rotations, expected_rotation)
        torch.testing.assert_close(in_memory.scales, expected_scales)
        torch.testing.assert_close(in_memory.densities, expected_densities)
        torch.testing.assert_close(in_memory.features_albedo, expected_features_albedo)
        assert in_memory.road_mask is not None
        assert in_memory.road_mask.dtype == torch.bool
        assert in_memory.road_mask.tolist() == [True, False]
        assert in_memory.sky_mask is not None
        assert in_memory.sky_mask.shape == (2, 1)
        assert in_memory.sky_mask.tolist() == [[0.0], [0.75]]
        for field_name in ("positions", "rotations", "scales", "densities", "features_albedo", "sky_mask"):
            value = getattr(in_memory, field_name)
            assert value.dtype == torch.float32
            assert not value.is_inference()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "instant_nurec.ply"
            export_ply(config, primitive, rig, output_path)  # type: ignore[arg-type]
            reloaded = PLYGaussianLoader(output_path, device="cpu")

        for field_name in ("positions", "rotations", "densities", "scales", "features_albedo"):
            torch.testing.assert_close(getattr(in_memory, field_name), getattr(reloaded, field_name), rtol=0, atol=0)
        assert reloaded.features_specular is None
        assert in_memory.features_specular is None
        assert reloaded.road_mask is not None
        assert in_memory.road_mask is not None
        torch.testing.assert_close(in_memory.road_mask, reloaded.road_mask, rtol=0, atol=0)
        assert reloaded.sky_mask is not None
        assert in_memory.sky_mask is not None
        torch.testing.assert_close(in_memory.sky_mask, reloaded.sky_mask, rtol=0, atol=0)


class TestExportKelvinPly:
    """Tests for export_kelvin_ply."""

    def _make_kelvin_primitive(self, n_gaussians: int) -> KelvinInstantNuRecPrimitive:
        """Create a minimal KelvinInstantNuRecPrimitive on CPU for testing."""
        return KelvinInstantNuRecPrimitive.random(n_gaussians, use_2dgs=True, device=torch.device("cpu"))

    def _make_kelvin_primitive_with_semantics(self, n_gaussians: int) -> KelvinInstantNuRecPrimitive:
        """Create a KelvinInstantNuRecPrimitive with semantic_class populated."""
        prim = self._make_kelvin_primitive(n_gaussians)
        semantic_class = torch.zeros(n_gaussians, 1, dtype=torch.uint8)
        prim = KelvinInstantNuRecPrimitive(
            static_layer=KelvinStaticLayer(
                positions=prim.static_layer.positions,
                rotations=prim.static_layer.rotations,
                scales=prim.static_layer.scales,
                densities=prim.static_layer.densities,
                rgb=prim.static_layer.rgb,
                semantic_class=semantic_class,
            ),
            dynamic_layers=prim.dynamic_layers,
            sky_cubemap=prim.sky_cubemap,
            affine_matrix=prim.affine_matrix,
            use_2dgs=prim.use_2dgs,
            gaussians_renderer=None,
        )
        return prim

    def test_basic_export(self):
        """Export produces PLYExportGaussians with correct tensor shapes."""
        primitives = self._make_kelvin_primitive(100)
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        # All gaussians should pass (no filters active)
        assert result.positions.shape == (100, 3)
        assert result.rotations.shape == (100, 4)
        assert result.scales.shape == (100, 3)
        assert result.densities.shape == (100, 1)
        assert result.rgb.shape == (100, 3)

    def test_masks_none_without_semantic_class(self):
        """Without semantic_class, road_mask and sky_mask are None."""
        primitives = self._make_kelvin_primitive(10)
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.road_mask is None
        assert result.sky_mask is None

    def test_road_mask_from_semantic_class(self):
        """road_mask is derived from semantic_class == ROAD."""
        primitives = self._make_kelvin_primitive_with_semantics(5)
        primitives.static_layer.semantic_class[1] = KelvinSemanticClass.ROAD
        primitives.static_layer.semantic_class[3] = KelvinSemanticClass.ROAD
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.road_mask is not None
        assert result.road_mask.shape == (5,)
        assert result.road_mask.dtype == torch.uint8
        assert result.road_mask.tolist() == [0, 1, 0, 1, 0]

    def test_sky_mask_from_semantic_class(self):
        """sky_mask is derived from semantic_class == SKY."""
        primitives = self._make_kelvin_primitive_with_semantics(4)
        primitives.static_layer.semantic_class[0] = KelvinSemanticClass.SKY
        primitives.static_layer.semantic_class[2] = KelvinSemanticClass.SKY
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.sky_mask is not None
        assert result.sky_mask.shape == (4,)
        assert result.sky_mask.dtype == torch.float32
        assert result.sky_mask.tolist() == [1.0, 0.0, 1.0, 0.0]

    def test_movable_gaussians_removed_by_default(self):
        """Kelvin MOVABLE gaussians are removed from exported background by default."""
        primitives = self._make_kelvin_primitive_with_semantics(5)
        primitives.static_layer.semantic_class[1] = KelvinSemanticClass.MOVABLE
        primitives.static_layer.semantic_class[3] = KelvinSemanticClass.MOVABLE
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.positions.shape[0] == 3
        assert torch.allclose(result.positions, primitives.static_layer.positions[[0, 2, 4]])

    def test_keep_movable_gaussians_override(self):
        """The export config can keep Kelvin MOVABLE gaussians for lidar-instant_nurec backgrounds."""
        primitives = self._make_kelvin_primitive_with_semantics(5)
        primitives.static_layer.semantic_class[1] = KelvinSemanticClass.MOVABLE
        primitives.static_layer.semantic_class[3] = KelvinSemanticClass.MOVABLE
        config = _make_default_export_config(keep_movable_gaussians=True)
        result = export_kelvin_ply(config, primitives)
        assert result.positions.shape[0] == 5
        assert torch.allclose(result.positions, primitives.static_layer.positions)

    def test_masks_with_filtering(self):
        """Masks survive density filtering and have correct length."""
        primitives = self._make_kelvin_primitive_with_semantics(6)
        primitives.static_layer.semantic_class[0] = KelvinSemanticClass.ROAD
        primitives.static_layer.semantic_class[2] = KelvinSemanticClass.ROAD
        primitives.static_layer.semantic_class[4] = KelvinSemanticClass.SKY
        # Filter out half by density
        primitives.static_layer.densities[:3] = 0.9
        primitives.static_layer.densities[3:] = 0.001
        config = _make_default_export_config(minimum_density=0.5)
        result = export_kelvin_ply(config, primitives)
        assert result.positions.shape[0] == 3
        assert result.road_mask is not None
        assert result.road_mask.shape == (3,)
        # Indices 0 (ROAD), 1 (OTHERS), 2 (ROAD) survive
        assert result.road_mask.tolist() == [1, 0, 1]

    def test_in_memory_payload_matches_kelvin_disk_roundtrip_exactly(self, monkeypatch: pytest.MonkeyPatch):
        """The production Kelvin conversion is identical with and without a PLY boundary."""
        # Kelvin rotates its render-only sky cubemap as part of a world transform.
        # Bypass that CUDA-only texture operation in this CPU tensor-codec test.
        monkeypatch.setattr(
            "nurec.instant_nurec.primitives.kelvin_primitive.rotate_sky_cubemap", lambda cubemap, _: cubemap
        )
        torch.manual_seed(0)
        primitives = self._make_kelvin_primitive_with_semantics(6)
        semantic_class = primitives.static_layer.semantic_class
        assert semantic_class is not None
        semantic_class[:, 0] = torch.tensor(
            [
                KelvinSemanticClass.ROAD,
                KelvinSemanticClass.SKY,
                KelvinSemanticClass.MOVABLE,
                KelvinSemanticClass.OTHERS,
                KelvinSemanticClass.ROAD,
                KelvinSemanticClass.OTHERS,
            ],
            dtype=torch.uint8,
        )
        primitives.static_layer.densities[:, 0] = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.1, 0.5])
        config = _make_default_export_config(minimum_density=0.5, keep_movable_gaussians=True)
        rig = SimpleNamespace(T_world_base=TestInMemoryPLYGaussianPayload._world_transform())

        in_memory = build_ply_gaussian_loader(config, primitives, rig, device="cpu")  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "kelvin.ply"
            export_ply(config, primitives, rig, output_path)  # type: ignore[arg-type]
            reloaded = PLYGaussianLoader(output_path, device="cpu")

        assert in_memory.quaternion_format == reloaded.quaternion_format == "wxyz"
        for field_name in PLYGaussianLoader.TENSOR_FIELDS:
            actual = getattr(in_memory, field_name)
            expected = getattr(reloaded, field_name)
            if actual is None or expected is None:
                assert actual is expected
                continue
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert not actual.is_inference()

    def test_build_payload_from_kelvin_inference_tensors(self, monkeypatch: pytest.MonkeyPatch):
        """Direct conversion leaves inference mode before creating trainable initialization state."""
        monkeypatch.setattr(
            "nurec.instant_nurec.primitives.kelvin_primitive.rotate_sky_cubemap", lambda cubemap, _: cubemap
        )
        with torch.inference_mode():
            primitives = self._make_kelvin_primitive_with_semantics(4)
        assert primitives.static_layer.positions.is_inference()

        rig = SimpleNamespace(T_world_base=TestInMemoryPLYGaussianPayload._world_transform())
        loader = build_ply_gaussian_loader(
            _make_default_export_config(),
            primitives,
            rig,  # type: ignore[arg-type]
            device="cpu",
        )

        for field_name in PLYGaussianLoader.TENSOR_FIELDS:
            value = getattr(loader, field_name)
            if value is not None:
                assert not value.is_inference()

    def test_density_filtering(self):
        """Density filter reduces the number of exported gaussians."""
        primitives = self._make_kelvin_primitive(100)
        # Set half the densities below threshold
        primitives.static_layer.densities[:50] = 0.01
        primitives.static_layer.densities[50:] = 0.9
        config = _make_default_export_config(minimum_density=0.5)
        result = export_kelvin_ply(config, primitives)
        assert result.positions.shape[0] == 50

    def test_single_gaussian_squeeze(self):
        """Single-gaussian case doesn't produce scalar tensors (squeeze(-1) fix)."""
        primitives = self._make_kelvin_primitive(1)
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.positions.ndim == 2
        assert result.positions.shape == (1, 3)
        assert result.densities.shape == (1, 1)

    def test_normals_propagated_to_export(self):
        """Predicted normals on the static layer survive filtering and are forwarded to the PLY export struct."""
        primitives = self._make_kelvin_primitive(6)
        normals = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
        primitives.static_layer.normals = normals
        primitives.static_layer.densities[:3] = 0.9
        primitives.static_layer.densities[3:] = 0.001
        config = _make_default_export_config(minimum_density=0.5)
        result = export_kelvin_ply(config, primitives)
        assert result.normals is not None
        assert result.normals.shape == (3, 3)
        assert torch.allclose(result.normals, normals[:3])

    def test_normals_none_when_absent(self):
        """When normals are not present on the static layer, export carries None."""
        primitives = self._make_kelvin_primitive(4)
        config = _make_default_export_config()
        result = export_kelvin_ply(config, primitives)
        assert result.normals is None

    def test_all_filtered_produces_empty(self):
        """When all gaussians are filtered, result has zero-length tensors."""
        primitives = self._make_kelvin_primitive(10)
        # Set all densities very low
        primitives.static_layer.densities[:] = 0.001
        config = _make_default_export_config(minimum_density=0.5)
        result = export_kelvin_ply(config, primitives)
        assert result.positions.shape[0] == 0
