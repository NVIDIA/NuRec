# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest

from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import torch

from nurec.models.gaussians.utils import (
    RGB2SH,
    SH2RGB,
    PLYGaussianLoader,
    apply_rgb_post_processing_to_features_dc,
    make_ppisp_crf_inverse_rgb_post_processing,
    ppisp_crf_inverse,
    subsample_points,
    track_random_initialization,
    write_ply_3dgs,
)
from nurec.utils.types import PointCloud


CRF_TEST_CAMERA_INDEX = 1


def _write_instant_nurec_ply(path: Path, n: int = 16, with_normals: bool = True, with_rgba: bool = True) -> None:
    """Write a PLY matching the Instant-NuRec output schema: positions, normals, uchar RGBA,
    rot_*, scale_*, opacity, f_dc_*, road_mask (uchar), sky_mask (float), and no f_rest_* SH."""
    rng = np.random.default_rng(0)
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = rng.standard_normal((n, 3)).astype(np.float32)
    if with_normals:
        normals = rng.standard_normal((n, 3)).astype(np.float32)
        mesh.vertex_data.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_rgba:
        mesh.vertex_data.colors = rng.integers(0, 255, size=(n, 4), dtype=np.uint8)

    custom = mesh.vertex_data.custom_attributes
    for i in range(4):
        custom[f"rot_{i}"] = rng.standard_normal(n).astype(np.float32)
    for i in range(3):
        custom[f"scale_{i}"] = rng.standard_normal(n).astype(np.float32)
    custom["opacity"] = rng.standard_normal(n).astype(np.float32)
    for i in range(3):
        custom[f"f_dc_{i}"] = rng.standard_normal(n).astype(np.float32)
    custom["road_mask"] = rng.integers(0, 2, size=n, dtype=np.uint8)
    custom["sky_mask"] = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))


class TestPLYGaussianLoaderInstantNuRec(unittest.TestCase):
    def test_loads_ply_without_f_rest_and_with_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instant_nurec.ply"
            _write_instant_nurec_ply(path, n=32)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            loader = PLYGaussianLoader(path, device=device)

            self.assertEqual(loader.positions.shape, (32, 3))
            self.assertEqual(loader.rotations.shape, (32, 4))
            self.assertEqual(loader.scales.shape, (32, 3))
            self.assertEqual(loader.densities.shape, (32, 1))
            self.assertEqual(loader.features_albedo.shape, (32, 3))
            self.assertIsNone(loader.features_specular)
            self.assertIsNotNone(loader.road_mask)
            self.assertIsNotNone(loader.sky_mask)

    def test_clone_copies_mutable_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instant_nurec.ply"
            _write_instant_nurec_ply(path, n=8)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            loader = PLYGaussianLoader(path, device=device)
            loader.cuboids_dims = torch.tensor([1.0, 2.0, 3.0], device=device)

            cloned = loader.clone()
            cloned.positions += 10.0
            assert cloned.cuboids_dims is not None
            cloned.cuboids_dims += 5.0

            self.assertFalse(torch.equal(cloned.positions, loader.positions))
            assert loader.cuboids_dims is not None
            torch.testing.assert_close(loader.cuboids_dims, torch.tensor([1.0, 2.0, 3.0], device=device))

    def test_from_tensors_owns_regular_payload_and_moves_every_field(self):
        n = 4
        with torch.inference_mode():
            positions = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
            rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
            densities = torch.linspace(-1.0, 1.0, n).unsqueeze(-1)
            scales = torch.full((n, 3), -2.0)
            features_albedo = torch.linspace(-0.5, 0.5, n * 3).reshape(n, 3)
            features_specular = torch.arange(n * 6, dtype=torch.float32).reshape(n, 6)
            road_mask = torch.tensor([True, False, True, False])
            sky_mask = torch.tensor([[0.0], [0.25], [0.5], [1.0]])
            cuboids_dims = torch.tensor([1.0, 2.0, 3.0])

        loader = PLYGaussianLoader.from_tensors(
            positions=positions,
            rotations=rotations,
            densities=densities,
            scales=scales,
            features_albedo=features_albedo,
            features_specular=features_specular,
            road_mask=road_mask,
            sky_mask=sky_mask,
            cuboids_dims=cuboids_dims,
        )

        expected = {
            "positions": positions,
            "rotations": rotations,
            "densities": densities,
            "scales": scales,
            "features_albedo": features_albedo,
            "features_specular": features_specular,
            "road_mask": road_mask,
            "sky_mask": sky_mask,
            "cuboids_dims": cuboids_dims,
        }
        self.assertEqual(tuple(expected), PLYGaussianLoader.TENSOR_FIELDS)
        for field_name, source in expected.items():
            value = getattr(loader, field_name)
            assert value is not None
            self.assertFalse(value.is_inference())
            self.assertNotEqual(value.data_ptr(), source.data_ptr())
            torch.testing.assert_close(value, source)

        cloned = loader.clone()
        cloned.positions.add_(10.0)
        self.assertFalse(torch.equal(cloned.positions, loader.positions))

    def test_from_tensors_can_take_ownership_without_copying(self):
        positions = torch.zeros((2, 3))
        required = {
            "positions": positions,
            "rotations": torch.zeros((2, 4)),
            "densities": torch.zeros((2, 1)),
            "scales": torch.zeros((2, 3)),
            "features_albedo": torch.zeros((2, 3)),
        }

        loader = PLYGaussianLoader.from_tensors(**required, copy=False)

        self.assertEqual(loader.positions.data_ptr(), positions.data_ptr())
        with torch.inference_mode():
            inference_positions = positions.clone()
        with self.assertRaisesRegex(ValueError, "cannot take ownership of an inference tensor"):
            PLYGaussianLoader.from_tensors(
                **(required | {"positions": inference_positions}),
                copy=False,
            )

    def test_from_tensors_rejects_unsupported_quaternion_format(self):
        with self.assertRaisesRegex(ValueError, "Unsupported quaternion_format 'invalid'"):
            PLYGaussianLoader.from_tensors(
                positions=torch.zeros((2, 3)),
                rotations=torch.zeros((2, 4)),
                densities=torch.zeros((2, 1)),
                scales=torch.zeros((2, 3)),
                features_albedo=torch.zeros((2, 3)),
                quaternion_format="invalid",  # type: ignore[arg-type]
            )

    def test_from_tensors_rejects_inconsistent_payload_shapes(self):
        required = {
            "positions": torch.zeros((2, 3)),
            "rotations": torch.zeros((2, 4)),
            "densities": torch.zeros((2, 1)),
            "scales": torch.zeros((2, 3)),
            "features_albedo": torch.zeros((2, 3)),
        }
        invalid_fields = {
            "rotations": torch.zeros((2, 3)),
            "densities": torch.zeros((2,)),
            "features_specular": torch.zeros((3, 6)),
            "road_mask": torch.zeros((2, 1), dtype=torch.bool),
            "sky_mask": torch.zeros((2,)),
        }

        for name, value in invalid_fields.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                PLYGaussianLoader.from_tensors(**(required | {name: value}))

    def test_from_tensors_requires_a_boolean_road_mask(self):
        with self.assertRaisesRegex(ValueError, "road_mask must have dtype bool"):
            PLYGaussianLoader.from_tensors(
                positions=torch.zeros((2, 3)),
                rotations=torch.zeros((2, 4)),
                densities=torch.zeros((2, 1)),
                scales=torch.zeros((2, 3)),
                features_albedo=torch.zeros((2, 3)),
                road_mask=torch.zeros((2,), dtype=torch.uint8),
            )


class TestPLYGaussianLoaderSphericalHarmonics(unittest.TestCase):
    def test_f_rest_round_trips_through_write_ply_3dgs(self):
        """The f_rest_* properties must come back in coefficient order.

        They are read out of a dict keyed by property name, and that dict is not ordered
        by index, so reading the keys as they come scrambles the specular coefficients.
        Distinct per-coefficient values make a permutation fail here instead of silently
        reshuffling view-dependent colour at render time.
        """
        n, num_speculars = 8, 15
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features_specular = torch.arange(n * num_speculars * 3, dtype=torch.float32, device=device).reshape(
            n, num_speculars * 3
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sh_degree_3.ply"
            write_ply_3dgs(
                path=path,
                positions=torch.zeros((n, 3), device=device),
                rotations=torch.zeros((n, 4), device=device),
                scales=torch.zeros((n, 3), device=device),
                densities=torch.zeros(n, device=device),
                features_albedo=torch.zeros((n, 3), device=device),
                features_specular=features_specular,
            )
            loader = PLYGaussianLoader(path, device=device)

            assert loader.features_specular is not None
            self.assertEqual(loader.features_specular.shape, (n, num_speculars * 3))
            torch.testing.assert_close(loader.features_specular, features_specular)


def _ppisp_crf_forward(rgb: torch.Tensor, crf_params: torch.Tensor) -> torch.Tensor:
    crf_params = crf_params.to(device=rgb.device, dtype=rgb.dtype)
    toe = 0.3 + torch.nn.functional.softplus(crf_params[:, 0])
    shoulder = 0.3 + torch.nn.functional.softplus(crf_params[:, 1])
    gamma = 0.1 + torch.nn.functional.softplus(crf_params[:, 2])
    center = torch.clamp(torch.sigmoid(crf_params[:, 3]), 1.0e-6, 1.0 - 1.0e-6)

    x = torch.clamp(rgb, 0.0, 1.0)
    lerp = torch.clamp((shoulder - toe) * center + toe, min=1.0e-6)
    a = (shoulder * center) / lerp
    b = 1.0 - a
    below = a * torch.pow(x / center, toe)
    above = 1.0 - b * torch.pow((1.0 - x) / (1.0 - center), shoulder)
    return torch.pow(torch.clamp(torch.where(x <= center, below, above), min=0.0), gamma)


def _make_varied_crf_params() -> torch.Tensor:
    return torch.tensor(
        [
            [-2.0, 1.5, -1.0, -1.5],
            [0.75, -0.8, 1.2, 0.25],
            [2.0, -1.5, 0.5, 1.5],
        ],
        dtype=torch.float32,
    )


class TestPPISPCRFInverseAssetBake(unittest.TestCase):
    def test_crf_inverse_round_trips_clamped_rgb(self):
        crf_params = _make_varied_crf_params()
        rgb = torch.tensor([[0.05, 0.2, 0.35], [0.55, 0.7, 0.95]], dtype=torch.float32)

        inverted = ppisp_crf_inverse(rgb, crf_params)
        round_tripped = _ppisp_crf_forward(inverted, crf_params)

        self.assertTrue(torch.allclose(round_tripped, rgb, atol=1.0e-6))

    def test_feature_bake_changes_dc_band_only(self):
        crf_params = _make_varied_crf_params()
        features = torch.tensor(
            [
                [[0.1, -0.2, 0.3], [0.4, 0.5, 0.6]],
                [[-0.3, 0.2, -0.1], [0.7, 0.8, 0.9]],
            ],
            dtype=torch.float32,
        )
        original = features.clone()

        apply_rgb_post_processing_to_features_dc(features, make_ppisp_crf_inverse_rgb_post_processing(crf_params))

        self.assertFalse(torch.allclose(features[:, 0, :], original[:, 0, :]))
        self.assertTrue(torch.equal(features[:, 1, :], original[:, 1, :]))
        round_tripped_rgb = _ppisp_crf_forward(SH2RGB(features[:, 0, :]), crf_params)
        self.assertTrue(
            torch.allclose(round_tripped_rgb, torch.clamp(SH2RGB(original[:, 0, :]), 0.0, 1.0), atol=1.0e-6)
        )

    def test_feature_bake_supports_camera_indexed_crf_params(self):
        crf_params = torch.zeros(2, 3, 4, dtype=torch.float32)
        crf_params[CRF_TEST_CAMERA_INDEX] = _make_varied_crf_params()
        features = RGB2SH(torch.full((2, 3), 0.25))

        apply_rgb_post_processing_to_features_dc(
            features,
            make_ppisp_crf_inverse_rgb_post_processing(crf_params, camera_index=CRF_TEST_CAMERA_INDEX),
        )

        expected_rgb = ppisp_crf_inverse(torch.full((2, 3), 0.25), crf_params[CRF_TEST_CAMERA_INDEX])
        self.assertTrue(torch.allclose(SH2RGB(features), expected_rgb, atol=1.0e-6))


class TestTrackRandomInitialization(unittest.TestCase):
    def test_track_random_initialization_shapes_and_values(self):
        device = torch.device("cuda")

        # Deterministic random for reproducibility
        torch.manual_seed(123)

        num_points = 64
        track_index = 7
        cuboid_extent = torch.tensor([2.0, 4.0, 6.0], device=device)
        default_scale = 0.3

        pc, gid, gscale = track_random_initialization(
            num_points=num_points,
            track_index=track_index,
            cuboid_extent=cuboid_extent,
            default_scale=default_scale,
            device=device,
        )

        # Basic types and shapes
        self.assertIsInstance(pc, PointCloud)
        self.assertEqual(pc.n_points, num_points)
        self.assertEqual(pc.xyz_start.shape, (num_points, 3))
        self.assertTrue(torch.equal(pc.xyz_start, pc.xyz_end))
        assert pc.color is not None
        self.assertEqual(pc.color.shape, (num_points, 3))
        self.assertEqual(pc.color.dtype, torch.uint8)
        assert pc.camera_footprint_scale is not None
        self.assertEqual(pc.camera_footprint_scale.shape, (num_points,))

        self.assertEqual(gid.shape, (num_points, 1))
        self.assertEqual(gid.dtype, torch.int32)
        self.assertTrue(torch.all(gid == track_index))

        self.assertEqual(gscale.shape, (num_points, 1))
        self.assertTrue(torch.allclose(gscale.squeeze(-1), pc.camera_footprint_scale))

        # Devices
        self.assertEqual(pc.xyz_start.device.type, device.type)
        self.assertEqual(pc.xyz_end.device.type, device.type)
        self.assertEqual(pc.color.device.type, device.type)
        self.assertEqual(pc.camera_footprint_scale.device.type, device.type)
        self.assertEqual(gid.device.type, device.type)
        self.assertEqual(gscale.device.type, device.type)

        # Values
        self.assertTrue(
            torch.allclose(pc.camera_footprint_scale, torch.full((num_points,), default_scale, device=device))
        )

        # Points should lie within [-extent/2, extent/2] per axis
        lower = -cuboid_extent / 2
        upper = cuboid_extent / 2
        self.assertTrue(torch.all(pc.xyz_end >= lower))
        self.assertTrue(torch.all(pc.xyz_end <= upper))


class TestSubsamplePoints(unittest.TestCase):
    def _make_pc(self, n: int, base: float, device: torch.device) -> PointCloud:
        # Construct deterministic, easily-checkable coordinates
        xyz = torch.stack(
            [
                torch.linspace(base, base + n - 1, n, device=device),
                torch.linspace(base + 1000, base + 1000 + n - 1, n, device=device),
                torch.linspace(base + 2000, base + 2000 + n - 1, n, device=device),
            ],
            dim=1,
        )

        color = (torch.arange(n, device=device)[:, None].repeat(1, 3) % 256).to(torch.uint8)
        scale = torch.linspace(0.1, 1.0, n, device=device)
        return PointCloud(
            xyz_start=xyz,
            xyz_end=xyz,
            color=color,
            camera_footprint_scale=scale,
        )

    def test_no_subsampling_when_below_threshold(self):
        device = torch.device("cuda")

        pc1 = self._make_pc(5, base=0.0, device=device)
        pc2 = self._make_pc(7, base=100.0, device=device)
        gid1 = torch.full((pc1.n_points, 1), 1, dtype=torch.int32, device=device)
        gid2 = torch.full((pc2.n_points, 1), 2, dtype=torch.int32, device=device)
        assert pc1.camera_footprint_scale is not None
        assert pc2.camera_footprint_scale is not None
        gsc1 = pc1.camera_footprint_scale.unsqueeze(-1)
        gsc2 = pc2.camera_footprint_scale.unsqueeze(-1)

        num_points_target = pc1.n_points + pc2.n_points  # equal -> no subsampling
        observed_counts = [pc1.n_points, pc2.n_points]

        pcs_out, gids_out, gscs_out, counts_out = subsample_points(
            num_points=num_points_target,
            point_clouds=[pc1, pc2],
            gaussian_cuboid_ids=[gid1, gid2],
            gaussian_scales=[gsc1, gsc2],
            observed_counts=observed_counts,
        )

        # Should be unchanged in sizes
        self.assertEqual(pcs_out[0].n_points, pc1.n_points)
        self.assertEqual(pcs_out[1].n_points, pc2.n_points)
        self.assertTrue(torch.equal(gids_out[0], gid1))
        self.assertTrue(torch.equal(gids_out[1], gid2))
        self.assertTrue(torch.allclose(gscs_out[0], gsc1))
        self.assertTrue(torch.allclose(gscs_out[1], gsc2))
        self.assertEqual(counts_out, observed_counts)

    def test_deterministic_subsampling(self):
        device = torch.device("cuda")

        pc1 = self._make_pc(10, base=0.0, device=device)
        pc2 = self._make_pc(6, base=100.0, device=device)
        gid1 = torch.full((pc1.n_points, 1), 11, dtype=torch.int32, device=device)
        gid2 = torch.full((pc2.n_points, 1), 22, dtype=torch.int32, device=device)
        assert pc1.camera_footprint_scale is not None
        assert pc2.camera_footprint_scale is not None
        gsc1 = pc1.camera_footprint_scale.unsqueeze(-1)
        gsc2 = pc2.camera_footprint_scale.unsqueeze(-1)

        total_points = pc1.n_points + pc2.n_points  # 16
        num_points_target = 8  # ratio = 0.5 -> expect 5 and 3
        ratio = float(num_points_target) / total_points
        n1 = int(pc1.n_points * ratio)
        n2 = int(pc2.n_points * ratio)
        self.assertEqual(n1, 5)
        self.assertEqual(n2, 3)

        # Prepare deterministic numpy RNG and compute expected indices
        seed = 0
        np.random.seed(seed)
        expected_idx1 = np.random.choice(pc1.n_points, n1, replace=False)
        expected_idx2 = np.random.choice(pc2.n_points, n2, replace=False)

        # Reset seed so function uses the same random draws
        np.random.seed(seed)
        observed_counts = [pc1.n_points, pc2.n_points]
        pcs_out, gids_out, gscs_out, counts_out = subsample_points(
            num_points=num_points_target,
            point_clouds=[pc1, pc2],
            gaussian_cuboid_ids=[gid1, gid2],
            gaussian_scales=[gsc1, gsc2],
            observed_counts=observed_counts,
        )

        # Validate sizes
        self.assertEqual(pcs_out[0].n_points, n1)
        self.assertEqual(pcs_out[1].n_points, n2)
        self.assertEqual(gids_out[0].shape, (n1, 1))
        self.assertEqual(gids_out[1].shape, (n2, 1))
        self.assertEqual(gscs_out[0].shape, (n1, 1))
        self.assertEqual(gscs_out[1].shape, (n2, 1))
        self.assertEqual(counts_out, observed_counts)

        # Check that subsampled elements match expected indices
        self.assertTrue(torch.allclose(pcs_out[0].xyz_end, pc1.xyz_end[torch.tensor(expected_idx1, device=device)]))
        self.assertTrue(torch.allclose(pcs_out[1].xyz_end, pc2.xyz_end[torch.tensor(expected_idx2, device=device)]))
        self.assertTrue(torch.equal(gids_out[0], gid1[torch.tensor(expected_idx1, device=device)]))
        self.assertTrue(torch.equal(gids_out[1], gid2[torch.tensor(expected_idx2, device=device)]))
        self.assertTrue(torch.allclose(gscs_out[0], gsc1[torch.tensor(expected_idx1, device=device)]))
        self.assertTrue(torch.allclose(gscs_out[1], gsc2[torch.tensor(expected_idx2, device=device)]))


if __name__ == "__main__":
    unittest.main()
