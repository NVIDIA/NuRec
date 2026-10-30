# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``.ply`` static-prop loader of the asset-edits bake.

CPU-only: fixtures are written with ``write_ply_3dgs`` into a tempdir and read back
through the real ``PLYGaussianLoader``; no scene data, no GPU, no USD.
"""

import tempfile
import unittest

from pathlib import Path

import numpy as np
import torch

from nurec.models.gaussians.utils import write_ply_3dgs
from nurec.utils.edit.layer_append import euler_xyz_to_matrix, matrix_to_quat_wxyz, quat_mul_wxyz
from nurec.utils.edit.ply_gaussians import ZUP_FROM_YUP, PlyAssetError, load_ply_asset


def _write_ply(
    path: Path,
    positions: np.ndarray,
    *,
    rotations: np.ndarray | None = None,
    sh_rest_cols: int = 0,
) -> dict[str, torch.Tensor]:
    """Write a 3DGS PLY of ``positions`` with distinct per-column values; returns what was written."""
    n = len(positions)
    if rotations is None:
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    tensors = {
        "positions": torch.tensor(positions, dtype=torch.float32),
        "rotations": torch.tensor(rotations, dtype=torch.float32),
        "scales": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 7 - 3.0,
        "densities": torch.linspace(-2.0, 2.0, n),
        "features_albedo": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10,
    }
    if sh_rest_cols:
        tensors["features_specular"] = torch.arange(n * sh_rest_cols, dtype=torch.float32).reshape(n, sh_rest_cols)
    write_ply_3dgs(
        path,
        positions=tensors["positions"],
        rotations=tensors["rotations"],
        scales=tensors["scales"],
        densities=tensors["densities"],
        features_albedo=tensors["features_albedo"],
        features_specular=tensors.get("features_specular"),
    )
    return tensors


# A Y-up box: 1 m wide (x), 2 m tall (y, base at 0), 0.5 m deep (z).
_BOX = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (0.0, 2.0) for z in (-0.25, 0.25)])


class LoadPlyAssetTest(unittest.TestCase):
    """The PLY becomes a one-part, controls-free asset in the Z-up asset frame."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "prop.ply"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_static_single_part_asset(self) -> None:
        """One static part named after the file; no controls, no variantSets, Z-up, metres, float64."""
        _write_ply(self.path, _BOX)
        asset = load_ply_asset(self.path)
        self.assertEqual(asset.asset_path, str(self.path))
        self.assertEqual(asset.root_prim_path, "")
        self.assertEqual(asset.controls, {})
        self.assertEqual(asset.variant_sets, {})
        self.assertEqual((asset.up_axis, asset.meters_per_unit), ("Z", 1.0))
        self.assertEqual([p.prim_path for p in asset.static_parts], ["prop.ply"])
        g = asset.static_parts[0].gaussians
        self.assertEqual(g.count, len(_BOX))
        self.assertEqual(g.positions.dtype, np.float64)
        self.assertEqual(g.densities_logit.shape, (len(_BOX), 1))

    def test_yup_file_is_rotated_to_zup(self) -> None:
        """The file's +y becomes +z and its +z becomes -y: the rotation insert_asset applies to every PLY."""
        _write_ply(self.path, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
        g = load_ply_asset(self.path).static_parts[0].gaussians
        np.testing.assert_allclose(g.positions, [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-7)
        # The box's 2 m extent along the file's y is the asset's height.
        _write_ply(self.path, _BOX)
        g = load_ply_asset(self.path).static_parts[0].gaussians
        np.testing.assert_allclose(g.positions.max(axis=0) - g.positions.min(axis=0), [1.0, 0.5, 2.0], atol=1e-7)

    def test_rotations_compose_with_the_up_axis_change_and_are_normalized(self) -> None:
        """Each splat's quaternion is pre-multiplied by the up-axis rotation, unit-normalised first."""
        own = matrix_to_quat_wxyz(euler_xyz_to_matrix(10.0, 20.0, 30.0))
        rotations = np.stack([[1.0, 0.0, 0.0, 0.0], own, 2.0 * own])  # the last one deliberately unnormalised
        _write_ply(self.path, np.zeros((3, 3)), rotations=rotations)
        g = load_ply_asset(self.path).static_parts[0].gaussians
        q_up = matrix_to_quat_wxyz(ZUP_FROM_YUP)
        np.testing.assert_allclose(g.rotations[0], q_up, atol=1e-6)
        expected = quat_mul_wxyz(q_up[None], own[None])[0]
        np.testing.assert_allclose(g.rotations[1], expected, atol=1e-6)
        np.testing.assert_allclose(g.rotations[2], expected, atol=1e-6)

    def test_pre_activation_attributes_pass_through(self) -> None:
        """Log scales, logit opacity and the DC band are copied, not re-activated."""
        written = _write_ply(self.path, _BOX)
        g = load_ply_asset(self.path).static_parts[0].gaussians
        np.testing.assert_allclose(g.scales_log, written["scales"].numpy(), atol=1e-6)
        np.testing.assert_allclose(g.densities_logit[:, 0], written["densities"].numpy(), atol=1e-6)
        np.testing.assert_allclose(g.sh_dc, written["features_albedo"].numpy(), atol=1e-6)

    def test_dc_only_file_has_no_rest_bands(self) -> None:
        """A DC-only PLY (the asset bank default) yields sh_degree 0 and an empty (N, 0, 3) sh_rest."""
        _write_ply(self.path, _BOX)
        g = load_ply_asset(self.path).static_parts[0].gaussians
        self.assertEqual(g.sh_degree, 0)
        self.assertEqual(g.sh_rest.shape, (len(_BOX), 0, 3))

    def test_degree3_rest_is_coefficient_major(self) -> None:
        """45 f_rest_* properties come back as (N, 15, 3) in coefficient order, degree 3."""
        written = _write_ply(self.path, _BOX, sh_rest_cols=45)
        g = load_ply_asset(self.path).static_parts[0].gaussians
        self.assertEqual(g.sh_degree, 3)
        self.assertEqual(g.sh_rest.shape, (len(_BOX), 15, 3))
        np.testing.assert_allclose(g.sh_rest.reshape(len(_BOX), 45), written["features_specular"].numpy(), atol=1e-6)

    def test_partial_sh_degree_rejected(self) -> None:
        """An f_rest_* count that is not a whole SH degree fails loud."""
        _write_ply(self.path, _BOX, sh_rest_cols=12)
        with self.assertRaisesRegex(PlyAssetError, "not a whole SH degree"):
            load_ply_asset(self.path)

    def test_controls_and_prim_path_refused(self) -> None:
        """A PLY has no variantSets and no prims: both requests are errors naming the file."""
        _write_ply(self.path, _BOX)
        with self.assertRaisesRegex(PlyAssetError, r"prop\.ply: .*no variantSets.*panel"):
            load_ply_asset(self.path, control_sets=["panel"])
        with self.assertRaisesRegex(PlyAssetError, r"prop\.ply: .*no prims"):
            load_ply_asset(self.path, root_prim_path="/Body")

    def test_plain_point_cloud_is_not_a_splat(self) -> None:
        """A PLY without the 3DGS vertex properties is rejected, naming the first missing one."""
        self.path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 1 1\n"
        )
        with self.assertRaisesRegex(PlyAssetError, "not a 3DGS PLY .*missing vertex property"):
            load_ply_asset(self.path)

    def test_non_finite_values_rejected(self) -> None:
        """A NaN anywhere in the payload fails loud instead of baking a poisoned splat."""
        bad = _BOX.copy()
        bad[0, 0] = np.nan
        _write_ply(self.path, bad)
        with self.assertRaisesRegex(PlyAssetError, "non-finite values in positions"):
            load_ply_asset(self.path)

    def test_zero_quaternion_rejected(self) -> None:
        """A zero-norm quaternion is corrupt authoring (same rule as the USD importer)."""
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (len(_BOX), 1))
        rotations[2] = 0.0
        _write_ply(self.path, _BOX, rotations=rotations)
        with self.assertRaisesRegex(PlyAssetError, "1 zero-norm quaternion"):
            load_ply_asset(self.path)


if __name__ == "__main__":
    unittest.main()
