# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests the PoissonRecon wrapper and executable by reconstructing a sphere."""

import numpy as np
import point_cloud_utils as pcu

from nurec.utils.io.poisson_recon import reconstruct_surface


def test_reconstruct_sphere(tmp_path):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(3000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    input_file = str(tmp_path / "points.ply")
    output_file = str(tmp_path / "mesh.ply")
    pcu.save_mesh_vn(input_file, points, points)

    reconstruct_surface(input_file, output_file, width=0.1, density=True, samples_per_node=1.0)

    vertices, faces = pcu.load_mesh_vf(output_file)
    assert len(faces) > 0
    radii = np.linalg.norm(vertices, axis=1)
    assert np.abs(radii - 1.0).max() < 0.1
