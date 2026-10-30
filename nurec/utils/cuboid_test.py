# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from nurec.utils.cuboid import cuboid_base_grid_local, cuboid_corners_local


def test_cuboid_corners_local_shape_and_symmetry() -> None:
    dims = np.array([4.0, 2.0, 1.5], dtype=np.float32)
    corners = cuboid_corners_local(dims)
    assert corners.shape == (8, 3)
    assert corners.dtype == np.float32
    # Each axis range must equal the corresponding dim
    np.testing.assert_allclose(corners[:, 0].max() - corners[:, 0].min(), dims[0], atol=1e-6)
    np.testing.assert_allclose(corners[:, 1].max() - corners[:, 1].min(), dims[1], atol=1e-6)
    np.testing.assert_allclose(corners[:, 2].max() - corners[:, 2].min(), dims[2], atol=1e-6)
    # Centroid must be at origin
    np.testing.assert_allclose(corners.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-6)


def test_cuboid_base_grid_local_shape_and_position() -> None:
    dims = np.array([4.0, 2.0, 1.5], dtype=np.float32)
    n = 5
    grid = cuboid_base_grid_local(dims, n)
    assert grid.shape == (n * n, 3)
    assert grid.dtype == np.float32
    # All points must be on the base plane
    np.testing.assert_allclose(grid[:, 2], -dims[2] / 2, atol=1e-6)
    # XY extent must match dims
    np.testing.assert_allclose(grid[:, 0].max() - grid[:, 0].min(), dims[0], atol=1e-6)
    np.testing.assert_allclose(grid[:, 1].max() - grid[:, 1].min(), dims[1], atol=1e-6)
