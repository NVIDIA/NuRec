# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numpy.typing import NDArray


def cuboid_corners_local(dims_xyz: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the 8 corners of a cuboid in its local frame.

    Args:
        dims_xyz: (3,) array of [length, width, height] (x/y/z extents).

    Returns:
        (8, 3) array of corner positions. Ordering: 4 bottom corners then 4 top corners,
        each quad starting with the 2 front corners (+x side).
    """
    dx, dy, dz = dims_xyz[0] / 2, dims_xyz[1] / 2, dims_xyz[2] / 2
    return np.array(
        [
            [+dx, +dy, -dz],  # 0 bottom front-left
            [+dx, -dy, -dz],  # 1 bottom front-right
            [-dx, -dy, -dz],  # 2 bottom back-right
            [-dx, +dy, -dz],  # 3 bottom back-left
            [+dx, +dy, +dz],  # 4 top front-left
            [+dx, -dy, +dz],  # 5 top front-right
            [-dx, -dy, +dz],  # 6 top back-right
            [-dx, +dy, +dz],  # 7 top back-left
        ],
        dtype=np.float32,
    )


def cuboid_base_grid_local(dims_xyz: NDArray[np.float32], n: int) -> NDArray[np.float32]:
    """Return an n×n grid of sample points over the cuboid base in its local frame.

    Args:
        dims_xyz: (3,) array of [length, width, height] (x/y/z extents).
        n: Grid resolution; produces n*n points.

    Returns:
        (n*n, 3) float32 array of sample positions on the base plane (z = -height/2).
    """
    dx, dy, dz = dims_xyz[0] / 2, dims_xyz[1] / 2, dims_xyz[2] / 2
    xs = np.linspace(-dx, +dx, n, dtype=np.float32)
    ys = np.linspace(-dy, +dy, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel(), np.full(n * n, -dz, dtype=np.float32)], axis=1)
