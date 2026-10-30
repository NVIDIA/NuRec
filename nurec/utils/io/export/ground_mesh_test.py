# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import pytest

from click.testing import CliRunner

from nurec.utils.io.export.ground_mesh import export_ground_mesh


@pytest.mark.parametrize(
    ("option", "invalid_value", "expected_range"),
    [
        ("--cuboids-base-margin", "-0.1", "x>=0"),
        ("--cuboids-exclude-points-radius", "-0.1", "x>=0"),
        ("--cuboids-exclude-points-knns", "0", "x>=1"),
        ("--cuboids-exclude-points-knns", "-1", "x>=1"),
        ("--cuboid-point-merge-cell-size", "0", "x>0"),
        ("--cuboid-point-merge-cell-size", "-0.1", "x>0"),
    ],
)
def test_cuboid_extension_cli_rejects_invalid_parameters(option: str, invalid_value: str, expected_range: str) -> None:
    result = CliRunner().invoke(
        export_ground_mesh,
        ["--output-dir", "unused", option, invalid_value],
    )

    assert result.exit_code == 2
    assert expected_range in result.output


def test_cuboid_extension_cli_uses_matching_range_types() -> None:
    params = {param.name: param for param in export_ground_mesh.params}

    extend_to_cuboids = params["extend_to_cuboids"]
    assert isinstance(extend_to_cuboids, click.Option)
    assert extend_to_cuboids.opts == ["--extend-to-cuboids"]
    assert extend_to_cuboids.secondary_opts == []
    assert extend_to_cuboids.is_flag
    assert not extend_to_cuboids.default

    for name in ("cuboids_base_margin", "cuboids_exclude_points_radius"):
        range_type = params[name].type
        assert isinstance(range_type, click.FloatRange)
        assert range_type.min == 0

    knns_range = params["cuboids_exclude_points_knns"].type
    assert isinstance(knns_range, click.IntRange)
    assert knns_range.min == 1

    merge_cell_size_range = params["cuboid_point_merge_cell_size"].type
    assert isinstance(merge_cell_size_range, click.FloatRange)
    assert merge_cell_size_range.min == 0
    assert merge_cell_size_range.min_open
    assert params["cuboid_point_merge_cell_size"].default == 0.5
    assert params["min_ray_length"].default == 0.1
    assert params["voxel_size"].default == 0.1
