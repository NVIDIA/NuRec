# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import pytest

from click.testing import CliRunner

from nurec.utils.io.export.camera_ground_mesh import export_camera_ground_mesh


ROAD_CLASS_ID = 0
NON_ROAD_CLASS_ID = 5


def _write_fused_ply(path: Path, with_segment_id: bool = True) -> None:
    """Write a synthetic fused point cloud PLY with positions, RGB colors and a
    ``segment_id`` per-point custom attribute, mimicking the geometry-inference output the
    tool consumes.

    The road points form a dense, near-planar patch so the BEV downsample + Delaunay build
    produces a non-degenerate mesh; a handful of elevated non-road points exercise the
    semantic class filter.
    """
    rng = np.random.RandomState(0)

    # Dense ~6 m x 6 m road patch at 0.15 m spacing (downsamples to a ~0.3 m grid).
    grid = np.arange(0.0, 6.0, 0.15, dtype=np.float32)
    gx, gy = np.meshgrid(grid, grid)
    road_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    road_z = 0.02 * rng.randn(len(road_xy)).astype(np.float32)
    road_xyz = np.column_stack([road_xy, road_z]).astype(np.float32)
    road_seg = np.full(len(road_xyz), ROAD_CLASS_ID, dtype=np.int32)

    # Non-road points elevated well above the road plane.
    non_road_xyz = rng.rand(64, 3).astype(np.float32) * np.array([6.0, 6.0, 1.0], np.float32)
    non_road_xyz[:, 2] += 3.0
    non_road_seg = np.full(len(non_road_xyz), NON_ROAD_CLASS_ID, dtype=np.int32)

    xyz = np.concatenate([road_xyz, non_road_xyz]).astype(np.float32)
    seg = np.concatenate([road_seg, non_road_seg]).astype(np.int32)
    colors = (rng.rand(len(xyz), 3) * 255).astype(np.uint8)

    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = xyz
    mesh.vertex_data.colors = colors
    if with_segment_id:
        mesh.vertex_data.custom_attributes = {"segment_id": seg}
    mesh.save(str(path))


@pytest.fixture
def fused_ply_path(tmp_path: Path) -> Path:
    path = tmp_path / "fused.ply"
    _write_fused_ply(path)
    return path


def _assert_valid_mesh(path: Path) -> None:
    assert path.is_file(), f"{path} not found"
    mesh = pcu.load_triangle_mesh(str(path))
    assert len(mesh.vertex_data.positions) >= 3
    assert len(mesh.face_data.vertex_ids) > 0


def test_export_camera_ground_mesh_default(fused_ply_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "mesh_ground.ply"
    result = CliRunner().invoke(
        export_camera_ground_mesh,
        [
            "--input",
            str(fused_ply_path),
            "--output",
            str(output_path),
            "--road-class-id",
            str(ROAD_CLASS_ID),
        ],
        catch_exceptions=False,
    )
    print(result.output)
    assert result.exit_code == 0
    _assert_valid_mesh(output_path)


def test_export_camera_ground_mesh_no_postprocess(fused_ply_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "mesh_ground.ply"
    result = CliRunner().invoke(
        export_camera_ground_mesh,
        [
            "--input",
            str(fused_ply_path),
            "--output",
            str(output_path),
            "--road-class-id",
            str(ROAD_CLASS_ID),
            "--no-postprocess",
        ],
        catch_exceptions=False,
    )
    print(result.output)
    assert result.exit_code == 0
    _assert_valid_mesh(output_path)


def test_export_camera_ground_mesh_missing_segment_id_errors(tmp_path: Path) -> None:
    fused_path = tmp_path / "no_segid.ply"
    _write_fused_ply(fused_path, with_segment_id=False)
    result = CliRunner().invoke(
        export_camera_ground_mesh,
        ["--input", str(fused_path), "--output", str(tmp_path / "out.ply")],
    )
    assert result.exit_code != 0
    assert "segment_id" in result.output
    assert not (tmp_path / "out.ply").exists()


def test_export_camera_ground_mesh_no_matching_road_class_errors(fused_ply_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "out.ply"
    result = CliRunner().invoke(
        export_camera_ground_mesh,
        ["--input", str(fused_ply_path), "--output", str(output_path), "--road-class-id", "99"],
    )
    assert result.exit_code != 0
    assert not output_path.exists()


def test_export_camera_ground_mesh_pins_fixed_parameters(
    fused_ply_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock the fixed meshing kwargs the tool hardcodes.

    They are inline literals, so a silent edit (e.g. ``2.5`` -> ``1.0`` or ``median`` -> ``mean``)
    would still produce a valid mesh and pass the smoke tests; assert the exact kwargs so drift
    fails here. Post-process knobs use ``patch_and_densify_ground_mesh`` defaults, so we only
    check it was invoked with the road point cloud.
    """
    import nurec.utils.io.export.camera_ground_mesh as cli

    captured: dict[str, dict[str, object]] = {}

    class _FakeMeshing:
        def __init__(self, **kwargs: object) -> None:
            captured["meshing"] = kwargs

        def build_mesh_from_points(
            self, points: np.ndarray, colors: np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
            verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            tris = np.array([[0, 1, 2]], dtype=np.int32)
            colors_out = np.zeros((3, 3), dtype=np.uint8) if colors is not None else None
            return verts, tris, verts, colors_out

    def _fake_patch(
        vertices: np.ndarray, triangles: np.ndarray, vertex_colors: np.ndarray | None, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        captured["postprocess"] = kwargs
        return vertices, triangles, vertex_colors

    monkeypatch.setattr(cli, "DelaunayElevationMeshingAlgorithm", _FakeMeshing)
    monkeypatch.setattr(cli, "patch_and_densify_ground_mesh", _fake_patch)

    output_path = tmp_path / "mesh_ground.ply"
    result = CliRunner().invoke(
        cli.export_camera_ground_mesh,
        ["--input", str(fused_ply_path), "--output", str(output_path), "--road-class-id", str(ROAD_CLASS_ID)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    assert captured["meshing"] == {
        "enable_downsampling": True,
        "voxel_size": 0.3,
        "min_points_per_voxel": 1,
        "smoothing_passes": 10,
        "elevation_axis": 2,
        "mesh_topology": "delaunay",
        "height_statistic": "median",
        "color_statistic": "median",
        "outlier_height_threshold": 0.35,
        "outlier_neighbor_radius": 1,
        "outlier_min_neighbors": 4,
        "max_triangle_edge_length": 2.5,
        "max_triangle_normal_angle_deg": 60.0,
        # Corridor protection is opt-in via --dataset: without it no exemption hook is supplied and
        # the mesh is built exactly as before.
        "protected_triangle_mask_fn": None,
    }

    # Post-process is invoked with the road point cloud (knobs left to helper defaults).
    assert "postprocess" in captured
    assert captured["postprocess"]["road_points_xyz"] is not None
    assert captured["postprocess"]["road_colors_rgb"] is not None
    assert captured["postprocess"]["drop_floating_components"] is True
