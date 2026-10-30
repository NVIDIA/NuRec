# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone CLI: build the camera-derived ground mesh and write it as a triangle PLY.

Loads a fused PLY -> filters road points by semantic class id -> DelaunayElevationMeshingAlgorithm
-> optional patch_and_densify_ground_mesh -> save (vertex RGB when the input has colors).

Meshing parameters are fixed in the build call below (not CLI flags) so the mesh is reproducible.
They are NOT the ``camera_ground_mesh_road`` config, which drives the road-gaussian initializer and
still carries ``max_triangle_edge_length: 1.0`` / ``color_statistic: mean``; this tool uses 2.5 and
median.

``--dataset`` enables ego-corridor protection: triangles under the rig path are exempt from the
long-edge filter up to ``--corridor-max-edge-length-m``, and every mesh component carrying corridor
area is kept rather than only the largest. Both only retain triangles built from observed road points. Elevation smoothing propagates
over triangle adjacency, so neighbouring vertex heights shift slightly (<= 0.05 mm beyond 6 m of the
path on the validation clip).

Example::

    bazel run //:run -- export-camera-ground-mesh \\
        --input /path/to/fused_004.ply \\
        --output /tmp/mesh_ground.ply \\
        --dataset /path/to/sequence.json \\
        --corridor-half-width-m 2.5 \\
        --postprocess
"""

from __future__ import annotations

import logging

import click
import numpy as np
import point_cloud_utils as pcu

from upath import UPath

from apps.av.road_mesh.corridor import (
    DEFAULT_CORRIDOR_HALF_WIDTH_M,
    DEFAULT_CORRIDOR_MAX_EDGE_LENGTH_M,
    RECOMMENDED_CORRIDOR_Z_BAND_M,
    corridor_triangle_mask,
    select_components_covering_corridor,
)
from apps.av.road_mesh.ground_mesh import (
    DelaunayElevationMeshingAlgorithm,
    patch_and_densify_ground_mesh,
)
from nurec.ncore_backends import DatasetRequest, resolve_dataset
from nurec.utils.io.ply import save_ply


logger = logging.getLogger(__name__)

MAX_TRIANGLE_EDGE_LENGTH_M = 2.5
"""Long-edge threshold for unprotected faces. A corridor cap must exceed it to have any effect."""


def load_rig_trajectory(dataset_path: str) -> np.ndarray:
    """Load the rig->world translation as an (N, 3) polyline, restricted to the sequence time range.

    Pose stores routinely extend beyond the clip; out-of-range poses would place the corridor over
    road the fused cloud never observed. Z is always kept so ``--corridor-z-band-m`` needs no
    second load path.
    """
    resolved = resolve_dataset(
        DatasetRequest(
            metadata_path=UPath(dataset_path),
            masks_component_group=None,
            cuboids_component_group=None,
        )
    )
    loader = resolved.create_sequence_loader()
    rig_world_edge = loader.pose_graph.get_edge("rig", "world")
    if rig_world_edge is None:
        raise click.ClickException(f"NCore dataset has no rig-to-world trajectory: {dataset_path}")
    if rig_world_edge.timestamps_us is None:
        raise click.ClickException(f"NCore rig-to-world trajectory has no timestamps: {dataset_path}")

    timestamps_us = np.asarray(rig_world_edge.timestamps_us)
    metadata = resolved.metadata
    in_range = (timestamps_us >= metadata.time_range_start_us) & (timestamps_us <= metadata.time_range_end_us)
    T_rig_world = np.asarray(rig_world_edge.T_source_target)[in_range]
    if len(T_rig_world) < 2:
        raise click.ClickException(
            f"Need at least two in-range rig poses to build a corridor, got {len(T_rig_world)}: {dataset_path}"
        )
    return T_rig_world[:, :3, 3].astype(np.float64)


@click.command("export-camera-ground-mesh")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--road-class-id",
    "road_class_ids",
    multiple=True,
    type=int,
    default=(0,),
    show_default=True,
    help="Semantic class id(s) treated as road. Repeatable.",
)
@click.option(
    "--postprocess/--no-postprocess",
    "postprocess",
    default=True,
    show_default=True,
    help="Run patch_and_densify_ground_mesh (drop floating + fill holes + densify + smooth) "
    "after the Delaunay mesh build. --no-postprocess emits the raw Delaunay mesh (debug).",
)
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False),
    help="NCore sequence JSON. Enables ego-corridor protection; without it the mesh is built as before.",
)
@click.option(
    "--corridor-half-width-m",
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_CORRIDOR_HALF_WIDTH_M,
    show_default=True,
    help="Half-width of the protected ego corridor in meters. Override where the road is narrower; "
    "a corridor wider than the road over-claims at the road edge.",
)
@click.option(
    "--corridor-z-band-m",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Vertical half-band in meters for corridor membership; off by default. Pass "
    f"{RECOMMENDED_CORRIDOR_Z_BAND_M:g} on stacked geometry so a lower-deck path does not select "
    "upper-deck triangles.",
)
@click.option(
    "--corridor-max-edge-length-m",
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_CORRIDOR_MAX_EDGE_LENGTH_M,
    show_default=True,
    help="Cap the longest BEV edge of a protected corridor triangle, bounding how far one exempted "
    "face may span. Above the corridor width, so a face may still span the corridor laterally. "
    "Coverage-neutral across 469 swept clips and never increases boundary-loop count. Pass a larger "
    "value to weaken the bound; a much smaller one can re-split the mesh and lose coverage.",
)
@click.option(
    "--steep-grade-threshold",
    type=float,
    default=None,
    hidden=True,
    help="Deprecated no-op, accepted so the NDAS template keeps running. The grade-triggered global "
    "edge-length override it enabled is superseded by ego-corridor protection, which fires on "
    "sparsity rather than on grade and stays local to the driven surface.",
)
@click.option(
    "--steep-max-triangle-edge-length-m",
    type=float,
    default=None,
    hidden=True,
    help="Deprecated no-op; see --steep-grade-threshold.",
)
def export_camera_ground_mesh(
    input_path: str,
    output_path: str,
    road_class_ids: tuple[int, ...],
    postprocess: bool,
    dataset_path: str | None,
    corridor_half_width_m: float,
    corridor_z_band_m: float | None,
    corridor_max_edge_length_m: float | None,
    steep_grade_threshold: float | None,
    steep_max_triangle_edge_length_m: float | None,
) -> None:
    """Build the camera ground mesh and write it as a triangle PLY."""

    if steep_grade_threshold is not None or steep_max_triangle_edge_length_m is not None:
        logger.warning(
            "--steep-grade-threshold / --steep-max-triangle-edge-length-m are deprecated no-ops: the "
            "grade-triggered global edge-length override is superseded by ego-corridor protection, "
            "which triggers on point sparsity rather than grade and stays local to the driven "
            "surface. Drop them from the workflow template."
        )

    if corridor_max_edge_length_m is not None and corridor_max_edge_length_m <= MAX_TRIANGLE_EDGE_LENGTH_M:
        raise click.BadParameter(
            f"must exceed the mesher's max edge length ({MAX_TRIANGLE_EDGE_LENGTH_M:g} m), otherwise "
            f"every face it would protect is already kept by the unprotected filter and corridor "
            f"protection silently does nothing.",
            param_hint="--corridor-max-edge-length-m",
        )

    corridor_trajectory = None
    if dataset_path is not None:
        corridor_trajectory = load_rig_trajectory(dataset_path)
        band = "off (BEV-only)" if corridor_z_band_m is None else f"{corridor_z_band_m:g} m"
        cap = "off (fully exempt)" if corridor_max_edge_length_m is None else f"{corridor_max_edge_length_m:g} m"
        logger.info(
            f"Ego corridor enabled: {len(corridor_trajectory)} rig poses, "
            f"half_width={corridor_half_width_m:g} m, z_band={band}, max_edge={cap}"
        )

    logger.info(f"Loading fused point cloud from {input_path}")
    ply = pcu.load_triangle_mesh(input_path).vertex_data
    xyz_world = ply.positions.astype(np.float32)
    color = ply.colors
    if "segment_id" not in ply.custom_attributes:
        raise click.ClickException(
            f"PLY missing 'segment_id' custom attribute: {input_path}. "
            "Tool expects fused MIR/ncore output (per-point semantic class id)."
        )
    segment_id = ply.custom_attributes["segment_id"]
    logger.info(
        f"Loaded {len(xyz_world)} points, colors={'yes' if color is not None else 'no'}, "
        f"unique segment ids: {sorted(int(v) for v in np.unique(segment_id).tolist())}"
    )

    road_mask = np.isin(segment_id, np.asarray(road_class_ids))
    road_points = xyz_world[road_mask]
    road_colors = None
    if color is not None:
        road_colors = color[road_mask][:, :3].astype(np.uint8)

    logger.info(
        f"Filtered {len(road_points)} road points (class_ids={list(road_class_ids)}) from "
        f"{len(xyz_world)} total, colors={'yes' if road_colors is not None else 'no'}"
    )
    if len(road_points) < 3:
        raise click.ClickException(
            f"Too few road points ({len(road_points)}) for meshing. Check --road-class-id and input PLY."
        )

    # Corridor policy lives here; the mesher just takes an exemption callback.
    protected_triangle_mask_fn = None
    if corridor_trajectory is not None:

        def protected_triangle_mask_fn(vertices, triangles):  # noqa: F811 - narrow local closure
            return corridor_triangle_mask(
                vertices,
                triangles,
                corridor_trajectory,
                half_width_m=corridor_half_width_m,
                z_band_m=corridor_z_band_m,
                max_edge_length_m=corridor_max_edge_length_m,
                elevation_axis=2,
            )

    # Fixed meshing parameters (see module docstring). mesh_topology is pinned to delaunay: the
    # class and lidar-config defaults are now "grid", which emits no triangles across unoccupied BEV
    # cells and never runs the long-edge filter, so corridor exemption would be inert there.
    # Smoothing/local-plane settings use the algorithm defaults.
    meshing = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=0.3,
        min_points_per_voxel=1,
        smoothing_passes=10,
        elevation_axis=2,
        mesh_topology="delaunay",
        height_statistic="median",
        color_statistic="median",
        outlier_height_threshold=0.35,
        outlier_neighbor_radius=1,
        outlier_min_neighbors=4,
        max_triangle_edge_length=MAX_TRIANGLE_EDGE_LENGTH_M,
        max_triangle_normal_angle_deg=60.0,
        protected_triangle_mask_fn=protected_triangle_mask_fn,
    )
    smooth_vertices, triangles, _, vertex_colors = meshing.build_mesh_from_points(road_points, colors=road_colors)

    logger.info(
        f"Delaunay mesh: {len(smooth_vertices)} verts, {len(triangles)} tris, "
        f"colors={'yes' if vertex_colors is not None else 'no'}"
    )

    if postprocess and len(triangles) > 0:
        logger.info("Applying patch_and_densify_ground_mesh (drop floating + fill holes + densify + smooth)")
        # Helper defaults own the post-process knobs; pass only the road point cloud (drives
        # midpoint densify coloring).
        # Corridor component retention replaces the helper's largest-component rule, so apply it
        # here and tell the helper not to prune again -- it would drop what was just retained.
        drop_floating_components = True
        if corridor_trajectory is not None:
            keep = select_components_covering_corridor(
                smooth_vertices,
                triangles,
                corridor_trajectory,
                half_width_m=corridor_half_width_m,
                z_band_m=corridor_z_band_m,
            )
            triangles = triangles[keep]
            drop_floating_components = False
            if len(triangles) == 0:
                raise click.ClickException("Corridor component selection left no triangles.")

        smooth_vertices, triangles, vertex_colors = patch_and_densify_ground_mesh(
            smooth_vertices,
            triangles,
            vertex_colors,
            road_points_xyz=road_points,
            road_colors_rgb=road_colors,
            drop_floating_components=drop_floating_components,
        )
        logger.info(f"Post-process result: {len(smooth_vertices)} verts, {len(triangles)} tris")

    if len(smooth_vertices) < 3 or len(triangles) == 0:
        raise click.ClickException(f"Output mesh degenerate ({len(smooth_vertices)} verts, {len(triangles)} tris).")

    save_ply(
        output_path,
        smooth_vertices.astype(np.float32),
        triangles=triangles.astype(np.int32),
        colors=vertex_colors,
        logger=logger,
    )
    logger.info(f"Wrote {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
    export_camera_ground_mesh()
