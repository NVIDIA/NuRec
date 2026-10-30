# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import click

from nurec.utils.io.export.ncore_diagnostic import export_ncore_diagnostic_func


@click.command("export-ncore-benchmark-gt")
@click.option(
    "--dataset-path",
    type=str,
    help="Path to an NCore sequence metadata file",
    required=False,
)
@click.option("--poses-component-group", type=str, help="Component group for 'poses'", default="default")
@click.option("--intrinsics-component-group", type=str, help="Component group for 'intrinsics'", default="default")
@click.option("--masks-component-group", type=str, help="Component group for 'masks'", default="default")
@click.option("--cuboids-component-group", type=str, help="Component group for 'cuboids'", default="default")
@click.option(
    "--output-dir",
    type=str,
    help="Path to an (preferably new) output directory to export the requested data to",
    required=False,
)
@click.option(
    "--frame-step-camera",
    type=int,
    help="Frame step to use when exporting camera RGB frames / images (>1 means frame skipping)",
    default=50,
)
@click.option(
    "--frame-step-lidar",
    type=int,
    help="Frame step to use when exporting Lidar frames / spins (>1 means frame skipping)",
    default=50,
)
@click.option(
    "--frame-name-width",
    type=click.IntRange(min=0),
    default=0,
    help=(
        "Zero-pad exported camera frame filenames to this many digits. Default 0 leaves them "
        "unpadded. Use 20 to match the render command, which pads frame-end timestamps to the "
        "full int64 microsecond range, so eval-rendering-metrics pairs renders with this ground "
        "truth by filename stem."
    ),
)
@click.option(
    "--per-class-binary-masks",
    multiple=True,
    type=str,
    help=(
        "Export per-frame BINARY (0/255) masks for the named semseg class. "
        "ONE mask per (class, camera, frame): 255 where the pixel is that class, "
        "0 elsewhere. Repeatable: pass once per class."
    ),
)
def export_benchmark_gt_from_ncore(
    dataset_path: Optional[str],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
    output_dir: str,
    frame_step_camera: int,
    frame_step_lidar: int,
    frame_name_width: int,
    per_class_binary_masks: tuple[str, ...],
    explicit_path_pattern: Optional[str] = None,
) -> None:
    """Exports ground-truth data needed for benchmarking novel-view prediction"""
    export_ncore_diagnostic_func(
        explicit_path_pattern=explicit_path_pattern,
        dataset_path=dataset_path,
        poses_component_group=poses_component_group,
        intrinsics_component_group=intrinsics_component_group,
        masks_component_group=masks_component_group,
        cuboids_component_group=cuboids_component_group,
        output_dir=output_dir,
        frame_step_camera=frame_step_camera,
        frame_step_lidar=frame_step_lidar,
        frame_naming="timestamp",
        frame_name_width=frame_name_width,
        video_fps=15,
        export_meta=False,
        export_camera_images=True,
        export_lidar_points=False,
        export_lidar_points_fused=False,
        export_semantic_labelmaps=False,
        export_per_class_binary_masks=per_class_binary_masks,
        export_semantic_overlays=False,
        export_ego_masks=True,
        export_ego_mask_overlays=False,
        invert_mask=False,
        format="image",
        export_all=False,
    )
