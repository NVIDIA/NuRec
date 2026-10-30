# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The heavy CLI imports below deliberately follow the runfiles self-repair prologue (the sys.path insert),
# so E402 (module-import-not-at-top) is expected for this file.
# ruff: noqa: E402
import os
import sys


# `multiprocessing.spawn` re-executes the entrypoint with the hermetic interpreter rather than the
# Bazel stage-2 bootstrap; under Python safe-path mode the raw child inherits neither the runfiles import hook
# nor this file's directory on `sys.path`. Add only the entrypoint's own directory so the child can import the
# sibling `nurec` package (the initial Bazel-launched process already has it, so this is a no-op there). Guard
# `__file__`: runtimes that `exec` this module without one resolve modules
# their own way and neither need nor want this prologue.
_entrypoint_file = globals().get("__file__")
if _entrypoint_file is not None:
    _entrypoint_dir = os.path.dirname(os.path.abspath(_entrypoint_file))
    if _entrypoint_dir not in sys.path:
        sys.path.insert(0, _entrypoint_dir)

import click
import click_default_group

from apps.viewer.ply_viewer import run_ply_viewer
from apps.viewer.usdz_viewer import run_viewer
from nurec import __version__
from nurec.benchmark.eval_ground_mesh import eval_ground_mesh
from nurec.benchmark.eval_rendering_metrics import eval_rendering_metrics
from nurec.benchmark.gt_from_ncore import export_benchmark_gt_from_ncore
from nurec.compression.compress_usdz import compress_usdz
from nurec.compression.decompress_usdz import decompress_usdz
from nurec.grpc.serve import serve_grpc
from nurec.metrics.compute_metrics import compute_metrics
from nurec.render.novel_trajectory.render_novel_trajectory import render_novel_trajectory
from nurec.run import main, profile_dataloader, run_script
from nurec.utils.edit.asset_edits_cli import compile_asset_edits
from nurec.utils.ffmpeg_config import (
    USER_FFMPEG_EXE_ENV,
    configure_ffmpeg,
)
from nurec.utils.io.export.artifact_structure import export_artifact_structure
from nurec.utils.io.export.camera_ground_mesh import export_camera_ground_mesh
from nurec.utils.io.export.custom_rig_trajectory import export_custom_rig_trajectory
from nurec.utils.io.export.depth import export_depth
from nurec.utils.io.export.ego_mask import export_ego_mask
from nurec.utils.io.export.external_assets import export_external_assets
from nurec.utils.io.export.gaussian_plys import export_gaussian_plys
from nurec.utils.io.export.gaussian_statistics import gaussianStatistics
from nurec.utils.io.export.gaussian_usd_asset import export_gaussian_usd_asset
from nurec.utils.io.export.ground_mesh import export_ground_mesh
from nurec.utils.io.export.mask_overlay import export_mask_overlay
from nurec.utils.io.export.mesh import export_mesh
from nurec.utils.io.export.ncore_diagnostic import export_ncore_diagnostic
from nurec.utils.io.export.ncore_tracks import export_ncore_tracks
from nurec.utils.io.export.parsed_config import export_parsed_config
from nurec.utils.io.export.point_cloud import export_point_cloud
from nurec.utils.io.export.render import render
from nurec.utils.io.export.render_grpc import render_grpc
from nurec.utils.io.export.rig_trajectories import export_rig_trajectories
from nurec.utils.io.export.sequence_tracks import export_sequence_tracks
from nurec.utils.io.export.usdz_artifact import export_usdz_artifact
from nurec.utils.upgrade.cli.artifact import upgrade_artifact
from nurec.utils.upgrade.cli.config import upgrade_config


@click.group(cls=click_default_group.DefaultGroup, default="main", default_if_no_args=True)
@click.version_option(version=str(__version__))
@click.option(
    "--ffmpeg-exe",
    type=click.Path(dir_okay=False),
    envvar=USER_FFMPEG_EXE_ENV,
    help="Use this ffmpeg executable for imageio-ffmpeg video encoding.",
)
@click.option(
    "--allow-ffmpeg-download",
    is_flag=True,
    help="Explicitly permit NuRec to download an ffmpeg binary into a local cache when no explicit executable is configured.",
)
def cli(
    ffmpeg_exe: str | None,
    allow_ffmpeg_download: bool,
):
    configure_ffmpeg(
        ffmpeg_exe=ffmpeg_exe,
        allow_download=allow_ffmpeg_download,
    )


cli.add_command(main)
cli.add_command(run_script)
cli.add_command(export_parsed_config)
cli.add_command(export_artifact_structure)
cli.add_command(upgrade_artifact)
cli.add_command(upgrade_config)
cli.add_command(export_depth)
cli.add_command(export_ego_mask)
cli.add_command(export_external_assets)
cli.add_command(export_ncore_tracks)
cli.add_command(export_mesh)
cli.add_command(export_gaussian_plys)
cli.add_command(export_gaussian_usd_asset)
cli.add_command(gaussianStatistics)
cli.add_command(export_ground_mesh)
cli.add_command(export_camera_ground_mesh)
cli.add_command(export_mask_overlay)
cli.add_command(export_point_cloud)
cli.add_command(export_rig_trajectories)
cli.add_command(export_sequence_tracks)
cli.add_command(export_usdz_artifact)
cli.add_command(export_ncore_diagnostic)
cli.add_command(export_custom_rig_trajectory)
cli.add_command(render)
cli.add_command(render_grpc)
cli.add_command(render_novel_trajectory)
cli.add_command(serve_grpc)
cli.add_command(profile_dataloader)
cli.add_command(run_viewer)
cli.add_command(run_ply_viewer)
cli.add_command(eval_ground_mesh)
cli.add_command(export_benchmark_gt_from_ncore)
cli.add_command(eval_rendering_metrics)
cli.add_command(compute_metrics)
cli.add_command(compress_usdz)
cli.add_command(decompress_usdz)
cli.add_command(compile_asset_edits)

if __name__ == "__main__":
    cli(show_default=True, max_content_width=100)
