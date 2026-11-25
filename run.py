# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The CLI imports below deliberately follow the runfiles self-repair prologue (the sys.path insert),
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

import pkgutil

import click
import click_default_group

from nurec import __version__
from nurec.utils.ffmpeg_config import (
    USER_FFMPEG_EXE_ENV,
    configure_ffmpeg,
)


# Subcommand name -> "module:attribute" of its click command. A command's module is imported only when
# that command is looked up, so every entry point pays for its own imports instead of every command's
# (the viewer, training and evaluation stacks are several seconds of imports a render server never uses).
LAZY_COMMANDS: dict[str, str] = {
    "main": "nurec.run.main:main",
    "run-script": "nurec.run.run_script:run_script",
    "export-parsed-config": "nurec.utils.io.export.parsed_config:export_parsed_config",
    "export-artifact-structure": "nurec.utils.io.export.artifact_structure:export_artifact_structure",
    "upgrade-artifact": "nurec.utils.upgrade.cli.artifact:upgrade_artifact",
    "upgrade-config": "nurec.utils.upgrade.cli.config:upgrade_config",
    "export-depth": "nurec.utils.io.export.depth:export_depth",
    "export-ego-mask": "nurec.utils.io.export.ego_mask:export_ego_mask",
    "export-external-assets": "nurec.utils.io.export.external_assets:export_external_assets",
    "export-ncore-tracks": "nurec.utils.io.export.ncore_tracks:export_ncore_tracks",
    "export-mesh": "nurec.utils.io.export.mesh:export_mesh",
    "export-gaussian-plys": "nurec.utils.io.export.gaussian_plys:export_gaussian_plys",
    "export-gaussian-usd-asset": "nurec.utils.io.export.gaussian_usd_asset:export_gaussian_usd_asset",
    "gaussian-statistics": "nurec.utils.io.export.gaussian_statistics:gaussianStatistics",
    "export-ground-mesh": "nurec.utils.io.export.ground_mesh:export_ground_mesh",
    "export-camera-ground-mesh": "nurec.utils.io.export.camera_ground_mesh:export_camera_ground_mesh",
    "export-mask-overlay": "nurec.utils.io.export.mask_overlay:export_mask_overlay",
    "export-point-cloud": "nurec.utils.io.export.point_cloud:export_point_cloud",
    "export-rig-trajectories": "nurec.utils.io.export.rig_trajectories:export_rig_trajectories",
    "export-sequence-tracks": "nurec.utils.io.export.sequence_tracks:export_sequence_tracks",
    "export-usdz-artifact": "nurec.utils.io.export.usdz_artifact:export_usdz_artifact",
    "export-ncore-diagnostic": "nurec.utils.io.export.ncore_diagnostic:export_ncore_diagnostic",
    "export-custom-rig-trajectory": "nurec.utils.io.export.custom_rig_trajectory:export_custom_rig_trajectory",
    "render": "nurec.utils.io.export.render:render",
    "render-grpc": "nurec.utils.io.export.render_grpc:render_grpc",
    "render-novel-trajectory": "nurec.render.novel_trajectory.render_novel_trajectory:render_novel_trajectory",
    "serve-grpc": "nurec.grpc.serve:serve_grpc",
    "profile-dataloader": "nurec.run.profile_dataloader:profile_dataloader",
    "viewer": "apps.viewer.usdz_viewer:run_viewer",
    "ply_viewer": "apps.viewer.ply_viewer:run_ply_viewer",
    "eval-ground-mesh": "nurec.benchmark.eval_ground_mesh:eval_ground_mesh",
    "export-ncore-benchmark-gt": "nurec.benchmark.gt_from_ncore:export_benchmark_gt_from_ncore",
    "eval-rendering-metrics": "nurec.benchmark.eval_rendering_metrics:eval_rendering_metrics",
    "compute-metrics": "nurec.metrics.compute_metrics:compute_metrics",
    "compress-usdz": "nurec.compression.compress_usdz:compress_usdz",
    "decompress-usdz": "nurec.compression.decompress_usdz:decompress_usdz",
    "compile-asset-edits": "nurec.utils.edit.asset_edits_cli:compile_asset_edits",
}


class LazyGroup(click_default_group.DefaultGroup):
    """A default group whose subcommands are imported from `lazy_commands` on first lookup.

    A command added with `add_command` takes precedence over the lazy entry of the same name, so a composition
    root can install its own variant without importing the public one; `--help` and name completion list both.
    """

    def __init__(self, *args, lazy_commands: dict[str, str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_commands = dict(lazy_commands or {})

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(super().list_commands(ctx)) | set(self.lazy_commands))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name not in self.commands:
            # DefaultGroup routes an unknown first argument to the default command, so that is the one to load then.
            lazy_name = cmd_name if cmd_name in self.lazy_commands else self.default_cmd_name
            if lazy_name in self.lazy_commands:
                self.add_command(pkgutil.resolve_name(self.lazy_commands.pop(lazy_name)))
        return super().get_command(ctx, cmd_name)


@click.group(cls=LazyGroup, lazy_commands=LAZY_COMMANDS, default="main", default_if_no_args=True)
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


if __name__ == "__main__":
    cli(show_default=True, max_content_width=100)
