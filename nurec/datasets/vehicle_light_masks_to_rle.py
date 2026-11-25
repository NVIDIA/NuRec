# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line adapter that adds run-length encoded copies of a clip's semantic masks."""

from pathlib import Path

import click

from nurec.datasets.vehicle_light import convert_pixel_masks_to_rle


@click.command("vehicle-light-masks-to-rle")
@click.option(
    "--mask-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Mask directory holding one subdirectory of PNG masks per camera, or the PNGs directly for a single camera.",
)
@click.option(
    "--manifest",
    "manifest_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Schema-4 mask manifest.json, rewritten in place with label_schema.mask_format set to rle.",
)
@click.option("--overwrite", is_flag=True, help="Replace RLE files left behind by an earlier, interrupted run.")
def vehicle_light_masks_to_rle(mask_dir: Path, manifest_path: Path, overwrite: bool) -> None:
    """Write a .rle.npz next to every semantic PNG mask and switch the manifest to RLE, keeping the PNGs."""
    try:
        converted = convert_pixel_masks_to_rle(mask_dir, manifest_path, overwrite=overwrite)
    except FileExistsError as exc:
        raise click.ClickException(f"{exc}, rerun with --overwrite to replace it") from exc
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    print(f"converted {converted} masks, {manifest_path} now has label_schema.mask_format=rle")


if __name__ == "__main__":
    vehicle_light_masks_to_rle()
