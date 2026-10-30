# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line adapter for compact NuRec USDZ decompression."""

import logging

from pathlib import Path

import click

import nurec.compression.usdz as usdz


@click.command("decompress-usdz")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Input compact .usdz artifact.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output eval-ready .usdz artifact with reconstructed checkpoint.",
)
def decompress_usdz(input_path: Path, output_path: Path) -> None:
    """Restore an eval-ready USDZ artifact from lossy compact tensors."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        usdz.decompress_usdz(input_path, output_path)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    input_mib = input_path.stat().st_size / 2**20
    output_mib = output_path.stat().st_size / 2**20
    print(f"\ninput:  {input_path} ({input_mib:.1f} MiB)")
    print(f"output: {output_path} ({output_mib:.1f} MiB)")


if __name__ == "__main__":
    decompress_usdz()
