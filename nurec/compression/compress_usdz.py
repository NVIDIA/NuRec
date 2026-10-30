# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line adapter for compact NuRec USDZ compression."""

import logging

from pathlib import Path

import click

import nurec.compression.usdz as usdz


_COMPRESSION_LEVEL_HELP = "Compression level 0..9. L0 is highest quality; higher levels are smaller and lossier."


@click.command("compress-usdz")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Input .usdz artifact.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output compact .usdz path.",
)
@click.option(
    "--opacity-threshold",
    default=usdz.DEFAULT_OPACITY_THRESHOLD,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0, max_open=True),
    help="Drop Gaussians below this sigmoid opacity. Use 0 to keep all.",
)
@click.option(
    "--level",
    "compression_level",
    default=usdz.DEFAULT_COMPRESSION_LEVEL,
    show_default=True,
    type=click.IntRange(0, 9),
    help=_COMPRESSION_LEVEL_HELP,
)
@click.option(
    "--exclude",
    "exclude_patterns",
    multiple=True,
    help="Shell-style USDZ member pattern to drop. Directory patterns may end in '/'. May be repeated.",
)
@click.option(
    "--preserve-extra-signal-tensor",
    "preserve_extra_signal_tensors",
    multiple=True,
    type=click.Choice(usdz.gaussian_codec.EXTRA_SIGNAL_TENSOR_NAMES),
    help="Per-Gaussian *_extra_signal tensor leaf to preserve by value. May be repeated.",
)
@click.option(
    "--renderer",
    default=None,
    type=str,
    help=(
        "Override model.renderer.name before upgrading the config. Required to compress "
        "artifacts trained with a legacy NRend renderer that has no automatic migration path "
        "(e.g. 'gsplat')."
    ),
)
def compress_usdz(
    input_path: Path,
    output_path: Path | None,
    opacity_threshold: float,
    compression_level: int,
    exclude_patterns: tuple[str, ...],
    preserve_extra_signal_tensors: tuple[str, ...],
    renderer: str | None,
) -> None:
    """Compress a .usdz artifact into a compact .usdz archive."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        output_path = usdz.compress_usdz(
            input_path,
            output_path,
            opacity_threshold=opacity_threshold,
            compression_level=compression_level,
            preserve_extra_signal_tensors=preserve_extra_signal_tensors,
            exclude_patterns=exclude_patterns,
            renderer=renderer,
        )
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    input_mib = input_path.stat().st_size / 2**20
    output_mib = output_path.stat().st_size / 2**20
    print(f"\ninput:  {input_path} ({input_mib:.1f} MiB)")
    print(f"output: {output_path} ({output_mib:.1f} MiB)")
    print(f"ratio:  {output_mib / input_mib:.1%}")


if __name__ == "__main__":
    compress_usdz()
