# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line adapter for compact NuRec USDZ compression."""

import json
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
    help="Input .usdz artifact, or a directory containing exactly one last.usdz.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output compact .usdz path.",
)
@click.option(
    "--output-root",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory. For directory input, preserve the artifact's relative path.",
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
    output_root: Path | None,
    opacity_threshold: float,
    compression_level: int,
    exclude_patterns: tuple[str, ...],
    preserve_extra_signal_tensors: tuple[str, ...],
    renderer: str | None,
) -> None:
    """Compress a .usdz artifact into a compact .usdz archive."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if output_path is not None and output_root is not None:
        raise click.UsageError("--output and --output-root are mutually exclusive")
    if input_path.is_dir():
        matches = sorted(input_path.rglob("last.usdz"))
        if len(matches) != 1:
            raise click.ClickException(f"expected exactly one last.usdz under {input_path}, found {len(matches)}")
        source_root = input_path
        input_path = matches[0]
    else:
        source_root = input_path.parent
    if output_root is not None:
        output_path = output_root / input_path.relative_to(source_root)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
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

    if output_root is not None:
        sidecar_path = output_path.with_name(f"{output_path.stem}.compression.json")
        sidecar_data = {
            "level": compression_level,
            "opacity_threshold": opacity_threshold,
            "renderer": renderer,
            "preserve_extra_signal_tensors": list(preserve_extra_signal_tensors),
            "exclude": list(exclude_patterns),
            "compact_artifact": output_path.name,
        }
        with sidecar_path.open("w", encoding="utf-8") as handle:
            json.dump(sidecar_data, handle, indent=2)
            handle.write("\n")

    input_mib = input_path.stat().st_size / 2**20
    output_mib = output_path.stat().st_size / 2**20
    print(f"\ninput:  {input_path} ({input_mib:.1f} MiB)")
    print(f"output: {output_path} ({output_mib:.1f} MiB)")
    print(f"ratio:  {output_mib / input_mib:.1%}")


if __name__ == "__main__":
    compress_usdz()
