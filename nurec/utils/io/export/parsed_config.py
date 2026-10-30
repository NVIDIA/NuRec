# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional

import click

from omegaconf import OmegaConf

from nurec.artifact.artifact import Artifact
from nurec.config.parse import parse_untyped_config
from nurec.utils.upgrade import apply_renderer_override, upgrade_config


@click.command("export-parsed-config")
@click.option(
    "-i",
    "--input",
    "artifact_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the artifact file.",
)
@click.option("-c", "--config-name", type=str, help="Path to the config file.")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output file path. If not provided, prints to stdout.",
)
@click.option(
    # It is important to sort the keys when the goal is to compare two configs.
    "--sort-keys/--no-sort-keys",
    type=bool,
    help="Sort the keys before saving or printing.",
    default=True,
)
@click.option(
    "--upgrade",
    type=bool,
    is_flag=True,
    help="Upgrade the config to the current version of the software before exporting",
)
@click.option(
    "--renderer",
    type=str,
    default=None,
    help=(
        "Override model.renderer.name before upgrading the config (only applies with --upgrade). "
        "Required to upgrade artifacts trained with a legacy NRend renderer that has no automatic "
        "migration path (e.g. 'gsplat')."
    ),
)
@click.argument("hydra_args", nargs=-1)
def export_parsed_config(
    artifact_path: Optional[Path],
    config_name: Optional[str],
    output: Optional[Path],
    sort_keys: bool,
    upgrade: bool,
    renderer: Optional[str],
    hydra_args: list[str],
) -> None:
    """
    Parses a YAML config file or an artifact and prints the resolved configuration.

    This command can process either a USDZ artifact file or a YAML config file,
    parse and resolve the configuration, optionally upgrade it to the current version,
    and output the result to stdout or a file.

    \b
    Examples:
    \b
      # Export from artifact to stdout
      export-parsed-config --input artifact.usdz
    \b
      # Export from YAML with upgrade
      export-parsed-config --config-name config.yaml --upgrade
    \b
      # Save to file with sorted keys
      export-parsed-config --config-name config.yaml --output parsed.yaml --sort-keys
    \b
      # With Hydra arguments
      export-parsed-config --config-name config.yaml -- dataset.batch_size=32
    \b
      # Artifact trained with a legacy NRend renderer that has no automatic migration path
      export-parsed-config --input artifact.usdz --upgrade --renderer gsplat
    """
    if bool(artifact_path) == bool(config_name):
        raise click.UsageError(
            "Please provide either --input (for an usdz artifact) or --config-name (for yaml), but not both."
        )

    if artifact_path:
        if hydra_args:
            click.echo("Warning: Hydra arguments are ignored when processing an artifact file.", err=True)
        artifact = Artifact(artifact_path)
        config = OmegaConf.create(artifact.parsed_config)
    else:
        assert config_name is not None
        config = parse_untyped_config(config_name, hydra_args)

    # Upgrade config if requested.
    if upgrade:
        # Logs a warning and returns the original config if the config does not contain version information.
        config = upgrade_config(apply_renderer_override(config, renderer))

    yaml_config = OmegaConf.to_yaml(config, sort_keys=sort_keys)

    if output:
        with open(output, "w") as f:
            f.write(yaml_config)
        click.echo(f"Parsed config written to {output}")
    else:
        click.echo(yaml_config)
