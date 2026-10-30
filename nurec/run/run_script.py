# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
import sys

from pathlib import Path

import click

from python.runfiles import runfiles


@click.command()
@click.argument("script_path", type=str)
def run_script(script_path: str):
    """Run a custom Python script with NuRec environment."""
    # Resolve script path - try multiple locations
    script_file = Path(script_path)

    # Try absolute path first
    if script_file.is_absolute() and script_file.exists():
        pass  # Already resolved
    elif script_file.exists():
        # Relative path that exists in current directory
        script_file = script_file.absolute()
    else:
        # Resolve via Bazel runfiles library
        rf = runfiles.Create()
        rf_path = rf.Rlocation(script_path)
        if not rf_path:
            raise click.ClickException(f"Script not found: {script_path}")
        script_file = Path(rf_path)

    # Final validation - try to open the file
    try:
        with open(str(script_file), "r"):
            pass  # File is readable
    except (FileNotFoundError, IOError):
        raise click.ClickException(f"Script not found: {script_file}")

    # Set up argv for the script
    sys.argv = [str(script_file)]

    # Run the script as __main__
    runpy.run_path(str(script_file), run_name="__main__")
