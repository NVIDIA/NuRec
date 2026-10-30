# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that example scripts can be run via run_script."""

from pathlib import Path

from click.testing import CliRunner
from python.runfiles import runfiles

from nurec.run.run_script import run_script


def test_examples_script() -> None:
    """Test that all example scripts run successfully."""
    # Discover all Python example scripts in the examples directory
    examples_path = Path(runfiles.Create().Rlocation("_main/nurec/libs/losses/examples"))
    if not examples_path.exists():
        raise AssertionError(
            f"Examples directory not found. This is an issue with your filesystem/test suite. Missing {examples_path=}"
        )

    example_scripts = sorted(examples_path.glob("*.py"))
    assert len(example_scripts) > 0, "No example scripts found in nurec/libs/losses/examples"

    for script_path in example_scripts:
        print(f"\n{'=' * 60}")
        print(f"Testing example script: {script_path.name}")
        print("=" * 60)

        result = CliRunner().invoke(
            run_script,
            [str(script_path)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, (
            f"Script {script_path.name} failed with exit code {result.exit_code}\nOutput: {result.output}"
        )
