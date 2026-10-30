#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# mask_annotator_test.py - Combined non-GUI tests for Mask Annotator

from click.testing import CliRunner

from apps.av.avmask_annotator.mask_annotator import run_mask_annotator


# CLI tests
def test_help_option():
    runner = CliRunner()
    result = runner.invoke(run_mask_annotator, ["--help"])
    assert result.exit_code == 0
    assert "Options" in result.output


def test_version_option():
    runner = CliRunner()
    result = runner.invoke(run_mask_annotator, ["--version"])
    assert result.exit_code == 0
    assert "Mask Annotator v" or "Mask Annotator version" in result.output


# CLI option parsing errors
def test_invalid_window_size():
    runner = CliRunner()
    result = runner.invoke(run_mask_annotator, ["--window-size", "bad_format"])
    assert result.exit_code != 0
    assert "Invalid window size format" in result.output


def test_invalid_window_pos():
    runner = CliRunner()
    result = runner.invoke(run_mask_annotator, ["--window-pos", "bad_format"])
    assert result.exit_code != 0
    assert "Invalid window position format" in result.output
