# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI surface of the benchmark ground-truth exporter."""

from click.testing import CliRunner

from nurec.benchmark.gt_from_ncore import export_benchmark_gt_from_ncore


def test_frame_name_width_option_is_available() -> None:
    result = CliRunner().invoke(export_benchmark_gt_from_ncore, ["--help"])

    assert result.exit_code == 0
    assert "--frame-name-width" in result.output


def test_frame_name_width_rejects_a_negative_width() -> None:
    """A negative width is a ValueError inside the format spec, so reject it at the CLI boundary."""
    result = CliRunner().invoke(export_benchmark_gt_from_ncore, ["--frame-name-width", "-1"])

    assert result.exit_code != 0
    assert "frame-name-width" in result.output
