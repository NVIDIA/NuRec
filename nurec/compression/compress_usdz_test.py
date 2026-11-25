# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

from click.testing import CliRunner

from nurec.compression.compress_usdz import compress_usdz


def test_directory_input_preserves_relative_artifact_path(tmp_path):
    input_root = tmp_path / "input"
    source = input_root / "usd_out" / "last.usdz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    output_root = tmp_path / "output"

    def write_compact(input_path, output_path, **_kwargs):
        assert input_path == source
        output_path.write_bytes(b"compact")
        return output_path

    with mock.patch("nurec.compression.compress_usdz.usdz.compress_usdz", side_effect=write_compact):
        result = CliRunner().invoke(
            compress_usdz,
            ["--input", str(input_root), "--output-root", str(output_root)],
        )

    assert result.exit_code == 0, result.output
    assert (output_root / "usd_out" / "last.usdz").read_bytes() == b"compact"
    assert (output_root / "usd_out" / "last.compression.json").exists()
    assert source.read_bytes() == b"original"


def test_directory_input_writes_compression_sidecar(tmp_path):
    import json

    input_root = tmp_path / "input"
    source = input_root / "usd_out" / "last.usdz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    output_root = tmp_path / "output"

    def write_compact(input_path, output_path, **_kwargs):
        output_path.write_bytes(b"compact")
        return output_path

    with mock.patch("nurec.compression.compress_usdz.usdz.compress_usdz", side_effect=write_compact):
        result = CliRunner().invoke(
            compress_usdz,
            [
                "--input",
                str(input_root),
                "--output-root",
                str(output_root),
                "--level",
                "3",
                "--opacity-threshold",
                "0",
                "--preserve-extra-signal-tensor",
                "extra_signal",
                "--exclude",
                "debug/*",
            ],
        )

    assert result.exit_code == 0, result.output
    sidecar_path = output_root / "usd_out" / "last.compression.json"
    assert sidecar_path.exists()
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert data["level"] == 3
    assert data["opacity_threshold"] == 0.0
    assert data["preserve_extra_signal_tensors"] == ["extra_signal"]
    assert data["exclude"] == ["debug/*"]
    assert data["compact_artifact"] == "last.usdz"


def test_directory_input_requires_one_final_artifact(tmp_path):
    input_root = tmp_path / "input"
    for directory in ["one", "two"]:
        artifact = input_root / directory / "last.usdz"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"original")

    result = CliRunner().invoke(
        compress_usdz,
        ["--input", str(input_root), "--output-root", str(tmp_path / "output")],
    )

    assert result.exit_code != 0
    assert "expected exactly one last.usdz" in result.output
