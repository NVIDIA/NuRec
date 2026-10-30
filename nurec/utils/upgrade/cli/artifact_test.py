# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zipfile

from click.testing import CliRunner
from omegaconf import OmegaConf

from nurec.utils.upgrade.cli import artifact as artifact_cli


def test_float_shaped_git_sha_is_preserved_as_string_in_artifact_config(tmp_path, monkeypatch):
    input_file = tmp_path / "input.usdz"
    output_file = tmp_path / "output.usdz"

    with zipfile.ZipFile(input_file, "w") as artifact_zip:
        artifact_zip.writestr("parsed_config.yaml", "version:\n  git_commit_sha_short: 4f2d6cbe\n")
        artifact_zip.writestr("checkpoint.ckpt", b"checkpoint")
        artifact_zip.writestr("datasource_summary.json", "{}")
        artifact_zip.writestr("passthrough.txt", "kept")

    upgraded_config = OmegaConf.create(
        {
            "version": {
                "git_commit_sha_short": "78e48072",
                "version_major": 1,
                "version_minor": 2,
                "version_patch": 3,
                "git_tree_dirty": False,
                "git_commit_date": "2025-01-01T00:00:00",
            }
        }
    )
    monkeypatch.setattr(artifact_cli, "upgrade_config", lambda orig_config, version_target: upgraded_config)
    monkeypatch.setattr(artifact_cli, "upgrade_model", lambda *args, **kwargs: {})
    monkeypatch.setattr(artifact_cli.DataSourceSummary, "from_dict", lambda *args, **kwargs: object())
    monkeypatch.setattr(artifact_cli.torch, "load", lambda *args, **kwargs: {"state_dict": {"model.foo": "bar"}})
    monkeypatch.setattr(
        artifact_cli.torch, "save", lambda checkpoint, checkpoint_file: checkpoint_file.write(b"checkpoint")
    )

    result = CliRunner().invoke(
        artifact_cli.upgrade_artifact,
        ["--input", str(input_file), "--output", str(output_file), "--target-version", "1.2.3"],
    )

    assert result.exit_code == 0, result.output
    with zipfile.ZipFile(output_file, "r") as artifact_zip:
        serialized = artifact_zip.read("parsed_config.yaml").decode("utf-8")
        assert artifact_zip.read("passthrough.txt").decode("utf-8") == "kept"
    assert "git_commit_sha_short: '78e48072'" in serialized
    reparsed = OmegaConf.create(serialized)
    assert reparsed.version.git_commit_sha_short == "78e48072"
    assert isinstance(reparsed.version.git_commit_sha_short, str)
