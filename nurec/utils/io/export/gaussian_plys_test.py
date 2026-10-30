# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omegaconf import OmegaConf

from nurec.config.parse import dump_config, parse_typed_config
from nurec.utils.io.export import gaussian_plys as gaussian_plys_module


_TESTDATA = Path(__file__).resolve().parent / "testdata"


def _dump_stale_in_process_instant_nurec_training_config(tmp_path: Path) -> tuple[Path, Path]:
    """Write a saved training config whose in_process_instant_nurec producer is stale.

    Returns:
        The saved config path and the outer checkpoint the exporter must load.
    """
    training_config = parse_typed_config(
        config_name=str(_TESTDATA / "ncore_ds.yaml"),
        hydra_args=[
            f"dataset.path={tmp_path / 'clip.json'}",
            f"out_dir={tmp_path / 'outer-output'}",
        ],
    )
    assert "train" in training_config.mode

    parsed_config = tmp_path / "parsed.yaml"
    dump_config(str(parsed_config), training_config)

    # Stand-in for a saved nested instant_nurec producer: a schema-incomplete block whose resume checkpoint does not
    # exist, so any nested validation or materialization fails.
    missing_nested_checkpoint = tmp_path / "missing-nested-instant_nurec.ckpt"
    saved_config = OmegaConf.load(parsed_config)
    saved_config.in_process_instant_nurec = {
        "mode": "predict",
        "resume": str(missing_nested_checkpoint),
        "resume_weights_only": True,
    }
    parsed_config.write_text("# @package _global_\n\n" + OmegaConf.to_yaml(saved_config))

    reloaded_config = OmegaConf.load(parsed_config)
    assert "train" in reloaded_config.mode
    assert reloaded_config.in_process_instant_nurec.resume == str(missing_nested_checkpoint)
    assert not missing_nested_checkpoint.exists()

    outer_checkpoint = Path(training_config.ckpt_dir) / "last.ckpt"
    outer_checkpoint.parent.mkdir(parents=True)
    outer_checkpoint.touch()
    return parsed_config, outer_checkpoint


def test_export_gaussian_plys_deactivates_stale_in_process_instant_nurec_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parsed_config, outer_checkpoint = _dump_stale_in_process_instant_nurec_training_config(tmp_path)

    # The exporter must classify the saved training config as validation before
    # nested instant_nurec validation or materialization can require the stale producer.
    materialize = MagicMock(side_effect=AssertionError("inactive in_process_instant_nurec must not be materialized"))
    monkeypatch.setattr("nurec.config.parse.materialize_pretrained_instant_nurec_config", materialize)

    class FakeGaussiansSystem:
        def __init__(self) -> None:
            self.model = MagicMock()

    system = FakeGaussiansSystem()

    def make_system(name: str, config, *, load_from_checkpoint: str):
        assert config.mode == "val"
        assert config.in_process_instant_nurec is None
        assert config.resume == str(outer_checkpoint)
        assert load_from_checkpoint == str(outer_checkpoint)
        return system

    monkeypatch.setattr(gaussian_plys_module, "GaussiansSystem", FakeGaussiansSystem)
    monkeypatch.setattr(gaussian_plys_module.nurec.systems, "make", make_system)

    callback = gaussian_plys_module.export_gaussian_plys.callback
    assert callback is not None
    callback(
        config_name=str(parsed_config),
        checkpoint_name="last.ckpt",
        output_dir=str(tmp_path / "plys"),
        hydra_args=[],
        format=gaussian_plys_module.GaussianExportFormat._3DGS,
        percentage_gaussians=100.0,
    )

    materialize.assert_not_called()
    system.model.export_plys.assert_called_once()
