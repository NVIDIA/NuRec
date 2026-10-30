# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from omegaconf import DictConfig, OmegaConf

from nurec.config.version import Version
from nurec.utils.upgrade.config import upgrade_config


def _config(**model) -> DictConfig:
    cfg = OmegaConf.create(
        {"version": {"version_major": 26, "version_minor": 10, "version_patch": 103}, "model": model}
    )
    assert isinstance(cfg, DictConfig)
    return cfg


class LegacyInstantNuRecNamesUpgradeTest(unittest.TestCase):
    TARGET = Version.from_components(26, 10, 103)

    def test_saved_artifact_names_upgrade_even_at_current_version(self) -> None:
        cfg = OmegaConf.create(
            {
                "version": {"version_major": 26, "version_minor": 10, "version_patch": 103},
                "dataset": {
                    "name": "nrm",
                    "train": {"name": "nrm-ncore"},
                    "val": {"name": "nrm-mixed", "datasets": [{"name": "nrm-dataverse"}]},
                },
                "system": {"name": "base-nrm-system"},
                "model": {
                    "layers": {
                        "background": {"initialization": {"name": "nrm-ply"}},
                        "road": {"initialization": {"name": "in-process-nrm-road"}},
                    }
                },
            }
        )
        out = upgrade_config(cfg, version_target=self.TARGET)
        self.assertEqual(out.dataset.name, "instant_nurec")
        self.assertEqual(out.dataset.train.name, "instant_nurec-ncore")
        self.assertEqual(out.dataset.val.datasets[0].name, "instant_nurec-dataverse")
        self.assertEqual(out.system.name, "base-instant_nurec-system")
        self.assertEqual(out.model.layers.background.initialization.name, "instant_nurec-ply")
        self.assertEqual(out.model.layers.road.initialization.name, "in-process-instant_nurec-road")
        self.assertEqual(cfg.model.layers.background.initialization.name, "nrm-ply")

    def test_names_upgrade_without_version_and_leave_unrelated_names_alone(self) -> None:
        cfg = OmegaConf.create(
            {
                "model": {"layers": {"background": {"initialization": {"name": "in-process-nrm"}}}},
                "other": {"name": "normal"},
            }
        )
        out = upgrade_config(cfg, version_target=self.TARGET)
        self.assertEqual(out.model.layers.background.initialization.name, "in-process-instant_nurec")
        self.assertEqual(out.other.name, "normal")


class TranslateWorldToNurecUpgradeTest(unittest.TestCase):
    TARGET = Version.from_components(26, 10, 104)

    def test_old_key_is_renamed_under_every_initialization_node(self) -> None:
        cfg = _config(
            initialization={"name": "ply", "translate_world_to_nre": False},
            layers={"background": {"initialization": {"name": "instant_nurec_ply", "translate_world_to_nre": True}}},
        )
        out = upgrade_config(cfg, version_target=self.TARGET)
        self.assertEqual(
            OmegaConf.to_container(out.model),
            {
                "initialization": {"name": "ply", "translate_world_to_nurec": False},
                "layers": {
                    "background": {"initialization": {"name": "instant_nurec_ply", "translate_world_to_nurec": True}}
                },
            },
        )
        self.assertEqual(out.version.version_string, "26.10.104")

    def test_new_key_is_unchanged(self) -> None:
        cfg = _config(initialization={"translate_world_to_nurec": True})
        out = upgrade_config(cfg, version_target=self.TARGET)
        self.assertEqual(OmegaConf.to_container(out.model), {"initialization": {"translate_world_to_nurec": True}})

    def test_equal_duplicates_collapse(self) -> None:
        cfg = _config(initialization={"translate_world_to_nurec": True, "translate_world_to_nre": True})
        out = upgrade_config(cfg, version_target=self.TARGET)
        self.assertEqual(OmegaConf.to_container(out.model), {"initialization": {"translate_world_to_nurec": True}})

    def test_conflicting_duplicates_are_rejected(self) -> None:
        cfg = _config(initialization={"translate_world_to_nurec": True, "translate_world_to_nre": False})
        with self.assertRaisesRegex(ValueError, "translate_world_to_nre"):
            upgrade_config(cfg, version_target=self.TARGET)

    def test_missing_mandatory_values_are_skipped(self) -> None:
        cfg = _config(initialization={"translate_world_to_nre": True, "path": "???"}, out_dir="???")
        out = upgrade_config(cfg, version_target=self.TARGET)
        self.assertTrue(out.model.initialization.translate_world_to_nurec)
        self.assertTrue(OmegaConf.is_missing(out.model.initialization, "path"))

    def test_older_configs_reach_the_rename_across_the_version_gap(self) -> None:
        cfg = _config(initialization={"translate_world_to_nre": True})
        cfg.version.version_minor = 9
        out = upgrade_config(cfg, version_target=self.TARGET)
        self.assertIn("translate_world_to_nurec", out.model.initialization)


if __name__ == "__main__":
    unittest.main()
