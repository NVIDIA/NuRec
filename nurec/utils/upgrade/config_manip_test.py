# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from omegaconf import OmegaConf

from nurec.utils.upgrade.config_manip import (
    copy_config_key,
    exists_config_key,
    insert_config_key,
    move_config_key,
    remove_config_key,
)


class DictConfigManipulationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create({"a": {"b": {"c": "c_value"}}, "x": {"y": {"z": None}}})

    def test_exists_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        self.assertTrue(exists_config_key(cfg, "a"))
        self.assertTrue(exists_config_key(cfg, "a.b"))
        self.assertTrue(exists_config_key(cfg, "a.b.c"))
        self.assertTrue(exists_config_key(cfg, "x.y.z"))  # Makes sure None values are handled correctly
        self.assertFalse(exists_config_key(cfg, "a.b.missing"))
        self.assertFalse(exists_config_key(cfg, "missing"))
        self.assertEqual(cfg, self.cfg)

    def test_remove_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        remove_config_key(cfg, "a.b.c")
        self.assertEqual(cfg, {"a": {"b": {}}, "x": {"y": {"z": None}}})
        remove_config_key(cfg, "x.y.z")
        self.assertEqual(cfg, {"a": {"b": {}}, "x": {"y": {}}})

    def test_remove_intermediate_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        remove_config_key(cfg, "a.b")
        self.assertEqual(cfg, {"a": {}, "x": {"y": {"z": None}}})

    def test_remove_missing_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        remove_config_key(cfg, "missing.key")  # Should do nothing
        self.assertEqual(cfg, self.cfg)
        remove_config_key(cfg, "a.b.missing")  # Should do nothing
        self.assertEqual(cfg, self.cfg)

    def test_insert_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        insert_config_key(cfg, "a.p.q", "q_value")
        self.assertEqual(cfg, {"a": {"b": {"c": "c_value"}, "p": {"q": "q_value"}}, "x": {"y": {"z": None}}})

    def test_insert_existing_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        insert_config_key(cfg, "a.b.c", "altered")
        self.assertEqual(cfg, {"a": {"b": {"c": "altered"}}, "x": {"y": {"z": None}}})

    def test_copy_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        copy_config_key(cfg, "a.b.c", "a.p.q", "default")
        self.assertEqual(cfg, {"a": {"b": {"c": "c_value"}, "p": {"q": "c_value"}}, "x": {"y": {"z": None}}})

    def test_copy_missing_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        copy_config_key(cfg, "a.missing", "a.p.q", "default_value")
        self.assertEqual(cfg, {"a": {"b": {"c": "c_value"}, "p": {"q": "default_value"}}, "x": {"y": {"z": None}}})

    def test_move_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        move_config_key(cfg, "a.b.c", "a.p.q", "default")
        self.assertEqual(cfg, {"a": {"b": {}, "p": {"q": "c_value"}}, "x": {"y": {"z": None}}})

    def test_move_missing_config_key(self):
        cfg = OmegaConf.create(self.cfg)  # Makes a deep copy
        move_config_key(cfg, "a.missing", "a.p.q", "default")
        self.assertEqual(cfg, {"a": {"b": {"c": "c_value"}, "p": {"q": "default"}}, "x": {"y": {"z": None}}})
