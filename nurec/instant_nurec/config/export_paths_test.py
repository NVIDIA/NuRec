# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from pathlib import Path

from nurec.instant_nurec.config.export_paths import resolve_export_ply_base_dir


class TestExportPlyPaths(unittest.TestCase):
    def test_resolve_export_ply_base_dir_none_uses_out_dir(self) -> None:
        self.assertEqual(
            resolve_export_ply_base_dir(None, out_dir="/results", run_id="instant_nurec"),
            "/results/instant_nurec",
        )

    def test_resolve_export_ply_base_dir_dataset_directory(self) -> None:
        meta = {"ncore_json_path": Path("/data/h81/input/clipgt-abc.json")}

        self.assertEqual(
            resolve_export_ply_base_dir("dataset_directory", out_dir="/results", run_id="instant_nurec", meta=meta),
            "/data/h81/input/instant_nurec",
        )

    def test_resolve_export_ply_base_dir_dataset_directory_requires_meta(self) -> None:
        with self.assertRaisesRegex(ValueError, "ncore_json_path"):
            resolve_export_ply_base_dir("dataset_directory", out_dir="/results", run_id="instant_nurec")

    def test_resolve_export_ply_base_dir_custom(self) -> None:
        self.assertEqual(
            resolve_export_ply_base_dir("/custom/ply/root", out_dir="/results", run_id="ignored"),
            "/custom/ply/root",
        )


if __name__ == "__main__":
    unittest.main()
