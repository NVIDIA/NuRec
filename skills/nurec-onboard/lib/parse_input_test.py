# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``lib/parse_input.py``."""

from __future__ import annotations

import tempfile
import unittest

from pathlib import Path
from typing import Any

import yaml

from lib.parse_input import REQUIRED_PROFILE_SECTIONS, load_defaults, parse_input


REPO_ROOT = Path(__file__).parents[1]
_SKELETON = REPO_ROOT / "nurec-onboard" / "steps" / "audit-dataset" / "input-skeleton.yaml"
_EXAMPLES = (
    REPO_ROOT / "examples" / "input-pandaset.yaml",
    REPO_ROOT / "examples" / "input-waymo.yaml",
)


def _base_profile() -> dict[str, Any]:
    """Return a profile carrying every required section.

    Section *contents* are not validated by ``parse_input`` (that is
    ``audit_dataset_validator``'s job), so placeholder values suffice.
    """
    return {section: {} for section in REQUIRED_PROFILE_SECTIONS}


def _base_input() -> dict[str, Any]:
    """Return a minimally valid workflow input."""
    return {
        "data": {
            "name": "demo_dataset",
            "path": "/tmp/demo",
            "description": "Demo dataset.",
            "profile": _base_profile(),
        },
        "ncore": {
            "camera_ids": ["front"],
            "lidar_ids": ["top"],
            "seek_sec": 1.0,
            "duration_sec": 2.0,
            "converter_dir": "/tmp/ncore-demo_dataset",
            "output_dir": "/tmp/ncore-demo_dataset-out",
        },
        "nurec": {
            "camera_ids": ["front"],
            "lidar_ids": ["top"],
            "output_dir": "/tmp/ncore-demo_dataset-nurec-out",
        },
        "audit": {
            "enabled": True,
            "output_dir": "/tmp/audit",
            "fail_on_unresolved": True,
        },
    }


class ProfileSchemaSyncTest(unittest.TestCase):
    """Pin ``REQUIRED_PROFILE_SECTIONS`` to the shipped profile authorities.

    ``REQUIRED_PROFILE_SECTIONS`` has two consumers (``parse_input`` and
    ``audit_dataset_validator``) and a comment asking authors to keep it in
    step with the audit input skeleton. These tests make that request fail
    loudly instead of drifting silently.
    """

    def test_skeleton_profile_sections_match(self) -> None:
        """The audit input skeleton declares exactly the required sections."""
        skeleton = yaml.safe_load(_SKELETON.read_text())
        self.assertEqual(
            set(skeleton["data"]["profile"]),
            set(REQUIRED_PROFILE_SECTIONS),
            f"{_SKELETON.name} and REQUIRED_PROFILE_SECTIONS disagree; update both together",
        )

    def test_shipped_examples_carry_a_complete_profile(self) -> None:
        """Both shipped examples parse, so the profile gate cannot strand them."""
        for example in _EXAMPLES:
            with self.subTest(example=example.name):
                parsed = parse_input(example)
                self.assertEqual(set(parsed["data"]["profile"]), set(REQUIRED_PROFILE_SECTIONS))


class ParseInputTest(unittest.TestCase):
    """Exercise workflow input preflight validation."""

    def _parse_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Write and parse ``doc`` through the public parser.

        Args:
            doc: Input document to serialize as YAML.

        Returns:
            Parsed input document.
        """
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "input.yaml"
            path.write_text(yaml.safe_dump(doc, sort_keys=False))
            return parse_input(path)

    def assertParseError(self, doc: dict[str, Any], *substrings: str) -> str:
        """Assert parsing fails with all requested substrings.

        Args:
            doc: Input document to serialize as YAML.
            substrings: Substrings expected in the exception message.

        Returns:
            The exception message.
        """
        with self.assertRaises((TypeError, ValueError)) as ctx:
            self._parse_doc(doc)
        message = str(ctx.exception)
        for substring in substrings:
            self.assertIn(substring, message)
        return message

    def test_new_profile_scope_passes(self) -> None:
        """New profile-scoped sensor and time fields pass validation."""
        parsed = self._parse_doc(_base_input())
        defaults = load_defaults()
        self.assertEqual(parsed["data"]["name"], "demo_dataset")
        self.assertIn("profile", parsed["data"])
        self.assertEqual(parsed["ncore"]["camera_ids"], ["front"])
        self.assertEqual(parsed["ncore"]["lidar_ids"], ["top"])
        self.assertEqual(parsed["ncore"]["seek_sec"], 1.0)
        self.assertEqual(parsed["ncore"]["duration_sec"], 2.0)
        self.assertEqual(parsed["nurec"]["lidar_ids"], ["top"])
        self.assertEqual(parsed["nurec"]["dev_deps_image"], defaults["nurec"]["dev_deps_image"])
        self.assertEqual(parsed["nurec"]["source_dir"], defaults["nurec"]["source_dir"])
        self.assertEqual(parsed["execution"], defaults["execution"])

    def test_absent_profile_passes(self) -> None:
        """An absent profile is valid; it is what schedules audit-dataset."""
        doc = _base_input()
        del doc["data"]["profile"]
        parsed = self._parse_doc(doc)
        self.assertNotIn("profile", parsed["data"])

    def test_empty_profile_rejected(self) -> None:
        """An empty profile cannot stand in for an audited dataset.

        Profile *presence* is what marks a dataset already audited, so an
        empty mapping would otherwise skip audit-dataset entirely.
        """
        doc = _base_input()
        doc["data"]["profile"] = {}
        self.assertParseError(doc, "data.profile", "must not be empty")

    def test_profile_missing_sections_rejected(self) -> None:
        """A profile omitting required sections is rejected, listing all gaps."""
        doc = _base_input()
        del doc["data"]["profile"]["cameras"]
        del doc["data"]["profile"]["lidars"]
        self.assertParseError(doc, "data.profile", "missing required sections", "cameras", "lidars")

    def test_profile_with_empty_sections_rejected(self) -> None:
        """A profile of bare section keys carries no facts to audit."""
        doc = _base_input()
        doc["data"]["profile"]["cameras"] = None
        doc["data"]["profile"]["cuboids"] = None
        self.assertParseError(doc, "data.profile", "present but empty", "cameras", "cuboids")

    def test_profile_empty_list_section_accepted(self) -> None:
        """An empty list is a fact: a camera-only dataset has no LiDARs."""
        doc = _base_input()
        doc["data"]["profile"]["lidars"] = []
        self.assertEqual(self._parse_doc(doc)["data"]["profile"]["lidars"], [])

    def test_non_mapping_profile_rejected(self) -> None:
        """A non-mapping profile is rejected."""
        doc = _base_input()
        doc["data"]["profile"] = ["file_layout"]
        self.assertParseError(doc, "data.profile", "must be a mapping")

    def test_explicit_null_block_keeps_defaults(self) -> None:
        """An explicit-null block inherits defaults instead of wiping them."""
        doc = _base_input()
        doc["execution"] = None
        parsed = self._parse_doc(doc)
        self.assertEqual(parsed["execution"], load_defaults()["execution"])

    def test_present_but_null_scalar_keeps_key(self) -> None:
        """A null scalar field keeps its key (audit placeholder semantics)."""
        doc = _base_input()
        doc["nurec"]["output_dir"] = None
        parsed = self._parse_doc(doc)
        self.assertIn("output_dir", parsed["nurec"])
        self.assertIsNone(parsed["nurec"]["output_dir"])

    def test_profile_sensor_ids_use_safe_charset(self) -> None:
        """New profile-scoped sensor IDs use the same charset validation."""
        doc = _base_input()
        doc["ncore"]["camera_ids"] = ["front;rm"]
        self.assertParseError(
            doc,
            "ncore.camera_ids[0]",
            "only [A-Za-z0-9_-]",
        )

    def test_nurec_lidar_must_be_converted(self) -> None:
        """NuRec LiDAR selection must name a converted LiDAR."""
        doc = _base_input()
        doc["nurec"]["lidar_ids"] = ["rear"]
        self.assertParseError(
            doc,
            "nurec.lidar_ids[0]",
            "ncore.lidar_ids",
        )

    def test_nurec_lidar_ids_must_have_one_item_for_now(self) -> None:
        """NuRec LiDAR scope is a one-item list until multi-LiDAR runs land."""
        doc = _base_input()
        doc["nurec"]["lidar_ids"] = ["top", "rear"]
        self.assertParseError(
            doc,
            "nurec.lidar_ids",
            "exactly 1 element",
        )

    def test_conversion_window_requires_seek_and_duration_pair(self) -> None:
        """Conversion time windows require both offset and duration."""
        doc = _base_input()
        doc["ncore"]["duration_sec"] = None
        self.assertParseError(
            doc,
            "ncore time window",
            "seek_sec and duration_sec",
        )

    def test_unresolved_lidar_membership_is_deferred(self) -> None:
        """Unresolved LiDAR scope placeholders do not fail membership checks."""
        doc = _base_input()
        doc["ncore"]["lidar_ids"] = ["UNRESOLVED: Pick LiDAR."]
        doc["nurec"]["lidar_ids"] = ["UNRESOLVED: Pick NuRec LiDAR."]
        parsed = self._parse_doc(doc)
        self.assertEqual(parsed["ncore"]["lidar_ids"], ["UNRESOLVED: Pick LiDAR."])
        self.assertEqual(parsed["nurec"]["lidar_ids"], ["UNRESOLVED: Pick NuRec LiDAR."])

    def test_partially_unresolved_lidar_membership_is_deferred(self) -> None:
        """Unresolved NCore LiDAR scope defers concrete NuRec membership checks."""
        doc = _base_input()
        doc["ncore"]["lidar_ids"] = ["UNRESOLVED: Pick converted LiDAR."]
        doc["nurec"]["lidar_ids"] = ["top"]
        parsed = self._parse_doc(doc)
        self.assertEqual(parsed["nurec"]["lidar_ids"], ["top"])

    def test_sensor_id_lists_reject_null_items(self) -> None:
        """Null entries in sensor ID lists fail at the element path."""
        doc = _base_input()
        doc["ncore"]["camera_ids"] = ["front", None]
        self.assertParseError(doc, "ncore.camera_ids[1]=null", "sensor IDs")

    def test_nurec_lidar_requires_concrete_ncore_scope(self) -> None:
        """Concrete NuRec LiDAR IDs must be present in a concrete NCore scope."""
        doc = _base_input()
        doc["ncore"]["lidar_ids"] = []
        doc["nurec"]["lidar_ids"] = ["top"]
        self.assertParseError(doc, "nurec.lidar_ids[0]", "ncore.lidar_ids")

    def test_parent_blocks_must_be_mappings(self) -> None:
        """Malformed parent blocks fail instead of hiding nested fields."""
        doc = _base_input()
        doc["ncore"] = "not-a-mapping"
        self.assertParseError(doc, "Invalid ncore", "must be a mapping")

    def test_conversion_window_rejects_non_finite_numbers(self) -> None:
        """NaN and infinity are not valid conversion-window numbers."""
        for value in (float("nan"), float("inf"), float("-inf")):
            doc = _base_input()
            doc["ncore"]["seek_sec"] = value
            self.assertParseError(doc, "ncore.seek_sec", "non-negative number")

    def test_audit_block_rejects_unknown_keys(self) -> None:
        """Typoed audit keys fail instead of silently disabling audit."""
        doc = _base_input()
        doc["audit"] = {"enable": True}
        self.assertParseError(doc, "Invalid audit keys", "enable", "enabled")

    def test_audit_block_rejects_wrong_types(self) -> None:
        """Audit keys use their documented scalar types."""
        doc = _base_input()
        doc["audit"]["enabled"] = "true"
        self.assertParseError(doc, "audit.enabled", "must be a bool")

    def test_runtime_paths_and_images_must_be_non_blank_strings(self) -> None:
        """Grouped runtime path and image settings reject blank values."""
        doc = _base_input()
        doc["nurec"]["dev_deps_image"] = ""
        self.assertParseError(doc, "nurec.dev_deps_image", "non-blank string")

    def test_execution_block_rejects_unknown_keys(self) -> None:
        """Typoed execution controls fail instead of being ignored."""
        doc = _base_input()
        doc["execution"] = {"max_outer_iteration": 5}
        self.assertParseError(doc, "Invalid execution keys", "max_outer_iteration")

    def test_execution_iteration_limits_must_be_positive_ints(self) -> None:
        """Execution iteration limits must be positive integers."""
        doc = _base_input()
        doc["execution"] = dict(load_defaults()["execution"])
        doc["execution"]["max_inner_iterations"] = 0
        self.assertParseError(doc, "execution.max_inner_iterations", "positive integer")


if __name__ == "__main__":
    unittest.main()
