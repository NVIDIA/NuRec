# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``validators/audit_dataset_validator.py``."""

from __future__ import annotations

import json
import tempfile
import unittest

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from lib.parse_input import load_defaults

from validators._validation_base import ValidationError
from validators.audit_dataset_validator import (
    NUREC_ONBOARD_VERSION,
    _extract_unresolved_keys,
    main,
    validate_audit_artifacts,
)
from validators.validator_test_helpers import ValidatorTestCase


def _base_input() -> dict[str, Any]:
    """Return a minimally valid generated workflow input."""
    return {
        "data": {
            "name": "demo_dataset",
            "path": "/tmp/demo_dataset",
            "description": "Demo dataset with one camera and one LiDAR.",
            "profile": {
                "file_layout": {
                    "summary": "metadata.json describes raw inputs.",
                    "root": "/tmp/demo_dataset",
                    "sequence_id_source": "directory name",
                    "raw_inputs": [],
                    "calibration_files": ["metadata.json"],
                    "metadata_files": ["metadata.json"],
                },
                "calibration_and_frames": {
                    "summary": "Calibration is in metadata.json.",
                    "calibration_sources": ["metadata.json"],
                    "rig_frame": "x-forward, y-left, z-up",
                    "world_frame": "local",
                    "camera_frame": "opencv",
                    "lidar_frame": "sensor",
                    "frames": ["rig", "world", "front", "top"],
                    "static_transforms": [],
                    "dynamic_transforms": [],
                    "transform_direction_conventions": [],
                    "known_mismatches": [],
                },
                "pose_sources": [
                    {
                        "id": "ego",
                        "raw_stream": "poses.json",
                        "source_frame": "rig",
                        "target_frame": "world",
                        "direction": "rig_to_world",
                        "timestamp_unit": "microseconds",
                        "coverage": "full",
                        "global_frame": "local",
                    }
                ],
                "timestamp_semantics": {
                    "summary": "Timestamps are in microseconds.",
                    "clock": "dataset",
                    "unit": "microseconds",
                    "camera_timestamp_meaning": "exposure_end",
                    "lidar_frame_timestamp_meaning": "sweep_start",
                    "per_ray_timestamps": "available",
                    "synchronization": "shared clock",
                },
                "cameras": [
                    {
                        "id": "front",
                        "raw_stream": "camera/front",
                        "model": "demo",
                        "resolution": [1920, 1080],
                        "image_format": "jpeg",
                        "model_family": "opencv_pinhole",
                        "intrinsics_source": "metadata.json",
                        "distortion_model": "none",
                        "distortion_coeff_order": [],
                        "image_state": "raw_distorted",
                        "shutter": {
                            "type": "global",
                            "direction": None,
                            "readout_duration_sec": "UNRESOLVED: Missing camera shutter timing.",
                            "timestamp_meaning": "exposure_end",
                        },
                        "masks": {"available": False, "entries": []},
                    }
                ],
                "lidars": [
                    {
                        "id": "top",
                        "raw_stream": "lidar/top",
                        "model": "demo",
                        "type": "mechanical_spinning",
                        "raw_format": "xyz",
                        "model_source": "metadata.json",
                        "beam_structure": {
                            "rows": 64,
                            "columns": 1024,
                            "row_elevations": "metadata.json",
                            "column_azimuths": "metadata.json",
                            "row_azimuth_offsets": "metadata.json",
                        },
                        "spinning": {"direction": "ccw", "frequency_hz": 10.0},
                        "timing": {
                            "frame_timestamp_meaning": "sweep_start",
                            "per_ray_timestamps": "available",
                        },
                        "per_ray_metadata": [],
                        "intensity": "float",
                        "motion_compensation": "uncompensated",
                    }
                ],
                "cuboids": {
                    "summary": "No cuboids.",
                    "exists": False,
                    "raw_stream": None,
                    "centroid": None,
                    "dimensions": None,
                    "yaw": None,
                    "reference_frame": None,
                    "timestamp_meaning": None,
                    "track_id_source": None,
                    "class_vocabulary": [],
                    "label_source": None,
                    "coverage": None,
                },
                "known_gaps": [],
            },
            "docs": ["/tmp/demo_dataset/README.md"],
        },
        "ncore": {
            "camera_ids": None,
            "lidar_ids": ["top"],
            "seek_sec": None,
            "duration_sec": None,
            "converter_dir": None,
            "output_dir": None,
        },
        "nurec": {
            "camera_ids": None,
            "lidar_ids": ["top"],
            "output_dir": None,
        },
    }


def _base_report() -> str:
    """Return a report containing all required audit sections."""
    return """# Data Quality Report -- demo_dataset

## Summary
Audited one sequence.

## Documentation Coverage
Dataset docs loaded from README.md.

## Minimum NCore/NuRec Facts
Most minimum facts were confirmed.

## Sensor Facts
One camera and one LiDAR were found.

## Confirmed Parameters
Sensor IDs were confirmed.

## Critical Gaps
- data.profile.cameras[0].shutter.readout_duration_sec: Camera shutter timing is missing.

## Warnings
_(none observed)_

## Operator Next Steps
Resolve critical gaps, then run nurec-onboard.
"""


def _base_profile() -> dict[str, Any]:
    """Return a valid audit profile matching ``_base_input``."""
    return {
        "nurec_onboard_version": NUREC_ONBOARD_VERSION,
        "dataset": {"name": "demo_dataset", "path": "/tmp/demo_dataset"},
        "minimum_facts": {
            "confirmed": ["data.profile.cameras[0].id", "data.profile.lidars[0].id"],
            "unresolved": ["data.profile.cameras[0].shutter.readout_duration_sec"],
        },
        "dataset_probe_hints": {
            "calibrations": [],
            "timestamps": [],
            "cuboids": [],
            "lidar_points": [],
        },
    }


class AuditValidatorTest(ValidatorTestCase):
    """Exercise audit artifact validation invariants."""

    def _write_artifacts(
        self,
        tmpdir: Path,
        *,
        input_doc: dict[str, Any] | None = None,
        report_text: str | None = None,
        profile_doc: dict[str, Any] | None = None,
    ) -> tuple[Path, Path, Path]:
        """Write audit artifacts for a test case.

        Args:
            tmpdir: Temporary directory for artifact files.
            input_doc: Generated input YAML document override.
            report_text: Data-quality report override.
            profile_doc: Audit profile override.

        Returns:
            Paths to ``input.yaml``, ``report.md``, and ``audit_profile.json``.
        """
        input_path = tmpdir / "input.yaml"
        report_path = tmpdir / "report.md"
        profile_path = tmpdir / "audit_profile.json"
        input_path.write_text(yaml.safe_dump(input_doc if input_doc is not None else _base_input(), sort_keys=False))
        report_path.write_text(report_text if report_text is not None else _base_report())
        profile_path.write_text(json.dumps(profile_doc if profile_doc is not None else _base_profile(), indent=2))
        return input_path, report_path, profile_path

    def _diagnostics_for(
        self,
        *,
        input_doc: dict[str, Any] | None = None,
        report_text: str | None = None,
        profile_doc: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return validation diagnostics for overridden artifacts."""
        with tempfile.TemporaryDirectory() as raw_tmp:
            paths = self._write_artifacts(
                Path(raw_tmp),
                input_doc=input_doc,
                report_text=report_text,
                profile_doc=profile_doc,
            )
            try:
                validate_audit_artifacts(*paths)
            except ValidationError as exc:
                return exc.diagnostics
        return []

    def test_valid_artifacts_pass(self) -> None:
        """Base audit artifacts pass validation."""
        self.assertEqual(self._diagnostics_for(), [])

    def test_cli_prints_nurec_onboard_version(self) -> None:
        """Print the release stamp used by generated audit profiles."""
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(["audit_dataset_validator", "--print-nurec-onboard-version"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), f"{NUREC_ONBOARD_VERSION}\n")

    def test_public_examples_match_audit_schema(self) -> None:
        """Waymo and PandaSet example inputs validate as audit outputs."""
        repo_root = Path(__file__).parents[1]
        for example_path in (
            repo_root / "examples/input-waymo.yaml",
            repo_root / "examples/input-pandaset.yaml",
        ):
            with self.subTest(example=example_path.name):
                doc = yaml.safe_load(example_path.read_text())
                profile: dict[str, Any] = {
                    "nurec_onboard_version": NUREC_ONBOARD_VERSION,
                    "dataset": {
                        "name": doc["data"]["name"],
                        "path": doc["data"]["path"],
                    },
                    "minimum_facts": {
                        "confirmed": [],
                        "unresolved": _extract_unresolved_keys(doc["data"]["profile"]),
                    },
                    "dataset_probe_hints": {
                        "calibrations": [],
                        "timestamps": [],
                        "cuboids": [],
                        "lidar_points": [],
                    },
                }
                unresolved = profile["minimum_facts"]["unresolved"]
                report = _base_report().replace(
                    "- data.profile.cameras[0].shutter.readout_duration_sec: Camera shutter timing is missing.",
                    "\n".join(
                        ["- data.profile.cameras[0].shutter.readout_duration_sec: Camera shutter timing is missing."]
                        + [f"- {key}: unresolved example field." for key in unresolved]
                    ),
                )
                self.assertEqual(self._diagnostics_for(input_doc=doc, report_text=report, profile_doc=profile), [])

    def test_nurec_images_are_optional_runtime_overrides(self) -> None:
        """Audit output does not require deployment image overrides."""
        doc = _base_input()

        self.assertEqual(self._diagnostics_for(input_doc=doc), [])

    def test_input_skeleton_matches_audit_schema(self) -> None:
        """Audit input skeleton validates when unresolved gaps are reported."""
        repo_root = Path(__file__).parents[1]
        skeleton_path = repo_root / "nurec-onboard/steps/audit-dataset/input-skeleton.yaml"
        doc = yaml.safe_load(skeleton_path.read_text())
        unresolved = (
            _extract_unresolved_keys(doc["data"]["profile"])
            + _extract_unresolved_keys(doc["ncore"], root_path="ncore")
            + _extract_unresolved_keys(doc["nurec"], root_path="nurec")
        )
        report = _base_report().replace(
            "- data.profile.cameras[0].shutter.readout_duration_sec: Camera shutter timing is missing.",
            "\n".join(f"- {key}: unresolved skeleton field." for key in unresolved),
        )
        profile = {
            "nurec_onboard_version": NUREC_ONBOARD_VERSION,
            "dataset": {
                "name": doc["data"]["name"],
                "path": doc["data"]["path"],
            },
            "minimum_facts": {"confirmed": [], "unresolved": unresolved},
            "dataset_probe_hints": {
                "calibrations": [],
                "timestamps": [],
                "cuboids": [],
                "lidar_points": [],
            },
        }
        self.assertEqual(self._diagnostics_for(input_doc=doc, report_text=report, profile_doc=profile), [])

    def test_unresolved_key_requires_critical_gap_entry(self) -> None:
        """Every YAML ``UNRESOLVED`` key must be named in Critical Gaps."""
        report = _base_report().replace(
            "data.profile.cameras[0].shutter.readout_duration_sec",
            "data.other_gap",
        )
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(report_text=report),
            "UNRESOLVED",
            "data.profile.cameras[0].shutter.readout_duration_sec",
            "Critical Gaps",
        )

    def test_critical_gap_entries_match_unresolved_keys_exactly(self) -> None:
        """Longer sibling gap paths do not satisfy shorter unresolved keys."""
        report = _base_report().replace(
            "data.profile.cameras[0].shutter.readout_duration_sec",
            "data.profile.cameras[0].shutter.readout_duration_sec_extra",
        )
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(report_text=report),
            "UNRESOLVED",
            "data.profile.cameras[0].shutter.readout_duration_sec",
            "Critical Gaps",
        )

    def test_unresolved_key_requires_profile_entry(self) -> None:
        """The audit profile unresolved list mirrors YAML unresolved keys."""
        profile = _base_profile()
        profile["minimum_facts"]["unresolved"] = []
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(profile_doc=profile),
            "minimum_facts.unresolved",
            "data.profile.cameras[0].shutter.readout_duration_sec",
        )

    def test_unresolved_execution_key_requires_report_and_profile_entry(self) -> None:
        """Execution placeholders are tracked like other generated input gaps."""
        doc = _base_input()
        doc["execution"] = dict(load_defaults()["execution"])
        doc["execution"]["max_inner_iterations"] = "UNRESOLVED: Choose retry budget."
        profile = _base_profile()
        profile["minimum_facts"]["unresolved"].append("execution.max_inner_iterations")
        report = _base_report().replace(
            "- data.profile.cameras[0].shutter.readout_duration_sec: Camera shutter timing is missing.",
            "- data.profile.cameras[0].shutter.readout_duration_sec: Camera shutter timing is missing.\n"
            "- execution.max_inner_iterations: Retry budget needs review.",
        )

        self.assertEqual(self._diagnostics_for(input_doc=doc, report_text=report, profile_doc=profile), [])

        profile["minimum_facts"]["unresolved"] = [
            key for key in profile["minimum_facts"]["unresolved"] if key != "execution.max_inner_iterations"
        ]
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc, report_text=report, profile_doc=profile),
            "execution.max_inner_iterations",
            "audit_profile.minimum_facts.unresolved",
        )

        profile["minimum_facts"]["unresolved"].append("execution.max_inner_iterations")
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc, report_text=_base_report(), profile_doc=profile),
            "execution.max_inner_iterations",
            "Critical Gaps",
        )

    def test_profile_sections_require_standard_fields(self) -> None:
        """Each structured profile section uses the standardized subsection shape."""
        doc = _base_input()
        del doc["data"]["profile"]["cuboids"]["coverage"]
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "data.profile.cuboids.coverage",
            "is missing",
        )

    def test_list_sections_require_standard_entry_fields(self) -> None:
        """Each listed profile entry uses its modality-specific shape."""
        doc = _base_input()
        del doc["data"]["profile"]["cameras"][0]["raw_stream"]
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "data.profile.cameras[0].raw_stream",
            "is missing",
        )

    def test_nested_profile_fields_are_required(self) -> None:
        """Nested converter-decision fields remain explicit even when unknown."""
        doc = _base_input()
        del doc["data"]["profile"]["lidars"][0]["beam_structure"]["row_elevations"]
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "data.profile.lidars[0].beam_structure.row_elevations",
            "is missing",
        )

    def test_nested_profile_fields_reject_wrong_types(self) -> None:
        """Descriptive nested fields reject meaningless scalar types."""
        doc = _base_input()
        doc["data"]["profile"]["lidars"][0]["beam_structure"]["row_elevations"] = 42
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "data.profile.lidars[0].beam_structure.row_elevations",
            "non-empty string",
        )

    def test_high_risk_fact_must_be_concrete_or_unresolved(self) -> None:
        """High-risk geometric facts cannot be blank."""
        doc = _base_input()
        doc["data"]["profile"]["cameras"][0]["model_family"] = ""
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "data.profile.cameras[0].model_family",
            "UNRESOLVED",
        )

    def test_high_risk_unresolved_fact_passes_with_gap_coverage(self) -> None:
        """High-risk facts may be explicit unresolved gaps."""
        doc = _base_input()
        gap = "data.profile.cameras[0].model_family"
        doc["data"]["profile"]["cameras"][0]["model_family"] = "UNRESOLVED: Camera model family needs docs."
        profile = _base_profile()
        profile["minimum_facts"]["unresolved"].append(gap)
        report = _base_report().replace(
            "- data.profile.cameras[0].shutter.readout_duration_sec: Camera shutter timing is missing.",
            "- data.profile.cameras[0].shutter.readout_duration_sec: Camera shutter timing is missing.\n"
            "- data.profile.cameras[0].model_family: Camera model family needs docs.",
        )
        self.assertEqual(self._diagnostics_for(input_doc=doc, report_text=report, profile_doc=profile), [])

    def test_cuboid_high_risk_fields_are_skipped_when_cuboids_absent(self) -> None:
        """Cuboid frame facts are not required when cuboids do not exist."""
        doc = _base_input()
        doc["data"]["profile"]["cuboids"]["reference_frame"] = ""
        doc["data"]["profile"]["cuboids"]["centroid"] = ""
        self.assertEqual(self._diagnostics_for(input_doc=doc), [])

    def test_nurec_lidar_must_be_in_converted_lidars(self) -> None:
        """NuRec LiDAR selection must be present in the converted shard scope."""
        doc = _base_input()
        doc["nurec"]["lidar_ids"] = ["rear"]
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "generated input YAML failed parse_input validation",
            "ncore.lidar_ids",
        )

    def test_nurec_lidar_ids_are_single_item_list_for_now(self) -> None:
        """NuRec LiDAR scope is a one-item list until multi-LiDAR runs land."""
        doc = _base_input()
        doc["nurec"]["lidar_ids"] = ["top", "rear"]
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "generated input YAML failed parse_input validation",
            "nurec.lidar_ids",
            "exactly 1 element",
        )

    def test_critical_gaps_header_must_be_exact(self) -> None:
        """Critical-gaps extraction does not match longer heading names."""
        report = _base_report().replace("## Critical Gaps", "## Critical Gaps Found")
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(report_text=report),
            "missing required section",
            "## Critical Gaps",
        )

    def test_unresolved_explicit_key_override_is_honored(self) -> None:
        """Explicit unresolved keys override the inferred YAML path."""
        doc = _base_input()
        doc["data"]["profile"]["cameras"][0]["shutter"]["readout_duration_sec"] = (
            "UNRESOLVED: data.profile.camera_shutter_readout Missing."
        )
        report = _base_report().replace(
            "data.profile.cameras[0].shutter.readout_duration_sec",
            "data.profile.camera_shutter_readout",
        )
        profile = _base_profile()
        profile["minimum_facts"]["unresolved"] = ["data.profile.camera_shutter_readout"]
        self.assertEqual(self._diagnostics_for(input_doc=doc, report_text=report, profile_doc=profile), [])

    def test_profile_copy_must_match_generated_profile(self) -> None:
        """Optional profile copy must exactly mirror generated ``data.profile``."""
        profile = _base_profile()
        profile["profile"] = {}
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(profile_doc=profile),
            "audit_profile.profile",
            "must match",
        )

    def test_cuboid_dimensions_use_canonical_mapping(self) -> None:
        """Cuboid dimensions use a length/width/height source-field mapping."""
        doc = _base_input()
        doc["data"]["profile"]["cuboids"]["exists"] = True
        doc["data"]["profile"]["cuboids"]["dimensions"] = ["length", "width", "height"]
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "data.profile.cuboids.dimensions",
            "mapping",
        )

    def test_placeholder_on_non_unresolved_line_is_rejected(self) -> None:
        """Placeholders are allowed only on explicit unresolved lines."""
        doc = _base_input()
        doc["data"]["description"] = "Sensor inventory: TODO inspect cameras."
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(input_doc=doc),
            "data.description",
            "TODO marker",
        )

    def test_profile_dataset_must_match_generated_input(self) -> None:
        """Profile dataset identity must match the generated input YAML."""
        profile = _base_profile()
        profile["dataset"]["name"] = "other_dataset"
        self.assertOneDiagnosticMatching(
            self._diagnostics_for(profile_doc=profile),
            "audit_profile.dataset.name",
            "does not match",
        )


if __name__ == "__main__":
    unittest.main()
