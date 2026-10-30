# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test per-run Claude Code settings generation."""

from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from lib.claude_config import update_settings


class ClaudeConfigTest(unittest.TestCase):
    """Verify dynamic Claude Code settings updates."""

    def test_rewrites_canonical_paths_and_webfetch_rules(self) -> None:
        """Replace dynamic settings while preserving static permissions."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            workdir = root / "work"
            cache = root / "cache"
            for path in (data, workdir, cache):
                path.mkdir()
            cache_link = root / "cache-link"
            cache_link.symlink_to(cache, target_is_directory=True)

            template_path = root / "settings.local.template.json"
            settings_path = root / "settings.local.json"
            template_path.write_text(
                json.dumps(
                    {
                        "permissions": {
                            "allow": [
                                "Bash",
                                "WebFetch",
                                "WebFetch(domain:stale.example.com)",
                            ]
                        },
                        "sandbox": {
                            "enabled": True,
                            "filesystem": {"allowWrite": ["/stale"]},
                        },
                    }
                )
            )
            input_path = root / "input.yaml"
            input_path.write_text(
                "data:\n"
                "  name: sample\n"
                f"  path: {data}\n"
                "  description: sample\n"
                "  docs:\n"
                "    - https://Docs.Example.com/reference\n"
                "    - https://docs.example.com/duplicate\n"
                "    - https://other.example.com/reference\n"
                "    - local.md\n"
            )

            update_settings(
                template_path=template_path,
                settings_path=settings_path,
                input_path=input_path,
                write_paths=[cache_link, workdir, cache_link],
            )

            settings = json.loads(settings_path.read_text())
            self.assertEqual(
                settings["sandbox"]["filesystem"]["allowWrite"],
                [str(cache), str(workdir)],
            )
            self.assertTrue(settings["sandbox"]["enabled"])
            self.assertEqual(
                settings["permissions"]["allow"],
                [
                    "Bash",
                    "WebFetch(domain:docs.example.com)",
                    "WebFetch(domain:other.example.com)",
                ],
            )

    def test_removes_webfetch_when_input_has_no_urls(self) -> None:
        """Remove all WebFetch grants when no documentation URL is declared."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            template_path = root / "settings.local.template.json"
            settings_path = root / "settings.local.json"
            template_path.write_text(
                json.dumps(
                    {
                        "permissions": {
                            "allow": [
                                "Read",
                                "WebFetch(domain:stale.example.com)",
                            ]
                        }
                    }
                )
            )
            input_path = root / "input.yaml"
            input_path.write_text(
                f"data:\n  name: sample\n  path: {data}\n  description: sample\n  docs:\n    - local.md\n"
            )

            update_settings(
                template_path=template_path,
                settings_path=settings_path,
                input_path=input_path,
                write_paths=[],
            )

            settings = json.loads(settings_path.read_text())
            self.assertEqual(settings["permissions"]["allow"], ["Read"])
            self.assertEqual(settings["sandbox"]["filesystem"]["allowWrite"], [])

    def test_replaces_previous_settings_and_leaves_template_unchanged(self) -> None:
        """Start each run from the template, discarding the previous run's file."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template_path = root / "settings.local.template.json"
            template_text = json.dumps({"permissions": {"allow": ["Read"]}, "sandbox": {"enabled": True}})
            template_path.write_text(template_text)
            settings_path = root / "settings.local.json"
            settings_path.write_text(
                json.dumps({"permissions": {"allow": ["Read", "Bash(rm:*)"]}, "sandbox": {"enabled": False}})
            )
            input_path = root / "input.yaml"
            input_path.write_text(f"data:\n  name: sample\n  path: {root}\n  description: sample\n")

            update_settings(
                template_path=template_path,
                settings_path=settings_path,
                input_path=input_path,
                write_paths=[],
            )

            settings = json.loads(settings_path.read_text())
            self.assertEqual(settings["permissions"]["allow"], ["Read"])
            self.assertTrue(settings["sandbox"]["enabled"])
            self.assertEqual(template_path.read_text(), template_text)

    def test_rejects_settings_that_are_not_an_object(self) -> None:
        """Reject a template whose top-level value cannot hold keys."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template_path = root / "settings.local.template.json"
            template_path.write_text(json.dumps(["Bash"]))
            input_path = root / "input.yaml"
            input_path.write_text(f"data:\n  name: sample\n  path: {root}\n  description: sample\n")

            with self.assertRaisesRegex(TypeError, "must be a JSON object"):
                update_settings(
                    template_path=template_path,
                    settings_path=root / "settings.local.json",
                    input_path=input_path,
                    write_paths=[],
                )

    def test_rejects_permission_allow_that_is_not_a_list(self) -> None:
        """Reject a permission allow-list the WebFetch rewrite cannot filter."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template_path = root / "settings.local.template.json"
            template_path.write_text(json.dumps({"permissions": {"allow": "Bash"}}))
            input_path = root / "input.yaml"
            input_path.write_text(f"data:\n  name: sample\n  path: {root}\n  description: sample\n")

            with self.assertRaisesRegex(TypeError, "permissions.allow must be a list"):
                update_settings(
                    template_path=template_path,
                    settings_path=root / "settings.local.json",
                    input_path=input_path,
                    write_paths=[],
                )


if __name__ == "__main__":
    unittest.main()
