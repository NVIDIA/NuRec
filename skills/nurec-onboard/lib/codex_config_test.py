# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test per-run Codex permission-profile overrides."""

from __future__ import annotations

import tempfile
import tomllib
import unittest

from pathlib import Path

from lib.codex_config import FIXED_NETWORK_DOMAINS, build_overrides


REPO_ROOT = Path(__file__).parents[1]


class CodexConfigTest(unittest.TestCase):
    """Verify generated Codex permission-profile overrides."""

    def test_builds_canonical_scoped_permissions(self) -> None:
        """Grant exact paths and input-derived domains."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            data = root / "data"
            workdir = root / "work"
            cache = root / "cache"
            runtime = root / "codex-runtime"
            for path in (repo, data, workdir, cache, runtime):
                path.mkdir()
            local_doc = repo / "dataset.md"
            local_doc.write_text("dataset")
            input_path = root / "input.yaml"
            input_path.write_text(
                "data:\n"
                "  name: sample\n"
                f"  path: {data}\n"
                "  description: sample\n"
                "  docs:\n"
                "    - dataset.md\n"
                "    - https://Docs.Example.com/reference\n"
            )

            overrides = build_overrides(
                repo_root=repo,
                input_path=input_path,
                data_path=data,
                workdir=workdir,
                requires_docker=False,
                read_paths=[runtime],
                write_paths=[cache],
            )

            rendered = "\n".join(overrides)
            self.assertIn(f'"{workdir}"="write"', rendered)
            self.assertIn(f'"{cache}"="write"', rendered)
            self.assertIn(f'"{repo}"="read"', rendered)
            self.assertIn(f'"{data}"="read"', rendered)
            self.assertNotIn(f'"{local_doc}"="read"', rendered)
            self.assertIn(f'"{runtime}"="read"', rendered)
            self.assertIn('"docs.example.com"="allow"', rendered)
            self.assertNotIn("docker.sock", rendered)
            for override in overrides:
                tomllib.loads(override)

    def test_keeps_write_grant_for_paths_that_are_also_readable(self) -> None:
        """Keep the stronger grant when a path appears in both path lists."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            data = root / "data"
            workdir = root / "work"
            for path in (repo, data, workdir):
                path.mkdir()
            input_path = root / "input.yaml"
            input_path.write_text(f"data:\n  name: sample\n  path: {data}\n  description: sample\n")

            overrides = build_overrides(
                repo_root=repo,
                input_path=input_path,
                data_path=data,
                workdir=workdir,
                requires_docker=False,
                read_paths=[workdir],
                write_paths=[],
            )

            rendered = "\n".join(overrides)
            self.assertIn(f'"{workdir}"="write"', rendered)
            self.assertNotIn(f'"{workdir}"="read"', rendered)

    def test_grants_doc_files_and_directories_outside_other_roots(self) -> None:
        """Grant every local doc entry, wherever the operator put it."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            repo = root / "repo"
            data = root / "data"
            workdir = root / "work"
            docs_dir = root / "elsewhere" / "docs"
            for path in (repo, data, workdir, docs_dir):
                path.mkdir(parents=True)
            doc_file = root / "elsewhere" / "calibration.md"
            doc_file.write_text("calibration")
            input_path = root / "input.yaml"
            input_path.write_text(
                "data:\n"
                "  name: sample\n"
                f"  path: {data}\n"
                "  description: sample\n"
                "  docs:\n"
                f"    - {doc_file}\n"
                f"    - {docs_dir}\n"
            )

            overrides = build_overrides(
                repo_root=repo,
                input_path=input_path,
                data_path=data,
                workdir=workdir,
                requires_docker=False,
                read_paths=[],
                write_paths=[],
            )

            rendered = "\n".join(overrides)
            self.assertIn(f'"{doc_file}"="read"', rendered)
            self.assertIn(f'"{docs_dir}"="read"', rendered)
            self.assertNotIn(f'"{doc_file.parent}"="read"', rendered)

    def test_resolves_doc_paths_relative_to_the_checkout(self) -> None:
        """Grant a relative doc entry at the path it resolves to."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            repo = root / "repo"
            data = root / "data"
            workdir = root / "work"
            for path in (repo, data, workdir):
                path.mkdir()
            outside_doc = root / "outside.md"
            outside_doc.write_text("outside")
            input_path = root / "input.yaml"
            input_path.write_text(
                f"data:\n  name: sample\n  path: {data}\n  description: sample\n  docs:\n    - ../outside.md\n"
            )

            overrides = build_overrides(
                repo_root=repo,
                input_path=input_path,
                data_path=data,
                workdir=workdir,
                requires_docker=False,
                read_paths=[],
                write_paths=[],
            )

            self.assertIn(f'"{outside_doc}"="read"', "\n".join(overrides))

    def test_refuses_read_grant_for_sensitive_doc_root(self) -> None:
        """Keep a sensitive root out of the grant table."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            repo = root / "repo"
            data = root / "data"
            workdir = root / "work"
            for path in (repo, data, workdir):
                path.mkdir()
            input_path = root / "input.yaml"
            input_path.write_text(
                f"data:\n  name: sample\n  path: {data}\n  description: sample\n  docs:\n    - /etc\n"
            )

            overrides = build_overrides(
                repo_root=repo,
                input_path=input_path,
                data_path=data,
                workdir=workdir,
                requires_docker=False,
                read_paths=[],
                write_paths=[],
            )

            self.assertNotIn('"/etc"="read"', "\n".join(overrides))

    def test_emits_no_deny_entry(self) -> None:
        """Keep the profile deny-free so rule-allowed commands can leave the sandbox."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            data = root / "data"
            workdir = root / "work"
            for path in (repo, data, workdir):
                path.mkdir()
            input_path = root / "input.yaml"
            input_path.write_text(f"data:\n  name: sample\n  path: {data}\n  description: sample\n")

            overrides = build_overrides(
                repo_root=repo,
                input_path=input_path,
                data_path=data,
                workdir=workdir,
                requires_docker=True,
                read_paths=[],
                write_paths=[],
            )

            # A single deny entry makes Codex discard the `docker` rule's
            # escalation, which container-backed steps depend on.
            for override in overrides:
                key, _, value = override.partition("=")
                if key.endswith(".filesystem"):
                    self.assertNotIn("deny", value)

    def test_adds_docker_socket_only_when_required(self) -> None:
        """Add the Docker socket allowlist only for Docker-backed manifests."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            data = root / "data"
            workdir = root / "work"
            for path in (repo, data, workdir):
                path.mkdir()
            input_path = root / "input.yaml"
            input_path.write_text(f"data:\n  name: sample\n  path: {data}\n  description: sample\n")

            overrides = build_overrides(
                repo_root=repo,
                input_path=input_path,
                data_path=data,
                workdir=workdir,
                requires_docker=True,
                read_paths=[],
                write_paths=[],
            )

            self.assertTrue(any('"/var/run/docker.sock"="allow"' in value for value in overrides))
            for override in overrides:
                tomllib.loads(override)

    def test_static_config_uses_custom_fail_closed_profile(self) -> None:
        """Keep legacy sandbox settings out of project configuration."""
        config = tomllib.loads((REPO_ROOT / ".codex" / "config.toml").read_text())

        self.assertEqual(config["approval_policy"], "never")
        self.assertEqual(config["default_permissions"], "nurec-onboard")
        self.assertNotIn("sandbox_mode", config)
        self.assertNotIn("sandbox_workspace_write", config)
        self.assertEqual(
            set(config["permissions"]["nurec-onboard"]["network"]["domains"]),
            FIXED_NETWORK_DOMAINS,
        )
        self.assertNotIn(
            "deny",
            config["permissions"]["nurec-onboard"]["filesystem"].values(),
        )
        # Domain rules are enforced by Codex's network proxy, and that proxy is
        # also what resolves hostnames for sandboxed commands.
        self.assertTrue(config["features"]["network_proxy"])


if __name__ == "__main__":
    unittest.main()
