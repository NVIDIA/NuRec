# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide isolated launcher integration-test infrastructure."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from pathlib import Path


# Left unresolved on purpose: under Bazel this file is a runfiles symlink, so
# resolving it copies the working tree instead of the declared inputs -- tests
# read uncommitted edits, and cached passes survive undeclared edits.
REPO_ROOT = Path(__file__).parents[1]


class LauncherTestBase(unittest.TestCase):
    """Create an isolated checkout and fake launcher dependencies."""

    def setUp(self) -> None:
        """Create isolated agent homes, inputs, and fake executables."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.nre_root = self.root / "nre"
        self.nurec_onboard_dir = self.nre_root / "skills" / "nurec-onboard"
        self.home = self.root / "home"
        self.fake_bin = self.root / "bin"
        self.data = self.root / "data"
        self.workdir = self.root / "work"
        for path in (
            self.nurec_onboard_dir,
            self.home,
            self.fake_bin,
            self.data,
            self.workdir,
        ):
            path.mkdir(parents=True)
        (self.nre_root / "MODULE.bazel").write_text('module(name = "nre")\n')
        (self.nre_root / "run.py").touch()
        (self.nre_root / ".bazelversion").write_text("8.3.1")
        (self.nre_root / "configs").mkdir()
        (self.nre_root / "docs/basics").mkdir(parents=True)
        (self.nre_root / "docs/basics/software.rst").write_text("ghcr.io/nvidia/nurec/nurec_dev_deps:test-release")

        for name in (
            "run.sh",
            "defaults.yaml",
            "lib",
            "nurec-onboard",
            ".claude",
            ".codex",
        ):
            source = REPO_ROOT / name
            destination = self.nurec_onboard_dir / name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        (self.nurec_onboard_dir / "ncore_template").mkdir(parents=True)
        self.input_path = self.root / "input.yaml"
        self.input_path.write_text(
            "data:\n"
            "  name: sample\n"
            f"  path: {self.data}\n"
            "  description: sample\n"
            "  docs:\n"
            "    - https://docs.example.com/reference\n"
        )
        self.fake_bin.joinpath("python3").symlink_to(sys.executable)
        self._write_executable("bazel", "#!/usr/bin/env bash\nexit 0\n")
        self._write_executable("bwrap", "#!/usr/bin/env bash\nexit 0\n")
        self._write_executable("socat", "#!/usr/bin/env bash\nexit 0\n")
        self._write_executable("docker", "#!/usr/bin/env bash\nexit 0\n")
        self._replace_in_launcher('if [[ ! -S "$docker_sock" ]]; then', "if false; then")
        self.extra_environment: dict[str, str] = {}

    def tearDown(self) -> None:
        """Remove isolated launcher state."""
        self.temp_dir.cleanup()

    def _run(
        self,
        *arguments: str,
        extra_env: dict[str, str] | None = None,
        step: str = "audit-dataset",
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run the copied launcher with isolated dependencies.

        Args:
            *arguments: Launcher arguments after the input YAML and ``--step``.
            extra_env: Environment variables layered over the isolated ones.
            step: Step passed to ``--step``.
            cwd: Directory to invoke the launcher from; defaults to the copied
                checkout.

        Returns:
            The completed launcher process, with text stdout and stderr.
        """
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{self.fake_bin}:{env['PATH']}",
                "HOME": str(self.home),
                "NUREC_WORKDIR": str(self.workdir),
                "PYTHONPATH": os.pathsep.join(sys.path),
                **self.extra_environment,
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [
                "bash",
                str(self.nurec_onboard_dir / "run.sh"),
                str(self.input_path),
                "--step",
                step,
                *arguments,
            ],
            cwd=cwd or self.nurec_onboard_dir,
            env=env,
            text=True,
            # The launcher's progress output carries non-ASCII text (status
            # markers, and agent prose with typographic quotes), so decode it
            # as what it is rather than as whatever locale the runner has.
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=300,
        )

    def _nest_checkout_in_git_repository(self) -> tuple[Path, Path]:
        """Make the checkout a subdirectory of a git repository.

        Returns:
            The resolved repository root and checkout paths, the forms the
            agents record in their trust configuration.
        """
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True)
        return self.root.resolve(), self.nurec_onboard_dir.resolve()

    def _write_executable(self, name: str, contents: str) -> None:
        """Create an executable in the isolated command path."""
        path = self.fake_bin / name
        path.write_text(contents)
        path.chmod(0o755)

    def _replace_in_launcher(self, old: str, new: str) -> None:
        """Rewrite the copied launcher, failing when the target text is absent.

        A silent no-op here would leave the test asserting against unmodified
        launcher behavior, so a launcher edit that retires ``old`` surfaces as a
        failure in the test that depends on it rather than as a puzzling
        assertion mismatch later.

        Args:
            old: Text expected to appear in the copied ``run.sh``.
            new: Replacement text.
        """
        launcher = self.nurec_onboard_dir / "run.sh"
        original = launcher.read_text()
        if old not in original:
            # fail() rather than assertIn() so the message stays one line
            # instead of echoing the whole launcher as the haystack.
            self.fail(f"run.sh no longer contains {old!r} to patch")
        launcher.write_text(original.replace(old, new, 1))
