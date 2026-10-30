# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Claude Code launcher integration."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest

from typing import Any

from lib.launcher_test_base import LauncherTestBase


class ClaudeLauncherIntegrationTest(LauncherTestBase):
    """Verify the Claude Code launcher path."""

    def setUp(self) -> None:
        """Configure an isolated trusted Claude Code installation."""
        super().setUp()
        self.claude_log = self.root / "claude.args"
        self.template_path = self.nurec_onboard_dir / ".claude" / "settings.local.template.json"
        self.settings_path = self.nurec_onboard_dir / ".claude" / "settings.local.json"
        self._write_executable(
            "claude",
            '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$FAKE_CLAUDE_LOG"\n',
        )
        self.claude_config = self.home / ".claude.json"
        self.claude_config.write_text(
            json.dumps({"projects": {str(self.nurec_onboard_dir): {"hasTrustDialogAccepted": True}}}) + "\n"
        )
        self.extra_environment["FAKE_CLAUDE_LOG"] = str(self.claude_log)

    def _read_settings(self) -> dict[str, Any]:
        """Read the settings the launcher generated."""
        return json.loads(self.settings_path.read_text())

    def _read_template(self) -> dict[str, Any]:
        """Read the copied settings template."""
        return json.loads(self.template_path.read_text())

    def _write_template(self, settings: dict[str, Any]) -> None:
        """Rewrite the copied settings template."""
        self.template_path.write_text(json.dumps(settings, indent=2) + "\n")

    def _patch_sandbox(self, **values: Any) -> None:
        """Replace sandbox keys in the copied settings template."""
        settings = self._read_template()
        settings["sandbox"].update(values)
        self._write_template(settings)

    def test_launch_uses_opus_by_default(self) -> None:
        """Launch with the opus model when the operator names no model."""
        result = self._run("--agent", "claude")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rendered = self.claude_log.read_text()
        args = rendered.splitlines()
        self.assertEqual(args[:2], ["--model", "opus"])
        self.assertIn("Read the skill file", rendered)

    def test_launch_starts_claude_in_the_checkout(self) -> None:
        """Start Claude in the checkout so it loads the project's settings."""
        claude_cwd = self.root / "claude.cwd"
        self._write_executable(
            "claude",
            f'#!/usr/bin/env bash\npwd > "{claude_cwd}"\n',
        )
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()

        result = self._run("--agent", "claude", cwd=elsewhere)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(claude_cwd.read_text().strip(), str(self.nurec_onboard_dir))

    def test_missing_agent_is_refused(self) -> None:
        """Name an agent explicitly: omitting one no longer implies Claude."""
        result = self._run()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--agent is required", result.stdout)
        self.assertFalse(self.claude_log.exists())

    def test_model_override_is_forwarded(self) -> None:
        """Forward an explicit model in place of the default."""
        result = self._run("--agent", "claude", "--model", "test-model")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.claude_log.read_text().splitlines()
        self.assertEqual(args[:2], ["--model", "test-model"])

    def test_extra_arguments_follow_the_prompt(self) -> None:
        """Forward operator arguments where a variadic one cannot eat the prompt."""
        result = self._run("--agent", "claude", "--", "--add-dir", "/scratch/shared")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.claude_log.read_text().splitlines()
        self.assertEqual(args[:2], ["--model", "opus"])
        self.assertEqual(args[-2:], ["--add-dir", "/scratch/shared"])
        self.assertIn("Read the skill file", args[2])
        self.assertIn("Args:     --add-dir /scratch/shared", result.stdout)

    def test_launch_grants_the_default_sandbox_write_paths(self) -> None:
        """Grant scratch, cache, and workdir writes for the run."""
        result = self._run("--agent", "claude")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        allow_write = self._read_settings()["sandbox"]["filesystem"]["allowWrite"]
        self.assertIn("/tmp", allow_write)
        self.assertIn(str(self.home / ".cache" / "bazel"), allow_write)
        self.assertIn(str(self.home / ".cache" / "uv"), allow_write)
        self.assertIn(str(self.workdir), allow_write)

    def test_bazel_wrapper_rejects_host_root_calls_and_preserves_converter_calls(self) -> None:
        """Root dependencies stay in the release container; converters stay on the host."""
        bazel_log = self.root / "bazel.args"
        self._write_executable("bazel", f'#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" >> "{bazel_log}"\n')
        result = self._run("--agent", "claude")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("//skills/nurec-onboard/...", bazel_log.read_text())
        prefix = (self.workdir / "nurec-runtime.md").read_text()
        self.assertIn("docker run -d", prefix)
        self.assertNotIn("--rm", prefix)
        self.assertIn("State.ExitCode=0", prefix)
        self.assertNotIn(".netrc", prefix)
        self.assertIn("nurec-bazel/output_base", prefix)
        wrapper = self.workdir / ".nurec-bin/bazel"
        result = subprocess.run([str(wrapper), "info"], cwd=self.nurec_onboard_dir, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("literal Docker prefix", result.stderr)
        converter = self.workdir / "ncore_sample"
        converter.mkdir()
        (converter / "MODULE.bazel").touch()
        bazel_log.unlink()
        subprocess.run([str(wrapper), "info"], cwd=converter, check=True)
        self.assertEqual(bazel_log.read_text().splitlines(), [str(converter), "info"])

        self.claude_log.unlink()
        self._write_executable("docker", '#!/bin/sh\ncase "$*" in *" fetch "*) exit 37 ;; esac\nexit 0\n')
        failed = self._run("--agent", "claude")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("source dependency resolution failed", failed.stdout)
        self.assertFalse(self.claude_log.exists())

    def test_preflight_rejects_conflicting_source_data_workdir_mounts(self) -> None:
        """Catch duplicate mount destinations before launching any workload."""
        self.input_path.write_text(f"data:\n  name: sample\n  path: {self.workdir}\n")
        result = self._run("--agent", "claude", "--preflight-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NUREC_WORKDIR must differ from data.path", result.stdout)
        self.assertFalse((self.workdir / "nurec-bazel").exists())
        self.input_path.write_text(f"data:\n  name: sample\n  path: {self.data}\n")
        result = self._run("--agent", "claude", "--preflight-only", extra_env={"NUREC_WORKDIR": str(self.nre_root)})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NUREC_WORKDIR must differ from the source checkout", result.stdout)

    def test_release_digest_is_preserved_and_invalid_pins_are_rejected(self) -> None:
        """Never silently weaken a documented digest pin to a mutable tag."""
        repository = "ghcr.io/nvidia/nurec/nurec_dev_deps"
        digest = "sha256:" + "a" * 64
        guide = self.nre_root / "docs/basics/software.rst"
        for pin in (f"{repository}:test-release@{digest}", f"{repository}@{digest}"):
            with self.subTest(pin=pin):
                guide.write_text(pin)
                result = self._run("--agent", "claude", "--preflight-only")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"NuRec source/image pair: {self.nre_root} / {pin}", result.stdout)
                self.assertIn(f"NuRec dev dependencies image is pullable: {pin}", result.stdout)
        for pin in (f"{repository}:latest@{digest}", f"{repository}:test-release@sha256:bad"):
            with self.subTest(pin=pin):
                guide.write_text(pin)
                result = self._run("--agent", "claude", "--preflight-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Pulling NuRec dev dependencies image", result.stdout)

    def test_public_checkout_reads_release_version(self) -> None:
        """Select the checkout's release image without docs or GitLab CI."""
        shutil.rmtree(self.nre_root / "docs")
        version_file = self.nre_root / "bazel/version/VERSION_FILE"
        version_file.parent.mkdir(parents=True)
        for version in ("91.03", "92.11"):
            with self.subTest(version=version):
                major, minor = version.split(".")
                version_file.write_text(f"# Release version\nVERSION_MAJOR={int(major)}\nVERSION_MINOR={int(minor)}\n")
                result = self._run("--agent", "claude", "--preflight-only")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(
                    f"NuRec source/image pair: {self.nre_root} / ghcr.io/nvidia/nurec/nurec_dev_deps:{version}",
                    result.stdout,
                )

    def test_public_checkout_requires_release_version_or_override(self) -> None:
        """Fail clearly when a public checkout has no usable release version."""
        shutil.rmtree(self.nre_root / "docs")
        version_file = self.nre_root / "bazel/version/VERSION_FILE"
        version_file.parent.mkdir(parents=True)
        for contents in (None, "VERSION_MAJOR=91\n", "VERSION_MAJOR=91\nVERSION_MINOR=invalid\n"):
            with self.subTest(contents=contents):
                if contents is not None:
                    version_file.write_text(contents)
                result = self._run("--agent", "claude", "--preflight-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("bazel/version/VERSION_FILE", result.stdout + result.stderr)
                self.assertIn("set nurec.dev_deps_image", result.stdout + result.stderr)

    def test_released_checkout_commands_use_the_built_image(self) -> None:
        """Later commands run the image built for this checkout."""
        result = self._run("--agent", "claude")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rendered = (self.workdir / "nurec-runtime.md").read_text()
        self.assertIn("Development image: nurec_dev:onboard-", rendered)

    def test_development_checkout_uses_pinned_image_and_readonly_auth(self) -> None:
        """Build the actual development checkout, without switching to released source."""
        image = "registry.example.com/nurec:dev-test"
        (self.nre_root / ".gitlab-ci.yml").write_text(
            f"default:\n  image:\n    # Comment.\n    name: {image}\n  tags: []\n"
        )
        missing = self._run("--agent", "claude", "--preflight-only")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("readable ~/.netrc", missing.stdout)
        (self.home / ".netrc").write_text("test fixture; never read by the launcher")
        result = self._run("--agent", "claude")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rendered = (self.workdir / "nurec-runtime.md").read_text()
        self.assertIn(f"Source: {self.nre_root}", rendered)
        self.assertIn(f"Development image: {image}", rendered)
        self.assertIn(f"{self.home}/.netrc:{self.workdir}/nurec-bazel/build-home/.netrc:ro", rendered)
        self.assertNotIn("test fixture", rendered)

    def test_launch_replaces_webfetch_rules_with_documentation_hosts(self) -> None:
        """Leave one WebFetch rule per documentation host and no stale rule."""
        settings = self._read_template()
        settings["permissions"]["allow"] = [
            "Bash",
            "WebFetch",
            "WebFetch(domain:stale.example.com)",
        ]
        self._write_template(settings)

        result = self._run("--agent", "claude")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        allow = self._read_settings()["permissions"]["allow"]
        self.assertIn("Bash", allow)
        self.assertIn("WebFetch(domain:docs.example.com)", allow)
        self.assertNotIn("WebFetch", allow)
        self.assertNotIn("WebFetch(domain:stale.example.com)", allow)

    def test_launch_regenerates_settings_and_leaves_template_unchanged(self) -> None:
        """Discard the previous run's settings and never modify the template."""
        template_text = self.template_path.read_text()
        self.settings_path.write_text(json.dumps({"permissions": {"allow": ["Bash(rm:*)"]}}) + "\n")

        result = self._run("--agent", "claude")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.template_path.read_text(), template_text)
        settings = self._read_settings()
        self.assertNotIn("Bash(rm:*)", settings["permissions"]["allow"])
        self.assertEqual(settings["sandbox"]["enabled"], self._read_template()["sandbox"]["enabled"])

    def test_missing_template_refuses_to_launch(self) -> None:
        """Refuse to launch from a stale generated file when the template is gone."""
        self.settings_path.write_text(self.template_path.read_text())
        self.template_path.unlink()

        result = self._run("--agent", "claude")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("settings.local.template.json not found", result.stdout)
        self.assertFalse(self.claude_log.exists())

    def test_disabled_sandbox_refuses_to_launch(self) -> None:
        """Refuse to launch once the sandbox has been switched off."""
        self._patch_sandbox(enabled=False)

        result = self._run("--agent", "claude")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sandbox.enabled is false", result.stdout)
        self.assertFalse(self.claude_log.exists())

    def test_silent_sandbox_fallback_refuses_to_launch(self) -> None:
        """Refuse to launch when an unavailable sandbox would not be fatal."""
        self._patch_sandbox(failIfUnavailable=False)

        result = self._run("--agent", "claude")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sandbox.failIfUnavailable is not true", result.stdout)
        self.assertFalse(self.claude_log.exists())

    def test_unsandboxed_retry_escape_refuses_to_launch(self) -> None:
        """Refuse to launch while a failed command may be retried unsandboxed."""
        self._patch_sandbox(allowUnsandboxedCommands=True)

        result = self._run("--agent", "claude")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sandbox.allowUnsandboxedCommands is not false", result.stdout)
        self.assertFalse(self.claude_log.exists())

    def test_preflight_reports_missing_python_for_workspace_trust(self) -> None:
        """Attribute a missing Python interpreter to the trust check."""
        bash_env = self.root / "bash-env"
        bash_env.write_text(
            "command() {\n"
            '  if [[ "${1:-}" == "-v" && "${2:-}" == "python3" ]]; then return 1; fi\n'
            '  builtin command "$@"\n'
            "}\n"
        )

        result = self._run(
            "--agent",
            "claude",
            "--preflight-only",
            extra_env={"BASH_ENV": str(bash_env)},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Claude workspace trust could not be checked because python3 is not installed", result.stdout)

    def test_preflight_requires_workspace_trust(self) -> None:
        """Reject a checkout whose trust dialog was never accepted."""
        self.claude_config.write_text(
            json.dumps({"projects": {str(self.nurec_onboard_dir): {"hasTrustDialogAccepted": False}}}) + "\n"
        )

        result = self._run("--agent", "claude", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Claude workspace trust is not recorded", result.stdout)
        self.assertNotIn("~/.claude.json not found", result.stdout)

    def _trust_nested_checkout(self, projects: dict[str, dict[str, Any]]) -> None:
        """Nest the checkout in a git repository and record Claude trust entries.

        Args:
            projects: Claude ``projects`` entries to record; the keys ``root``
                and ``checkout`` stand for the repository root and the checkout.
        """
        root, checkout = self._nest_checkout_in_git_repository()
        paths = {"root": str(root), "checkout": str(checkout)}
        self.claude_config.write_text(
            json.dumps({"projects": {paths[key]: value for key, value in projects.items()}}) + "\n"
        )

    def test_preflight_accepts_trust_of_the_repository_root(self) -> None:
        """Accept a checkout nested in a repository whose root is trusted."""
        self._trust_nested_checkout({"root": {"hasTrustDialogAccepted": True}})

        result = self._run("--agent", "claude", "--preflight-only")

        self.assertIn(f"Claude workspace is trusted (via repository root {self.root.resolve()})", result.stdout)
        self.assertNotIn("Claude workspace trust is not recorded", result.stdout)

    def test_preflight_prefers_the_checkout_entry_over_the_root(self) -> None:
        """Reject a checkout marked untrusted even when its root is trusted."""
        self._trust_nested_checkout(
            {
                "root": {"hasTrustDialogAccepted": True},
                "checkout": {"hasTrustDialogAccepted": False},
            }
        )

        result = self._run("--agent", "claude", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Claude workspace trust is not recorded", result.stdout)
        self.assertIn(f"or its repository root {self.root.resolve()}", result.stdout)

    def test_preflight_reports_missing_trust_config(self) -> None:
        """Name the absent configuration when no trust file exists at all."""
        self.claude_config.unlink()

        result = self._run("--agent", "claude", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("~/.claude.json not found", result.stdout)

    def test_preflight_reports_unreadable_trust_config(self) -> None:
        """Separate a configuration that cannot be read from a missing one."""
        # A directory raises the same OSError the check reports as unreadable,
        # and raises it for any test user; a mode-000 file stays readable when
        # the tests run as root.
        self.claude_config.unlink()
        self.claude_config.mkdir()

        result = self._run("--agent", "claude", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("~/.claude.json is not readable", result.stdout)

    def test_preflight_reports_malformed_trust_config(self) -> None:
        """Separate invalid JSON from an absent or unreadable configuration."""
        self.claude_config.write_text("{not json")

        result = self._run("--agent", "claude", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("~/.claude.json is malformed", result.stdout)

    def _hide_command(self, name: str) -> dict[str, str]:
        """Return an environment that hides one command from ``command -v``.

        Args:
            name: Command to report as absent.

        Returns:
            Environment overrides that install the shim for a launcher run.
        """
        bash_env = self.root / f"bash-env-{name}"
        bash_env.write_text(
            "command() {\n"
            f'  if [[ "${{1:-}}" == "-v" && "${{2:-}}" == "{name}" ]]; then return 1; fi\n'
            '  builtin command "$@"\n'
            "}\n"
        )
        return {"BASH_ENV": str(bash_env)}

    def test_preflight_reports_missing_bwrap(self) -> None:
        """Fail pre-flight on the Claude path when bubblewrap is absent."""
        result = self._run(
            "--agent",
            "claude",
            "--preflight-only",
            extra_env=self._hide_command("bwrap"),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bubblewrap (bwrap) is not installed", result.stdout)
        self.assertFalse(self.claude_log.exists())

    def test_preflight_reports_missing_socat(self) -> None:
        """Fail pre-flight when the Claude sandbox has no socat to proxy through."""
        result = self._run(
            "--agent",
            "claude",
            "--preflight-only",
            extra_env=self._hide_command("socat"),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("socat is not installed", result.stdout)
        self.assertFalse(self.claude_log.exists())

    def test_preflight_accepts_a_working_sandbox_host(self) -> None:
        """Pass the prerequisite check when bwrap starts and socat is present."""
        result = self._run("--agent", "claude", "--preflight-only")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("bubblewrap starts a sandbox", result.stdout)
        self.assertIn("socat is installed", result.stdout)

    def test_preflight_separates_blocked_bwrap_from_absent_bwrap(self) -> None:
        """Fail with the kernel remedy when bwrap is installed but cannot start."""
        self._write_executable(
            "bwrap",
            "#!/usr/bin/env bash\necho 'bwrap: No permissions to creating new namespace' >&2\nexit 1\n",
        )

        result = self._run("--agent", "claude", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bubblewrap is installed but cannot start a sandbox", result.stdout)
        self.assertIn("Enable unprivileged user namespaces", result.stdout)
        self.assertNotIn("is not installed", result.stdout)
        self.assertFalse(self.claude_log.exists())


if __name__ == "__main__":
    unittest.main()
