# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Codex launcher integration."""

from __future__ import annotations

import socket
import unittest

from lib.launcher_test_base import LauncherTestBase


class CodexLauncherIntegrationTest(LauncherTestBase):
    """Verify the Codex launcher path and generated arguments."""

    def setUp(self) -> None:
        """Configure an isolated trusted Codex installation."""
        super().setUp()
        self.codex_home = self.root / "codex"
        self.codex_home.mkdir()
        self.bazel_cache = self.root / "bazel-cache"
        self.bazel_cache.mkdir()
        self.home.joinpath(".cache").mkdir()
        self.home.joinpath(".cache", "bazel").symlink_to(
            self.bazel_cache,
            target_is_directory=True,
        )
        self.codex_log = self.root / "codex.args"
        self.codex_stdin = self.root / "codex.stdin"
        self.codex_bazelrc = self.root / "codex.bazelrc"
        self.codex_bazel = self.root / "codex.bazel"
        self._write_executable(
            "codex",
            "#!/usr/bin/env bash\n"
            'if [[ "${1:-}" == "--version" ]]; then\n'
            '  echo "codex-cli ${FAKE_CODEX_VERSION:-0.147.0}"\n'
            "  exit 0\n"
            "fi\n"
            'if [[ "${1:-}" == "sandbox" ]]; then\n'
            '  if [[ "${2:-}" == "--help" ]]; then\n'
            "    exit 0\n"
            "  fi\n"
            '  exit "${FAKE_CODEX_SANDBOX_EXIT:-0}"\n'
            "fi\n"
            'if [[ "${1:-}" == "execpolicy" ]]; then\n'
            '  if [[ "$*" == *--help* ]]; then\n'
            "    exit 0\n"
            "  fi\n"
            '  printf \'{"decision":"%s"}\\n\' "${FAKE_CODEX_EXECPOLICY_DECISION:-allow}"\n'
            "  exit 0\n"
            "fi\n"
            'printf \'%s\\n\' "$@" > "$FAKE_CODEX_LOG"\n'
            'command -v bazel > "$FAKE_CODEX_BAZEL"\n'
            # The launcher removes its managed rc on exit, so capture it while
            # the run is still in flight. The wrapper is the only thing that
            # names the file, so read the path back out of it.
            "sed -n 's/.*--bazelrc=\\([^ )]*\\).*/\\1/p' \"$(command -v bazel)\" "
            '| tr -d "\'" | xargs -I {} cp {} "$FAKE_CODEX_BAZELRC"\n'
            'dd of="$FAKE_CODEX_STDIN" status=none\n'
            # Stands in for Codex's own rendering, including a diff line, so a
            # test can tell that the launcher relays it rather than reading it.
            "printf '%s\\n' 'Continuing workflow.' '+ added_line'\n",
        )
        self.codex_home.joinpath("config.toml").write_text(
            f'[projects."{self.nurec_onboard_dir}"]\ntrust_level = "trusted"\n'
        )
        self.extra_environment.update(
            {
                "CODEX_HOME": str(self.codex_home),
                "FAKE_CODEX_LOG": str(self.codex_log),
                "FAKE_CODEX_STDIN": str(self.codex_stdin),
                "FAKE_CODEX_BAZELRC": str(self.codex_bazelrc),
                "FAKE_CODEX_BAZEL": str(self.codex_bazel),
            }
        )

    def test_launch_uses_fail_closed_generated_policy(self) -> None:
        """Launch Codex with scoped paths, domains, and no model override."""
        result = self._run("--agent", "codex")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Codex sandbox reaches allowlisted domains", result.stdout)
        args = self.codex_log.read_text().splitlines()
        rendered = "\n".join(args)
        self.assertEqual(args[0], "exec")
        self.assertIn("--strict-config", args)
        self.assertIn('approval_policy="never"', rendered)
        self.assertIn('default_permissions="nurec-onboard"', rendered)
        self.assertIn(f'"{self.workdir}"="write"', rendered)
        self.assertIn(f'"{self.bazel_cache}"="write"', rendered)
        self.assertIn(f'"{self.data}"="read"', rendered)
        self.assertIn(f'"{self.fake_bin}"="read"', rendered)
        self.assertIn(f'"{self.codex_home / "tmp"}"="read"', rendered)
        self.assertIn(f'"{self.home / ".cache" / "bazelisk"}"="read"', rendered)
        self.assertIn(f'"{self.nre_root}"="read"', rendered)
        self.assertIn('"docs.example.com"="allow"', rendered)
        self.assertNotIn("--model", args)
        self.assertIn("docker.sock", rendered)
        self.assertIn("Read the skill file", self.codex_stdin.read_text())
        # Codex owns the terminal: its output is relayed verbatim, and the
        # launcher neither selects an output format nor writes an event log.
        self.assertIn("Continuing workflow.\n+ added_line", result.stdout)
        self.assertNotIn("--json", args)
        self.assertEqual(list(self.workdir.glob("codex-*")), [])
        managed_cache_rc = self.codex_bazelrc.read_text()
        self.assertIn(f'build --repository_cache="{self.bazel_cache / "repository-cache"}"', managed_cache_rc)
        self.assertIn(f'build --disk_cache="{self.bazel_cache / "disk-cache" / "nurec-onboard"}"', managed_cache_rc)
        self.assertIn(f'startup --output_user_root="{self.workdir / ".bazel_roots"}"', managed_cache_rc)
        bazel_wrapper = self.workdir / ".nurec-bin" / "bazel"
        self.assertEqual(self.codex_bazel.read_text().strip(), str(bazel_wrapper))
        self.assertIn("--bazelrc=", bazel_wrapper.read_text())

    def test_extra_arguments_follow_the_generated_ones(self) -> None:
        """Forward operator arguments last, where a repeat of one wins."""
        result = self._run(
            "--agent",
            "codex",
            "--model",
            "generated-model",
            "--",
            "--json",
            "--model",
            "operator-model",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.codex_log.read_text().splitlines()
        # Last except for the `-` that keeps the prompt on stdin.
        self.assertEqual(args[-4:], ["--json", "--model", "operator-model", "-"])
        # Both occurrences are sent; Codex resolves the repeat, and the launcher
        # reports what it appended so an override is not silent.
        self.assertEqual(args.count("--model"), 2)
        self.assertIn("Args:     --json --model operator-model", result.stdout)

    def test_help_after_the_separator_belongs_to_codex(self) -> None:
        """Treat the separator as final: even --help passes through."""
        result = self._run("--agent", "codex", "--", "--help")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--help", self.codex_log.read_text().splitlines())
        self.assertNotIn("Show this help", result.stdout)

    def test_model_override_is_forwarded(self) -> None:
        """Forward an explicit model only when the operator supplies one."""
        result = self._run("--agent", "codex", "--model", "test-model")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.codex_log.read_text().splitlines()
        model_index = args.index("--model")
        self.assertEqual(args[model_index + 1], "test-model")

    def test_bazelisk_home_is_readable(self) -> None:
        """Honor Bazelisk's explicit cache location."""
        bazelisk_home = self.root / "custom-bazelisk"
        result = self._run(
            "--agent",
            "codex",
            extra_env={
                "BAZELISK_HOME": str(bazelisk_home),
                "XDG_CACHE_HOME": str(self.root / "xdg-cache"),
            },
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rendered = self.codex_log.read_text()
        self.assertIn(f'"{bazelisk_home}"="read"', rendered)
        self.assertNotIn(f'"{self.root / "xdg-cache" / "bazelisk"}"', rendered)

    def test_launch_grants_local_docs_outside_other_roots(self) -> None:
        """Grant documentation files and directories wherever they live."""
        docs_parent = self.root / "elsewhere"
        docs_dir = docs_parent / "docs"
        docs_dir.mkdir(parents=True)
        doc_file = docs_parent / "calibration.md"
        doc_file.write_text("calibration")
        # The launcher grants /tmp for scratch, which covers this test fixture.
        # Remove that broader grant so the per-doc grants are observable.
        self._replace_in_launcher('    "/tmp" ', '    "/var/tmp" ')
        self.input_path.write_text(
            "data:\n"
            "  name: sample\n"
            f"  path: {self.data}\n"
            "  description: sample\n"
            "  docs:\n"
            f"    - {doc_file}\n"
            f"    - {docs_dir}\n"
        )

        result = self._run("--agent", "codex")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rendered = self.codex_log.read_text()
        self.assertIn(f'"{doc_file}"="read"', rendered)
        self.assertIn(f'"{docs_dir}"="read"', rendered)
        self.assertNotIn(f'"{docs_parent}"="read"', rendered)

    def test_preflight_rejects_old_versions(self) -> None:
        """Reject Codex versions without custom permission profiles."""
        result = self._run(
            "--agent",
            "codex",
            "--preflight-only",
            extra_env={"FAKE_CODEX_VERSION": "0.137.0"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("0.138.0 or newer is required", result.stdout)

    def test_preflight_reports_missing_python_for_codex_setup(self) -> None:
        """Attribute a missing Python interpreter to the Codex setup check."""
        bash_env = self.root / "bash-env"
        bash_env.write_text(
            "command() {\n"
            '  if [[ "${1:-}" == "-v" && "${2:-}" == "python3" ]]; then return 1; fi\n'
            '  builtin command "$@"\n'
            "}\n"
        )

        result = self._run(
            "--agent",
            "codex",
            "--preflight-only",
            extra_env={"BASH_ENV": str(bash_env)},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Codex setup could not be checked because python3 is not installed", result.stdout)

    def test_preflight_requires_workspace_trust(self) -> None:
        """Reject a checkout without persisted Codex trust."""
        self.codex_home.joinpath("config.toml").write_text("")

        result = self._run("--agent", "codex", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Codex workspace trust is not recorded", result.stdout)

    def _trust_nested_checkout(self, config: str) -> None:
        """Nest the checkout in a git repository and write Codex trust entries.

        Args:
            config: Codex user configuration to write; ``{root}`` and
                ``{checkout}`` expand to the repository root and the checkout.
        """
        root, checkout = self._nest_checkout_in_git_repository()
        self.codex_home.joinpath("config.toml").write_text(config.format(root=root, checkout=checkout))

    def test_preflight_accepts_trust_of_the_repository_root(self) -> None:
        """Accept a checkout nested in a repository whose root is trusted."""
        self._trust_nested_checkout('[projects."{root}"]\ntrust_level = "trusted"\n')

        result = self._run("--agent", "codex", "--preflight-only")

        self.assertIn(f"Codex workspace is trusted (via repository root {self.root.resolve()})", result.stdout)
        self.assertNotIn("Codex workspace trust is not recorded", result.stdout)

    def test_preflight_prefers_the_checkout_entry_over_the_root(self) -> None:
        """Reject a checkout marked untrusted even when its root is trusted."""
        self._trust_nested_checkout(
            '[projects."{root}"]\ntrust_level = "trusted"\n[projects."{checkout}"]\ntrust_level = "untrusted"\n'
        )

        result = self._run("--agent", "codex", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Codex workspace trust is not recorded", result.stdout)
        self.assertIn(f"or its repository root {self.root.resolve()}", result.stdout)

    def test_preflight_rejects_unreachable_allowlisted_domains(self) -> None:
        """Reject a sandbox whose allowlisted egress does not work."""
        result = self._run(
            "--agent",
            "codex",
            "--preflight-only",
            extra_env={"FAKE_CODEX_SANDBOX_EXIT": "1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Codex sandbox cannot reach", result.stdout)
        self.assertIn("features.network_proxy", result.stdout)

    def test_preflight_reports_docs_are_readable(self) -> None:
        """Pass local documentation that sits outside every other root."""
        outside_doc = self.root / "outside.md"
        outside_doc.write_text("outside")
        self.input_path.write_text(
            f"data:\n  name: sample\n  path: {self.data}\n  description: sample\n  docs: [{outside_doc}]\n"
        )

        result = self._run("--agent", "codex", "--preflight-only")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Local data.docs paths are readable by Codex", result.stdout)

    def test_preflight_rejects_sensitive_docs_directory(self) -> None:
        """Reject a data.docs entry that would grant a sensitive subtree."""
        self.input_path.write_text(
            f"data:\n  name: sample\n  path: {self.data}\n  description: sample\n  docs: [/etc]\n"
        )

        result = self._run("--agent", "codex", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("data.docs entry resolves to a sensitive root: /etc", result.stdout)

    def test_launch_resolves_symlinked_codex_home_tmp(self) -> None:
        """Grant the resolved Codex helper directory when CODEX_HOME is symlinked."""
        real_codex_home = self.root / "real-codex"
        self.codex_home.rename(real_codex_home)
        self.codex_home.symlink_to(real_codex_home, target_is_directory=True)

        result = self._run("--agent", "codex")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rendered = self.codex_log.read_text()
        self.assertIn(f'"{real_codex_home / "tmp"}"="read"', rendered)
        self.assertNotIn(f'"{self.codex_home / "tmp"}"="read"', rendered)

    def test_preflight_reports_host_docker_route(self) -> None:
        """Pass when the rule allows Docker to run outside the sandbox."""
        self._write_executable("docker", "#!/usr/bin/env bash\nexit 0\n")
        docker_socket_path = self.root / "docker.sock"
        self._replace_in_launcher(
            'local docker_sock="/var/run/docker.sock"',
            f'local docker_sock="{docker_socket_path}"',
        )

        with socket.socket(socket.AF_UNIX) as docker_socket:
            docker_socket.bind(str(docker_socket_path))
            result = self._run(
                "--agent",
                "codex",
                "--preflight-only",
                step="run-nurec",
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Codex runs Docker commands on the host", result.stdout)

    def test_preflight_rejects_missing_docker_rule(self) -> None:
        """Reject a policy that does not let Docker leave the sandbox."""
        self._write_executable("docker", "#!/usr/bin/env bash\nexit 0\n")
        self.extra_environment["FAKE_CODEX_EXECPOLICY_DECISION"] = "prompt"

        result = self._run("--agent", "codex", "--preflight-only", step="run-nurec")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Docker commands are not allowed out of the Codex sandbox", result.stdout)

    def test_preflight_rejects_deny_entry_that_blocks_docker(self) -> None:
        """Reject a denied read, which suppresses the rule's escalation."""
        self._write_executable("docker", "#!/usr/bin/env bash\nexit 0\n")
        # Filesystem tables merge across configuration layers, so a deny entry in
        # the user layer breaks Docker even though the tracked profile has none.
        self.codex_home.joinpath("config.toml").write_text(
            f'[projects."{self.nurec_onboard_dir}"]\ntrust_level = "trusted"\n'
            '\n[permissions.nurec-onboard.filesystem]\n":root" = "deny"\n'
        )

        result = self._run("--agent", "codex", "--preflight-only", step="run-nurec")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("denies reads", result.stdout)

    def test_audit_preflight_checks_image_without_creating_build_cache(self) -> None:
        """Audit validation needs the release image but not GPU or cache preparation."""
        for guide in (self.nre_root / "docs").rglob("software.*"):
            guide.unlink()
        self.input_path.write_text(
            self.input_path.read_text() + "nurec:\n  dev_deps_image: ghcr.io/nvidia/nurec/nurec_dev_deps:test-release\n"
        )
        docker_log = self.root / "docker.invocations"
        self._write_executable("docker", f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{docker_log}"\n')
        result = self._run("--agent", "codex", "--preflight-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("pull ghcr.io/nvidia/nurec/nurec_dev_deps:test-release", docker_log.read_text())
        self.assertNotIn("nvidia-smi", docker_log.read_text())
        self.assertFalse((self.workdir / "nurec-bazel").exists())

    def test_dev_image_is_built_on_the_pinned_dependency_image(self) -> None:
        """The checkout runs in the development image built on the pinned dev dependencies image."""
        docker_log = self.root / "docker.invocations"
        self._write_executable("docker", f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{docker_log}"\n')
        result = self._run("--agent", "codex", "--preflight-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        base = "ghcr.io/nvidia/nurec/nurec_dev_deps:test-release"
        self.assertIn(f"pull {base}", docker_log.read_text())
        self.assertIn(f"--build-arg BASE_IMAGE={base} -t nurec_dev:onboard-", docker_log.read_text())
        self.assertIn(f"NuRec source/image pair: {self.nre_root} / {base}", result.stdout)
        self.assertIn("NuRec development image is built: nurec_dev:onboard-", result.stdout)

    def test_preflight_rejects_legacy_sandbox_settings(self) -> None:
        """Reject settings that shadow the custom permission profile."""
        self.codex_home.joinpath("config.toml").write_text(
            f'sandbox_mode = "workspace-write"\n[projects."{self.nurec_onboard_dir}"]\ntrust_level = "trusted"\n'
        )

        result = self._run("--agent", "codex", "--preflight-only")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("custom permission profile is shadowed", result.stdout)


if __name__ == "__main__":
    unittest.main()
