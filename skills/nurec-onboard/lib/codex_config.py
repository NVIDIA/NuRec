# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build per-run Codex permission-profile overrides."""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from urllib.parse import urlparse

from lib.parse_input import get_nested, parse_input


FIXED_NETWORK_DOMAINS = {
    "bcr.bazel.build",
    "download-r2.pytorch.org",
    "download.pytorch.org",
    "files.pythonhosted.org",
    "github.com",
    "ghcr.io",
    "objects.githubusercontent.com",
    "pypi.org",
    "release-assets.githubusercontent.com",
    "releases.bazel.build",
}


def build_overrides(
    *,
    repo_root: Path,
    input_path: Path,
    data_path: Path,
    workdir: Path,
    requires_docker: bool,
    read_paths: list[Path],
    write_paths: list[Path],
) -> list[str]:
    """Build canonical Codex configuration overrides for one workflow run.

    Args:
        repo_root: NuRec Onboard checkout.
        input_path: Validated workflow input YAML.
        data_path: Validated raw dataset directory.
        workdir: Validated workflow output directory.
        requires_docker: Whether the selected manifest requires Docker.
        read_paths: Additional readable Codex runtime paths.
        write_paths: Additional writable cache and scratch paths.

    Returns:
        Codex ``-c`` values for filesystem, network, and Docker permissions.
    """
    repo_root = repo_root.expanduser().resolve()
    input_path = input_path.expanduser().resolve(strict=False)
    data_path = data_path.expanduser().resolve(strict=False)
    workdir = workdir.expanduser().resolve(strict=False)
    canonical_writes = {path.expanduser().resolve(strict=False) for path in write_paths}
    canonical_writes.add(workdir)

    document = parse_input(input_path)
    canonical_reads = {path.expanduser().resolve(strict=False) for path in read_paths}
    canonical_reads.update({repo_root, data_path, input_path})
    granted_roots = canonical_reads | canonical_writes
    doc_grants, _ = _documentation_access(document, repo_root, granted_roots)
    canonical_reads.update(doc_grants)

    domains: set[str] = set()
    for item in _documentation_items(document):
        if item.startswith(("http://", "https://")):
            try:
                hostname = urlparse(item).hostname
            except ValueError:
                hostname = None
            if hostname:
                domains.add(hostname.lower())

    filesystem = {":minimal": "read"}
    for path in sorted(canonical_writes, key=lambda value: (len(value.parts), str(value))):
        filesystem[str(path)] = "write"
    for path in sorted(canonical_reads, key=lambda value: (len(value.parts), str(value))):
        filesystem.setdefault(str(path), "read")

    network_domains = {domain: "allow" for domain in FIXED_NETWORK_DOMAINS | domains}
    profile = "permissions.nurec-onboard"
    entries = [
        f"{profile}.filesystem={_inline_table(filesystem)}",
        f"{profile}.network.domains={_inline_table(network_domains)}",
    ]
    if requires_docker:
        # Codex's documented socket allowlist. It grants nothing on a Linux host,
        # where the sandbox namespace omits /run and the socket path therefore
        # does not exist inside it. Container-backed steps reach the daemon
        # instead through the `docker` execpolicy rule, which runs the command
        # outside the sandbox.
        entries.append(f"{profile}.network.unix_sockets={_inline_table({'/var/run/docker.sock': 'allow'})}")
    return entries


def _documentation_items(document: dict[str, object]) -> list[str]:
    docs = get_nested(document, "data.docs")
    if isinstance(docs, str):
        return [docs]
    if isinstance(docs, list):
        return [item for item in docs if isinstance(item, str)]
    return []


def _documentation_access(
    document: dict[str, object],
    repo_root: Path,
    granted_roots: set[Path],
) -> tuple[set[Path], set[Path]]:
    """Split existing local documentation paths into grants and refusals.

    Every local ``data.docs`` entry earns a read grant, matching the operator's
    declaration in the input YAML. A sensitive root is refused instead, since a
    single entry there would open its whole subtree to the agent.

    Args:
        document: Parsed input YAML.
        repo_root: NuRec Onboard checkout, used to resolve relative entries.
        granted_roots: Paths already readable or writable for the run.

    Returns:
        The paths to grant read access, and the sensitive roots to refuse.
    """
    grants: set[Path] = set()
    sensitive: set[Path] = set()
    home = Path.home().resolve(strict=False)
    sensitive_roots = {Path("/"), home, Path("/root"), Path("/etc"), Path("/usr"), Path("/var")}

    for item in _documentation_items(document):
        if item.startswith(("http://", "https://")):
            continue
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve(strict=False)
        if not candidate.exists():
            continue
        if candidate in sensitive_roots:
            sensitive.add(candidate)
        elif not any(candidate.is_relative_to(root) for root in granted_roots):
            # Covered entries stay out of the table; a broader grant already reaches them.
            grants.add(candidate)
    return grants, sensitive


def _inline_table(entries: dict[str, str]) -> str:
    items = (f"{json.dumps(key)}={json.dumps(value)}" for key, value in sorted(entries.items()))
    return "{" + ",".join(items) + "}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-run Codex permission-profile overrides.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--check-docs", action="store_true")
    parser.add_argument("--requires-docker", action="store_true")
    parser.add_argument("--read-path", action="append", default=[], type=Path)
    parser.add_argument("--write-path", action="append", default=[], type=Path)
    return parser.parse_args(argv)


def _main() -> None:
    args = _parse_args()
    if args.check_docs:
        # Only the refusals matter here, and a sensitive root is refused whether
        # or not other grants reach it, so the caller need not supply them.
        document = parse_input(args.input)
        _, sensitive = _documentation_access(document, args.repo_root.expanduser().resolve(), granted_roots=set())
        for path in sorted(sensitive):
            print(f"sensitive\t{path}")
        return

    overrides = build_overrides(
        repo_root=args.repo_root,
        input_path=args.input,
        data_path=args.data_path,
        workdir=args.workdir,
        requires_docker=args.requires_docker,
        read_paths=args.read_path,
        write_paths=args.write_path,
    )
    print("\n".join(overrides))


if __name__ == "__main__":
    _main()
