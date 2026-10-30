# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if deps/npm/pnpm-lock.yaml is out of sync with deps/npm/package.json.

This is a hermetic check (no network): it compares the dependency
name -> version-range specifiers declared in package.json against the top-level
importer specifiers recorded in the pnpm lockfile. They must match exactly. If
they do not, the lockfile needs regenerating with `bazel run //deps/npm:update_lock`.

It compares only the declared specifiers, so it catches a package.json edit that was
not followed by a regeneration. It does not inspect the lockfile's packages section, so a
resolved version or integrity hash edited directly there would slip through (validating
those would require contacting the registry).
"""

import json
import os

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def package_json_specifiers(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    specifiers: dict[str, str] = {}
    for field in ("dependencies", "devDependencies", "optionalDependencies"):
        specifiers.update(data.get(field) or {})
    return specifiers


def lockfile_specifiers(path: str) -> dict[str, str]:
    """Return name -> declared specifier for the lockfile's root importer.

    A pnpm v9 lockfile nests every importer under `importers`, keyed by its path
    relative to the lockfile; `.` is the root one. We have no workspace, so `.` is
    the only importer:

        importers:
          .:
            devDependencies:
              prettier:
                specifier: ^2.8.7
                version: 2.8.8
    """
    with open(path, encoding="utf-8") as handle:
        lock = yaml.safe_load(handle)
    root = (lock.get("importers") or {}).get(".") or {}
    specifiers: dict[str, str] = {}
    for field in ("dependencies", "devDependencies", "optionalDependencies"):
        for name, entry in (root.get(field) or {}).items():
            specifiers[name] = entry["specifier"]
    return specifiers


def test_lockfile_matches_package_json() -> None:
    package = package_json_specifiers(os.path.join(SCRIPT_DIR, "package.json"))
    lock = lockfile_specifiers(os.path.join(SCRIPT_DIR, "pnpm-lock.yaml"))

    mismatches = []
    for name, specifier in sorted(package.items()):
        if name not in lock:
            mismatches.append(f"  {name} ({specifier}) is in package.json but missing from the lockfile")
        elif lock[name] != specifier:
            mismatches.append(f"  {name}: package.json requests '{specifier}' but the lockfile records '{lock[name]}'")
    for name in sorted(set(lock) - set(package)):
        mismatches.append(f"  {name} is in the lockfile but not in package.json")

    assert not mismatches, "\n".join(
        ["deps/npm/pnpm-lock.yaml is out of sync with deps/npm/package.json:", *mismatches]
        + ["", "Regenerate it with: bazel run //deps/npm:update_lock"]
    )
