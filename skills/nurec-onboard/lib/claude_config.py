# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build per-run Claude Code settings."""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lib.parse_input import get_nested, parse_input


def update_settings(
    *,
    template_path: Path,
    settings_path: Path,
    input_path: Path,
    write_paths: list[Path],
) -> None:
    """Generate Claude Code settings for one workflow run from the template.

    ``settings_path`` is replaced wholesale, so nothing from an earlier run
    or from edits Claude Code made to it carries over.

    Args:
        template_path: Committed settings template; read, never modified.
        settings_path: Claude Code project settings JSON to generate.
        input_path: Validated workflow input YAML.
        write_paths: Writable cache, scratch, and workflow paths.
    """
    settings = _load_settings(template_path)
    canonical_writes = list(
        dict.fromkeys(str(path.expanduser().resolve(strict=False)) for path in write_paths if str(path))
    )
    sandbox = settings.setdefault("sandbox", {})
    filesystem = sandbox.setdefault("filesystem", {})
    filesystem["allowWrite"] = canonical_writes

    permissions = settings.setdefault("permissions", {})
    existing = permissions.get("allow", [])
    if not isinstance(existing, list):
        raise TypeError("permissions.allow must be a list")
    non_webfetch = [
        entry
        for entry in existing
        if not (isinstance(entry, str) and (entry == "WebFetch" or entry.startswith("WebFetch(")))
    ]
    webfetch_rules = [f"WebFetch(domain:{hostname})" for hostname in _documentation_hosts(input_path)]
    permissions["allow"] = non_webfetch + webfetch_rules

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def _load_settings(settings_path: Path) -> dict[str, Any]:
    value = json.loads(settings_path.read_text())
    if not isinstance(value, dict):
        raise TypeError("Claude Code settings must be a JSON object")
    return value


def _documentation_hosts(input_path: Path) -> list[str]:
    document = parse_input(input_path)
    docs = get_nested(document, "data.docs")
    if docs is None:
        items: list[str] = []
    elif isinstance(docs, str):
        items = [docs]
    elif isinstance(docs, list):
        items = [item for item in docs if isinstance(item, str)]
    else:
        items = []

    hosts = set()
    for item in items:
        if not item.startswith(("http://", "https://")):
            continue
        try:
            hostname = urlparse(item).hostname
        except ValueError:
            hostname = None
        if hostname:
            hosts.add(hostname.lower())
    return sorted(hosts)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-run Claude Code settings from the template.")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--write-path", action="append", default=[], type=Path)
    return parser.parse_args(argv)


def _main() -> None:
    args = _parse_args()
    update_settings(
        template_path=args.template,
        settings_path=args.settings,
        input_path=args.input,
        write_paths=args.write_path,
    )


if __name__ == "__main__":
    _main()
