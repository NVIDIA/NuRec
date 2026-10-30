# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the AV Benchmark chain from a declarative spec.

Tasks run in order; `$VAR` placeholders resolve from spec vars and earlier tasks' outputs.
No DAG, retries, parallelism or scheduling -- an external orchestrator owns those.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time

from pathlib import Path
from typing import Any

import yaml

from python.runfiles import runfiles  # type: ignore[import-not-found]


# Runfiles locations of the binaries a task may name.
BINARIES = {
    "nurec": "_main/run",
    "nurec_tools": "_main/apps/nurec_tools",
}

# Published "reconstruction time by phase" breakdown. Unphased tasks are timed but excluded.
RECONSTRUCTION_PHASES = ("input_preparation", "training_and_optimization", "export_and_packaging")

_VAR = re.compile(r"\$([A-Z_][A-Z0-9_]*)")


def resolve(value: str, variables: dict[str, str]) -> str:
    """Substitute $VAR placeholders, failing loudly on an unknown name."""

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise KeyError(f"undefined variable ${name}")
        return variables[name]

    return _VAR.sub(_sub, value)


def load_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("tasks"), list) or not spec["tasks"]:
        raise ValueError(f"{path}: spec must define a non-empty 'tasks' list")
    if not isinstance(spec.get("vars") or {}, dict):
        raise ValueError(f"{path}: 'vars' must be a mapping")
    for task in spec["tasks"]:
        if not isinstance(task, dict):
            raise ValueError(f"{path}: every task must be a mapping, got {type(task).__name__}")
        if not isinstance(task.get("name"), str) or not task["name"]:
            raise ValueError(f"{path}: every task must define a non-empty 'name'")
        if task.get("binary") not in BINARIES:
            raise ValueError(f"{path}: task {task.get('name')!r} has unknown binary {task.get('binary')!r}")
        if not isinstance(task.get("args"), list):
            raise ValueError(f"{path}: task {task.get('name')!r} must define an 'args' list")
        if not isinstance(task.get("produces") or {}, dict):
            raise ValueError(f"{path}: task {task['name']!r} must define 'produces' as a mapping")
        if not isinstance(task.get("optional", False), bool):
            raise ValueError(f"{path}: task {task['name']!r} must define 'optional' as a boolean")
        if (phase := task.get("phase")) is not None and phase not in RECONSTRUCTION_PHASES:
            raise ValueError(
                f"{path}: task {task['name']!r} has unknown phase {phase!r}, "
                f"expected one of {', '.join(RECONSTRUCTION_PHASES)}"
            )
    return spec


def _binary_path(binary: str) -> str:
    resolved = runfiles.Create().Rlocation(BINARIES[binary])
    if not resolved or not Path(resolved).exists():
        raise FileNotFoundError(f"{binary}: not in runfiles at {BINARIES[binary]}")
    return resolved


def run_workflow(
    spec: dict[str, Any],
    variables: dict[str, str],
    work_dir: Path,
    enabled_optional: frozenset[str] = frozenset(),
) -> int:
    results: list[dict[str, Any]] = []
    phase_seconds: dict[str, float] = {}
    exit_code = 0
    manifest = work_dir / "manifest.json"

    def write_manifest() -> None:
        manifest.write_text(
            json.dumps(
                {
                    "name": spec.get("name"),
                    "tasks": results,
                    "phase_seconds": {phase: round(seconds, 1) for phase, seconds in phase_seconds.items()},
                    "reconstruction_seconds": round(sum(phase_seconds.values()), 1),
                },
                indent=2,
            )
        )

    write_manifest()

    for task in spec["tasks"]:
        name = task["name"]
        phase = task.get("phase")
        # Recorded as skipped rather than dropped, so a manifest shows what was not run.
        if task.get("optional") and name not in enabled_optional:
            print(f"\n=== [{name}] optional, not enabled; skipping", flush=True)
            results.append({"task": name, "phase": phase, "skipped": True})
            # No `produces`: a task that did not run wrote nothing to point later tasks at.
            write_manifest()
            continue
        args = [resolve(str(a), variables) for a in task["args"]]
        cmd = [_binary_path(task["binary"]), *args]
        print(f"\n=== [{name}] {shlex.join(cmd)}", flush=True)

        started = time.monotonic()
        completed = subprocess.run(cmd, check=False)
        elapsed = time.monotonic() - started
        print(f"=== [{name}] exit={completed.returncode} in {elapsed:.1f}s", flush=True)

        results.append({"task": name, "phase": phase, "returncode": completed.returncode, "seconds": round(elapsed, 1)})
        if phase is not None:
            phase_seconds[phase] = phase_seconds.get(phase, 0.0) + elapsed
        write_manifest()
        if completed.returncode != 0:
            exit_code = completed.returncode
            print(f"=== [{name}] failed; stopping", flush=True)
            break

        for key, template in (task.get("produces") or {}).items():
            variables[key] = resolve(str(template), variables)

    write_manifest()
    print(f"\nWrote {manifest}", flush=True)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Path to the workflow spec YAML")
    parser.add_argument("--work-dir", type=Path, required=True, help="Directory for all task outputs")
    parser.add_argument(
        "--clip-root",
        type=str,
        help="Directory holding a raw PhysicalAI-AV clip; required unless --with download is enabled.",
    )
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="Override a spec variable")
    parser.add_argument(
        "--with",
        dest="with_tasks",
        action="append",
        default=[],
        metavar="NAME",
        help="Enable an optional task by name. Repeatable.",
    )
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    variables: dict[str, str] = {str(k): str(v) for k, v in (spec.get("vars") or {}).items()}
    variables["WORK"] = str(args.work_dir.resolve())
    # The public CLI takes configs by absolute path only (no root flag reaches it
    # anymore): locate the spec's repo-relative preset in runfiles so it composes
    # regardless of the caller's cwd. An absolute (--set) CONFIG is left untouched.
    if "CONFIG" in variables and not Path(variables["CONFIG"]).is_absolute():
        runfiles_path = f"_main/{variables['CONFIG']}"
        located = runfiles.Create().Rlocation(runfiles_path)
        if not located or not Path(located).exists():
            raise FileNotFoundError(f"CONFIG is not in runfiles at {runfiles_path}")
        variables["CONFIG"] = str(Path(located))
    if args.clip_root is not None:
        variables["CLIP_ROOT"] = args.clip_root

    for override in args.set:
        key, sep, value = override.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {override!r}")
        variables[key] = value

    optional_names = {task["name"] for task in spec["tasks"] if task.get("optional")}
    unknown = sorted(set(args.with_tasks) - optional_names)
    if unknown:
        raise SystemExit(f"--with names no optional task: {unknown}; optional tasks are {sorted(optional_names)}")

    if "download" not in args.with_tasks and (not args.clip_root or not variables.get("CLIP_ROOT")):
        parser.error("--clip-root is required unless --with download is enabled")

    return run_workflow(spec, variables, args.work_dir, frozenset(args.with_tasks))


if __name__ == "__main__":
    sys.exit(main())
