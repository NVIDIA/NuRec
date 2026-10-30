#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate that the host meets the prerequisites for the NuRec skill.

Checks (no network calls):
  - Host is Linux x86_64.
  - `docker` is on PATH and the daemon responds to `docker info`.
  - `nvidia-smi` is on PATH and reports an NVIDIA driver.
        - FAIL when major < 580 (source release requires R580+).
  - NVIDIA Container Toolkit is wired into Docker
    (looks for the `nvidia` runtime via `docker info`).
  - GPU has >= 24 GB of memory (best-effort via `nvidia-smi`).
  - Docker Buildx is available to build the development image.
  - Released source checkout and a cached development image with build tools.

Usage:
    python scripts/validate_setup.py [--source PATH] [--strict]

Arguments:
    --strict      Treat WARN lines as FAIL.
    --source      Released checkout; defaults to NUREC_SRC.

Environment variables:
    NUREC_SRC     Released checkout matching the server/client version.
    NUREC_IMAGE   Development image built on the matching dependency image.

Exit codes:
    0 - no hard failures. WARN lines may remain and DO still exit 0 unless
        --strict is passed, so 0 does NOT mean every prerequisite was
        verified -- read the WARN lines.
    1 - one or more prerequisites missing (details on stderr)
    2 - unexpected error (tool invocation crashed)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys


OK = "OK"
WARN = "WARN"
FAIL = "FAIL"

SUBPROCESS_TIMEOUT_S = 15
COMMAND_NOT_FOUND_RC = 127
UNEXPECTED_ERROR_EXIT = 2

MIN_DRIVER_MAJOR_FULL = 580

MIN_GPU_MEMORY_GB = 24.0
RECOMMENDED_GPU_MEMORY_GB = 48.0
MB_PER_GB = 1024.0


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return COMMAND_NOT_FOUND_RC, "", f"{cmd[0]}: not found"
    except Exception as exc:  # pragma: no cover - defensive
        return UNEXPECTED_ERROR_EXIT, "", f"{cmd[0]}: {exc}"


def check_platform() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine()
    if system != "Linux":
        return FAIL, f"OS is {system!r}; NuRec supports Linux only."
    if machine not in ("x86_64", "amd64"):
        return FAIL, (f"Architecture is {machine!r}; NuRec supports x86_64 only (aarch64 is not supported).")
    return OK, f"{system} {machine}"


def check_docker() -> tuple[str, str]:
    if shutil.which("docker") is None:
        return FAIL, "docker not on PATH. Install Docker >= 23.0.1."
    rc, out, err = _run(["docker", "--version"])
    if rc != 0:
        return FAIL, f"`docker --version` failed: {(err or out).strip()}"
    version_line = out.strip()
    # An unparsable version must NOT short-circuit the daemon check below:
    # returning early here meant a DEAD DAEMON still produced a non-strict
    # exit 0, because docker info was never reached. Record and carry on.
    version_note = ""
    m = re.search(r"(\d{1,9})\.(\d{1,9})\.(\d{1,9})", version_line)
    if not m:
        version_note = f" (could not parse a version from {version_line!r}; the >= 23.0.1 requirement is UNVERIFIED)"
    else:
        got = tuple(int(x) for x in m.groups())
        if got < (23, 0, 1):
            return FAIL, (f"Docker {'.'.join(map(str, got))} is older than the required 23.0.1 ({version_line}).")
    rc2, out2, err2 = _run(["docker", "info", "--format", "{{json .}}"])
    if rc2 != 0:
        return FAIL, (
            f"`docker info` failed ({(err2 or out2).strip() or 'no output'}). "
            "Is the Docker daemon running and is your user in the docker group?"
        )
    if version_note:
        return WARN, version_line + version_note
    return OK, version_line


def check_nvidia_runtime() -> tuple[str, str]:
    # Presence of the runtime is necessary but not sufficient: the docs require
    # Container Toolkit >= 1.13.5. Check the version when the CLI is available
    # rather than implying the requirement is enforced when it is not.
    toolkit_checked = False
    toolkit_reason = "nvidia-container-cli is not on PATH"
    if shutil.which("nvidia-container-cli") is not None:
        rc_t, out_t, err_t = _run(["nvidia-container-cli", "--version"])
        if rc_t != 0:
            detail = (err_t or out_t).strip().splitlines()
            first = detail[0] if detail else f"exit {rc_t}"
            toolkit_reason = f"nvidia-container-cli is present but `--version` failed ({first})"
        else:
            m = re.search(r"(\d{1,9})\.(\d{1,9})\.(\d{1,9})", out_t)
            if not m:
                toolkit_reason = (
                    f"nvidia-container-cli is present but its version output was unparsable ({out_t.strip()!r})"
                )
            else:
                toolkit_checked = True
                got = tuple(int(x) for x in m.groups())
                if got < (1, 13, 5):
                    return FAIL, (
                        f"NVIDIA Container Toolkit {'.'.join(map(str, got))} is older than the required 1.13.5."
                    )

    rc, out, err = _run(["docker", "info", "--format", "{{json .}}"])
    if rc != 0:
        return WARN, "Skipping (docker info failed; see docker check)."
    try:
        info = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return WARN, "Could not parse `docker info` JSON output."
    runtimes = info.get("Runtimes") or {}
    if "nvidia" in runtimes and not toolkit_checked:
        return WARN, (
            "The 'nvidia' runtime is registered with Docker, but the Container "
            f"Toolkit version is UNVERIFIED: {toolkit_reason}. The >= 1.13.5 "
            "requirement has not been checked; confirm it by hand."
        )
    if "nvidia" not in runtimes:
        return FAIL, (
            "NVIDIA Container Toolkit not wired into Docker (no 'nvidia' "
            "runtime). See https://docs.nvidia.com/datacenter/cloud-native/"
            "container-toolkit/install-guide.html"
        )
    return OK, "Docker runtime 'nvidia' is registered."


def check_driver() -> tuple[str, str]:
    if shutil.which("nvidia-smi") is None:
        return FAIL, "nvidia-smi not on PATH. Install the NVIDIA driver."
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    if rc != 0:
        return FAIL, f"nvidia-smi failed: {(err or out).strip() or 'unknown'}"
    first = out.strip().splitlines()[0] if out.strip() else ""
    match = re.match(r"^(\d+)\.(\d+)", first)
    if not match:
        return WARN, f"Could not parse driver version from {first!r}."
    major = int(match.group(1))
    if major < MIN_DRIVER_MAJOR_FULL:
        return FAIL, f"Driver {first} is below R{MIN_DRIVER_MAJOR_FULL}, required by this source release."
    return OK, f"NVIDIA driver {first}"


def check_gpu_memory() -> tuple[str, str]:
    if shutil.which("nvidia-smi") is None:
        return WARN, "nvidia-smi not on PATH; skipping GPU memory check."
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        return WARN, "Could not query GPU memory via nvidia-smi."
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return WARN, "No GPUs reported by nvidia-smi."
    try:
        sizes_mb = [int(line) for line in lines]
    except ValueError:
        return WARN, f"Unexpected nvidia-smi memory output: {out.strip()!r}"
    biggest_gb = max(sizes_mb) / MB_PER_GB
    if biggest_gb < MIN_GPU_MEMORY_GB:
        return FAIL, (
            f"Largest GPU has {biggest_gb:.1f} GB; NuRec needs >= "
            f"{MIN_GPU_MEMORY_GB:.0f} GB ({RECOMMENDED_GPU_MEMORY_GB:.0f}+ GB recommended)."
        )
    if biggest_gb < RECOMMENDED_GPU_MEMORY_GB:
        return WARN, (
            f"Largest GPU has {biggest_gb:.1f} GB; usable but "
            f"{RECOMMENDED_GPU_MEMORY_GB:.0f}+ GB is recommended for AV training."
        )
    return OK, f"Largest GPU has {biggest_gb:.1f} GB VRAM"


def check_buildx() -> tuple[str, str]:
    rc, out, err = _run(["docker", "buildx", "version"])
    if rc:
        return FAIL, f"Docker Buildx is required to build the development image: {(err or out).strip()}"
    return OK, out.strip()


def check_source_environment() -> tuple[str, str]:
    source = os.environ.get("NUREC_SRC", "")
    if not os.path.isabs(source) or not os.path.isfile(os.path.join(source, "MODULE.bazel")):
        return FAIL, "Set NUREC_SRC (or --source) to the absolute released checkout."
    image = os.environ.get("NUREC_IMAGE", "")
    if not image:
        return FAIL, "Set NUREC_IMAGE to the development image built using references/install.md."
    rc, _, _ = _run(["docker", "image", "inspect", image])
    if rc:
        return FAIL, f"Development image is not cached: {image}; build it using references/install.md."
    rc, out, err = _run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"nurec-setup-check-{os.getpid()}",
            "--pull=never",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "command -v bazelisk && command -v git && command -v gcc && command -v python3",
        ]
    )
    if rc:
        return FAIL, (
            f"Build-tool probe failed in {image}: {(err or out).strip() or f'exit {rc}'}. "
            "Build NUREC_IMAGE on NUREC_DEV_DEPS_IMAGE using references/install.md."
        )
    return OK, "Source checkout and development-image tools present (NuRec build/runtime not tested)."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (non-zero exit code).",
    )
    parser.add_argument("--source", default=os.environ.get("NUREC_SRC", ""), help="Released source checkout.")
    args = parser.parse_args()
    os.environ["NUREC_SRC"] = args.source

    checks = [
        ("Platform", check_platform),
        ("Docker", check_docker),
        ("Docker Buildx", check_buildx),
        ("NVIDIA runtime", check_nvidia_runtime),
        ("NVIDIA driver", check_driver),
        ("GPU memory", check_gpu_memory),
        ("Source environment", check_source_environment),
    ]

    failures = 0
    warnings = 0
    for name, fn in checks:
        try:
            status, detail = fn()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[{FAIL}] {name}: unexpected error: {exc}", file=sys.stderr)
            return UNEXPECTED_ERROR_EXIT
        stream = sys.stdout if status == OK else sys.stderr
        print(f"[{status}] {name}: {detail}", file=stream)
        if status == FAIL:
            failures += 1
        elif status == WARN:
            warnings += 1

    print(file=sys.stderr)
    if failures:
        print(
            f"{failures} prerequisite(s) missing; resolve before building NuRec from source.",
            file=sys.stderr,
        )
        return 1
    if args.strict and warnings:
        print(
            f"{warnings} warning(s) in --strict mode — treating as failure.",
            file=sys.stderr,
        )
        return 1
    if warnings:
        print(
            f"{warnings} warning(s); host is usable but review the messages.",
            file=sys.stderr,
        )
    else:
        print("All prerequisites look good.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
