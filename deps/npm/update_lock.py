#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate deps/npm/pnpm-lock.yaml after editing deps/npm/package.json.

    bazel run //deps/npm:update_lock

pnpm updates the lockfile incrementally, leaving already-satisfied entries untouched.

pnpm resolves through the Artifactory npm mirror, which needs a registry and auth. We pass
those to pnpm via a throwaway npmrc rather than committing them to deps/npm/.npmrc, because
a committed `registry=` line breaks normal builds: npm_import then presents empty
credentials to the mirror instead of falling back to ~/.netrc, and 401s. With no registry
committed, builds resolve the default registry.npmjs.org URLs, which Bazel's downloader
rewrites to the mirror and authenticates via ~/.netrc. (pnpm cannot read ~/.netrc itself, so
the token below is bridged from it.)
"""

import base64
import netrc
import os
import subprocess
import sys
import tempfile

from pathlib import Path

from python.runfiles import runfiles


REGISTRY = "https://artifactory.nvidia.com/artifactory/api/npm/sw-nrs-registry-npmjs-remote/"


def artifactory_npm_auth() -> str:
    """Return an npm basic-auth token bridged from the Artifactory ~/.netrc credentials."""
    try:
        creds = netrc.netrc().authenticators("artifactory.nvidia.com")
    except FileNotFoundError:
        sys.exit("error: ~/.netrc not found; cannot authenticate to the Artifactory npm mirror")
    except netrc.NetrcParseError as exc:
        sys.exit(f"error: failed to parse ~/.netrc: {exc}")
    if not creds:
        sys.exit("error: no 'machine artifactory.nvidia.com' entry in ~/.netrc")
    login, _, password = creds
    return base64.b64encode(f"{login}:{password}".encode()).decode()


def main() -> None:
    repo_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not repo_root:
        sys.exit("error: run this with `bazel run //deps/npm:update_lock`")
    pnpm = runfiles.Create().Rlocation(sys.argv[1])

    # npm's per-registry auth key is the registry URL without its scheme ("//host/path/").
    auth_key = REGISTRY.split("://", 1)[1]
    npmrc = "\n".join(
        [
            f"registry={REGISTRY}",
            f"//{auth_key}:_auth={artifactory_npm_auth()}",
            "always-auth=true",
            # pnpm 8.6.7 defaults resolution-mode to lowest-direct (reverted to highest in
            # 8.7.0), which would downgrade a range like ^2.8.7 to 2.8.7; pin highest to
            # match npm's conventional behavior.
            "resolution-mode=highest",
        ]
    )

    # Throwaway store + userconfig (the 0700 temp dir keeps the credentials owner-only) so
    # regeneration leaves nothing behind in the repo. The userconfig layers the mirror
    # registry + scoped auth on top of the committed deps/npm/.npmrc.
    with tempfile.TemporaryDirectory() as store_dir:
        regen_npmrc = Path(store_dir) / "npmrc"
        regen_npmrc.write_text(f"{npmrc}\nstore-dir={store_dir}\n", encoding="utf-8")

        print("Regenerating deps/npm/pnpm-lock.yaml through the Artifactory npm mirror...", flush=True)
        result = subprocess.run(
            [pnpm, "--dir", os.path.join(repo_root, "deps", "npm"), "install", "--lockfile-only"],
            env={
                **os.environ,
                "BAZEL_BINDIR": ".",
                "npm_config_userconfig": str(regen_npmrc),
            },
            check=False,
        )

    if result.returncode != 0:
        sys.exit(result.returncode)
    print("Done. Review and commit deps/npm/pnpm-lock.yaml (and deps/npm/package.json).")


if __name__ == "__main__":
    main()
