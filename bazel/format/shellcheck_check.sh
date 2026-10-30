#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Bazel-driven shellcheck wrapper.  Enumerates shell scripts via
# `git ls-files` (same pattern as get_starlark_files() in
# bazel/format/starlark_custom_lint.py) so the file list:
#   - includes tracked AND new-untracked-not-gitignored scripts
#   - naturally excludes bazel-*, .git, node_modules, build outputs, etc.
#     because they're .gitignored
#   - covers .sh / .bash extensions plus extension-less executables that
#     start with a shell shebang
# Config lives in //:.shellcheckrc.  Wired into //:format.check via the
# :custom_check multirun.

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
source "$0.runfiles/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
{ echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

SHELLCHECK_BIN="$(rlocation multitool/tools/shellcheck/shellcheck)"
WORKSPACE_ROOT="${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}"
cd "$WORKSPACE_ROOT"

# 1. Tracked + new-untracked-not-ignored .sh / .bash files.
mapfile -t SH_BY_EXT < <(
  {
    git ls-files '*.sh' '*.bash'
    git ls-files --others --exclude-standard '*.sh' '*.bash'
  } | sort -u
)

# 2. Extension-less shell scripts that file(1) identifies.  Same git ls-files coverage
#    as section 1 (tracked + new-untracked-not-ignored), restricted to
#    executable files so we don't scan every text file in the repo.
mapfile -t SH_NOEXT < <(
  {
    git ls-files
    git ls-files --others --exclude-standard
  } | sort -u \
    | while IFS= read -r f; do
        case "$f" in *.sh|*.bash) continue ;; esac
        [[ -f "$f" && -x "$f" ]] || continue
        file --brief "$f" 2>/dev/null | grep -q "shell script" \
          && printf '%s\n' "$f"
      done
)

SH_FILES=("${SH_BY_EXT[@]}" "${SH_NOEXT[@]}")

if [[ "${#SH_FILES[@]}" -eq 0 ]]; then
  echo "shellcheck_check: no shell scripts found"
  exit 0
fi

echo "shellcheck_check: linting ${#SH_FILES[@]} shell scripts (config: //:.shellcheckrc)"
"$SHELLCHECK_BIN" -f gcc "${SH_FILES[@]}"
