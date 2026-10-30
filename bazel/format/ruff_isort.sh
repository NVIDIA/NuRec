#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper to call Ruff via Bazel for import sorting, formatting and linting.
#
# This is wired into //:format (fix mode) and //:format.check (check mode) as
# the `python` formatter of the format_multirun in bazel/format/BUILD.bazel.

# Processed files: format_multirun enumerates the files to process with
# `git ls-files -t --cached --modified --other --exclude-standard`, i.e. tracked
# files plus untracked-but-not-gitignored files, and passes them as positional
# arguments, adding `--check` in check mode.
#
# Two passes run on that same file list:
#   1. `ruff check` -- import sorting (isort) and linting in one pass. The rule
#      selection (including `I` for import sorting and `RUF018` for assignment
#      expressions in asserts) and per-file ignores live in [tool.ruff.lint]
#      in pyproject.toml.
#   2. `ruff format` -- formatting. Kept separate because `ruff format` does not
#      sort imports (https://docs.astral.sh/ruff/formatter/#sorting-imports).

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

# Get the path to the Ruff binary from Bazel
RUFF_BIN="$(rlocation multitool/tools/ruff/ruff)"

# Get the workspace root
WORKSPACE_ROOT=$(bazel info workspace)

# We are called with either
# - check-mode: `format --check --force-exclude --diff FILES...` (check mode)
# or
# - fix-mode: `format --force-exclude FILES...`

# Parse arguments to distinguish between --check (diff mode) and normal (fix mode)
# and collect the files to process.  Files go into an array so they survive
# expansion as individual ruff arguments (a space-separated string would be
# passed as a single non-existent path to ruff).
FILES=()
CHECK_ARG=""
for arg in "$@"; do
    if [[ "$arg" = "--check" ]]; then
        CHECK_ARG="--check"
    else
        abs_path="$WORKSPACE_ROOT/$arg"
        if [[ -f "$abs_path" ]]; then
            FILES+=("$abs_path")
        fi
    fi
done

if [[ -n "$CHECK_ARG" ]]; then
    # Check mode: fail on any lint/import-order violation, then on format diff.
    # Do not pass `--diff` to `ruff check`: it implies `--fix-only`, which
    # exits 0 for unfixable rules such as RUF018 and would not gate CI.
    set -e
    "$RUFF_BIN" check --force-exclude "${FILES[@]}"
    "$RUFF_BIN" format --check --force-exclude --diff "${FILES[@]}"
else
    # Fix mode: apply import sorting + auto-fixable lint fixes, then format.
    # Do not abort on unfixable violations so that //:format still formats
    # everything it can; //:format.check reports the remaining violations.
    "$RUFF_BIN" check --force-exclude --fix "${FILES[@]}" || true
    "$RUFF_BIN" format --force-exclude "${FILES[@]}"
fi
