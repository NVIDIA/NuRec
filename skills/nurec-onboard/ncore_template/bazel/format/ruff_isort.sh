#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper that runs Ruff for both import sorting and formatting. Mirrors the
# NCore reference implementation (`ncore/bazel/format/ruff_isort.sh`); see
# https://docs.astral.sh/ruff/formatter/#sorting-imports for why import order
# is handled by `ruff check --select I` rather than by `ruff format`.

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

RUFF_BIN="$(rlocation multitool/tools/ruff/ruff)"
WORKSPACE_ROOT=$(bazel info workspace)

# We are called with either
# - check-mode: `format --check --force-exclude --diff FILES...`
# - fix-mode:   `format --force-exclude FILES...`
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

# With no explicit paths ruff falls back to scanning the cwd recursively, so
# bail out when nothing was collected rather than linting the whole repo.
if [[ "${#FILES[@]}" -eq 0 ]]; then
    echo "No files to format." >&2
    exit 0
fi

if [[ -n "$CHECK_ARG" ]]; then
    set -e
    $RUFF_BIN check --select I --force-exclude --diff "${FILES[@]}"
    $RUFF_BIN format --check --force-exclude --diff "${FILES[@]}"
else
    $RUFF_BIN check --select I --fix --force-exclude "${FILES[@]}"
    $RUFF_BIN format --force-exclude "${FILES[@]}"
fi
