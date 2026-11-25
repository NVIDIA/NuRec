#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uploads the artifact wheel to a PyPI registry, unless a file of that name is already there.
# Credentials are read from ~/.netrc. Usage
# (the first argument is set by Bazel):
#   publish_wheel.sh <wheel> <registry url>

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

set -euo pipefail

UV="$(rlocation multitool/tools/uv/uv)"
WHEEL="$(rlocation "$1")"
REGISTRY="$2"

# Skip the upload when the registry already lists a file of this name, whatever its content. A
# registry version is immutable, and uv --check-url would fail on a differing hash instead.
FILE="$(basename "$WHEEL")"
PROJECT="${FILE%%-*}"
if INDEX="$(curl -fsS --netrc "${REGISTRY}/simple/${PROJECT//_/-}/" 2>/dev/null)" && grep -qF -- ">${FILE}<" <<<"$INDEX"; then
  echo "${FILE} is already published, skipping"
  exit 0
fi

# Trusted publishing is a PyPI feature that uv may try by default, and which is not supported by
# all registries, bloating logs with errors.
exec "$UV" publish --trusted-publishing never --publish-url "$REGISTRY" "$WHEEL"
