#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Asserts the converter template (`ncore_template/`) stays in sync with the
# enclosing repository, which builds the rest of NuRec Onboard. Reports every
# mismatch before failing:
#
#   1. Toolchain -- the template's `.bazelversion`, Python minor version and
#      every `bazel_dep` it shares with the enclosing repository's
#      `MODULE.bazel` match it. The template is a standalone workspace copied
#      into converter workdirs, so it does not pick up the enclosing
#      repository's versions on its own and a bump there must be repeated here.
#   2. NCore pin -- `nvidia-ncore==<version>` in the template's
#      `deps/pip/requirements_3_13.in` matches the enclosing repository's pin,
#      which the shard validators run against, so converters write data with
#      the NCore version that validates it.
#   3. Shared pins -- every package locked in the template's
#      `deps/pip/requirements_3_13.txt` that the enclosing repository's x86_64
#      lock also pins is at or above the enclosing repository's version. The
#      template lock is what converter workdirs install, and it is not covered
#      by the enclosing repository's dependency scanning, so it must not trail
#      the pins that are.
#   4. NCore links -- every NCore GitHub source link in NuRec Onboard's tracked
#      files points at the release tag of that pin, so doc links don't drift
#      against `main`. A bump is a one-line requirements edit plus the `sed`
#      rewrite printed on failure. Line anchors (`#L123`) are rejected: line
#      numbers drift between releases and the rewrite cannot recompute them, so
#      name the symbol in the link text instead.
#
# Runnable both in CI and locally via
# `bazel run //skills/nurec-onboard:version_check`.

# `bazel run` starts the binary in the runfiles tree, whose entry symlinks back
# to the source; hop to the NuRec Onboard directory so `git grep` (which searches
# only below the current directory) and the relative paths resolve.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" || exit 1

fail=0
error() {
  echo "ERROR: $*" >&2
  fail=1
}

if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
  enclosing_root="$BUILD_WORKSPACE_DIRECTORY"
else
  enclosing_root="$PWD"
  while [[ "$enclosing_root" != / && ! -f "$enclosing_root/MODULE.bazel" ]]; do
    enclosing_root="$(dirname "$enclosing_root")"
  done
fi
enclosing_module="$enclosing_root/MODULE.bazel"
if [[ ! -f "$enclosing_module" ]]; then
  error "no MODULE.bazel found above $PWD"
  exit 1
fi

# --- 1. Toolchain -------------------------------------------------------------

bazel_version() { head -n1 "$1" | tr -d '[:space:]'; }
python_minor() { sed -nE 's/^[[:space:]]*python_version = "([0-9]+\.[0-9]+)",?$/\1/p' "$1" | head -n1; }
dep_names() { sed -nE 's/^[[:space:]]*bazel_dep\(name = "([^"]+)".*/\1/p' "$1"; }
dep_version() { sed -nE "s/^[[:space:]]*bazel_dep\(name = \"$2\", version = \"([^\"]+)\".*/\1/p" "$1" | head -n1; }

expected_bazel=$(bazel_version "$enclosing_root/.bazelversion")
expected_python=$(python_minor "$enclosing_module")
template=ncore_template
[[ "$(bazel_version "$template/.bazelversion")" == "$expected_bazel" ]] ||
  error "$template/.bazelversion: '$(bazel_version "$template/.bazelversion")' != '$expected_bazel' (enclosing repository)"
[[ "$(python_minor "$template/MODULE.bazel")" == "$expected_python" ]] ||
  error "$template/MODULE.bazel: Python '$(python_minor "$template/MODULE.bazel")' != '$expected_python' (enclosing repository)"
while IFS= read -r dep; do
  expected_dep=$(dep_version "$enclosing_module" "$dep")
  [[ -n "$expected_dep" ]] || continue
  actual_dep=$(dep_version "$template/MODULE.bazel" "$dep")
  [[ "$actual_dep" == "$expected_dep" ]] ||
    error "$template/MODULE.bazel: $dep '$actual_dep' != '$expected_dep' (enclosing repository)"
done < <(dep_names "$template/MODULE.bazel")
[[ "$fail" -eq 0 ]] && echo "OK: template toolchain matches the enclosing repository (Bazel $expected_bazel, Python $expected_python)"

# --- 2. NCore pin -------------------------------------------------------------

ncore_pin() { sed -nE 's/^nvidia-ncore==([^ ;#]+).*/\1/p' "$1" | head -n1; }
enclosing_requirements="$enclosing_root/deps/python/requirements_3_13_common.in"
version=$(ncore_pin "$enclosing_requirements")
template_version=$(ncore_pin "$template/deps/pip/requirements_3_13.in")
if [[ -z "$version" ]]; then
  error "no 'nvidia-ncore==' pin in $enclosing_requirements"
  exit 1
fi
if [[ "$version" != "$template_version" ]]; then
  error "nvidia-ncore version differs (enclosing repository='$version', template='$template_version')."
else
  echo "OK: nvidia-ncore pinned to $version in the template and the enclosing repository"
fi

# --- 3. Shared pins -----------------------------------------------------------

# Emits `name version` per `name==version` lock line, with the name normalized
# per PEP 503 and any local version label (`+cu128`) dropped.
locked_pins() {
  sed -nE 's/^([A-Za-z0-9._-]+)==([^ +;\\]+).*/\1 \2/p' "$1" |
    awk '{ name = tolower($1); gsub(/[-_.]+/, "-", name); print name, $2 }'
}
enclosing_lock="$enclosing_root/deps/python/requirements_3_13_x86_64.txt"
template_lock="$template/deps/pip/requirements_3_13.txt"
pins_ok=1
declare -A enclosing_pins
while read -r name pin; do
  enclosing_pins[$name]=$pin
done < <(locked_pins "$enclosing_lock")
while read -r name pin; do
  expected_pin=${enclosing_pins[$name]:-}
  [[ -n "$expected_pin" && "$pin" != "$expected_pin" ]] || continue
  # sort -V orders plain release versions as PEP 440 does.
  if [[ "$(printf '%s\n%s\n' "$pin" "$expected_pin" | sort -V | head -n1)" == "$pin" ]]; then
    error "$template_lock: $name $pin < $expected_pin (enclosing repository)"
    pins_ok=0
  fi
done < <(locked_pins "$template_lock")
if [[ "$pins_ok" -eq 1 ]]; then
  echo "OK: no package in the template lock trails the enclosing repository's pin"
else
  echo "Re-lock the template at or above those pins, e.g. from $template/:" >&2
  echo "  bazel run //deps/pip:requirements -- --upgrade-package <name>" >&2
fi

# --- 4. NCore links -----------------------------------------------------------

expected="v$version"
links_ok=1
# git grep -no emits file:line:URL for every NCore blob link in tracked files.
while IFS= read -r hit; do
  [[ -n "$hit" ]] || continue
  loc=${hit%%:https*}
  url=https${hit#*:https}
  if printf '%s' "$url" | grep -q '#L'; then
    error "$loc: line anchor not allowed (line numbers drift between releases): $url"
    links_ok=0
  else
    ref=$(printf '%s' "$url" | sed -E 's#.*/NVIDIA/ncore/blob/([^/]+)/.*#\1#')
    if [[ "$ref" != "$expected" ]]; then
      error "$loc: ref '$ref' != pinned '$expected': $url"
      links_ok=0
    fi
  fi
done < <(git grep -noE 'https://github\.com/NVIDIA/ncore/blob/[^ )"]*' || true)
if [[ "$links_ok" -eq 1 ]]; then
  echo "OK: all NCore source URLs pinned to $expected"
else
  echo "Fix the refs to $expected, e.g.:" >&2
  printf '%s\n' "  git grep -lE 'github\.com/NVIDIA/ncore/blob/' | xargs sed -i -E 's#(NVIDIA/ncore/blob/)[^/]+#\\1$expected#g'" >&2
fi

exit "$fail"
