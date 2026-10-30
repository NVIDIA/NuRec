#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this helper from Bash so nurec_dev is defined in your shell." >&2
    exit 1
fi

# Source this file after building NUREC_IMAGE on the release-matched
# NUREC_DEV_DEPS_IMAGE with Dockerfile-dev.build (see references/install.md).
# Matches references/install.md; does not pull images
# or run a container on source.
if [[ -z "${NUREC_SRC:-}" || ! -f "$NUREC_SRC/MODULE.bazel" || ! -f "$NUREC_SRC/run.py" ]]; then
    echo "Set NUREC_SRC to an absolute released NuRec checkout." >&2
    return 1
fi
if [[ "$NUREC_SRC" != /* || -z "${NUREC_IMAGE:-}" ]]; then
    echo "NUREC_SRC must be absolute; set NUREC_IMAGE to the built development image (see references/install.md)." >&2
    return 1
fi
export NUREC_SRC NUREC_IMAGE
export NUREC_OUTPUT_BASE="${NUREC_OUTPUT_BASE:-$HOME/.cache/nurec-bazel}"
if [[ "$NUREC_OUTPUT_BASE" != /* || "$NUREC_OUTPUT_BASE" == *$'\n'* ]]; then
    echo "NUREC_OUTPUT_BASE must be an absolute path without newlines." >&2
    return 1
fi
export NUREC_BUILD_HOME="$NUREC_OUTPUT_BASE/build-home"
export CCACHE_DIR="$NUREC_OUTPUT_BASE/ccache"
# Preserve extra same-path mounts when the helper is sourced again in this shell.
if ! declare -p NUREC_DOCKER_ARGS &>/dev/null; then NUREC_DOCKER_ARGS=(); fi
mkdir -p "$NUREC_BUILD_HOME" "$CCACHE_DIR" "$NUREC_SRC/outputs" || return 1
# Keep the generated rc separate from the user's own Bazel configuration.
{
    printf 'build --action_env="HOME=%s"\n' "$NUREC_BUILD_HOME"
    printf 'build --host_action_env="HOME=%s"\n' "$NUREC_BUILD_HOME"
    printf 'build --host_action_env="CCACHE_DIR=%s"\n' "$CCACHE_DIR"
    printf 'test --test_env=USER\n'
    printf 'build --sandbox_writable_path="%s"\n' "$NUREC_BUILD_HOME" "$CCACHE_DIR"
} > "$NUREC_OUTPUT_BASE/nurec.bazelrc" || return 1

nurec_dev() {
    if [[ $# == 0 ]]; then
        echo 'Usage: nurec_dev COMMAND [ARG ...]' >&2
        return 2
    fi
    if [[ "$1" == bazelisk ]]; then
        shift
        set -- bazelisk --output_base="$NUREC_OUTPUT_BASE" \
            --bazelrc="$NUREC_OUTPUT_BASE/nurec.bazelrc" "$@"
    fi
    local tty_args=()
    if [[ -t 0 && -t 1 ]]; then tty_args=(-t); fi
    docker run --rm -i "${tty_args[@]}" --gpus all --shm-size=64g --network host \
        --user "$(id -u):$(id -g)" \
        --env HOME="$NUREC_BUILD_HOME" --env USER="$(id -un)" \
        --env CCACHE_DIR --env HF_TOKEN --env CUDA_VISIBLE_DEVICES \
        --volume "$NUREC_SRC:$NUREC_SRC" \
        --volume "$NUREC_OUTPUT_BASE:$NUREC_OUTPUT_BASE" \
        --workdir "$NUREC_SRC" \
        "${NUREC_DOCKER_ARGS[@]}" \
        "$NUREC_IMAGE" "$@"
}
