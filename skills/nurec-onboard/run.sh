#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Exported for the `$NUREC_ONBOARD_DIR/...` paths the SKILL files reference.
NUREC_ONBOARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUREC_ONBOARD_DIR
# Root workspace that builds `//skills/nurec-onboard/...`.
NRE_ROOT="$NUREC_ONBOARD_DIR"
while [[ "$NRE_ROOT" != / && ! -f "$NRE_ROOT/MODULE.bazel" ]]; do
    NRE_ROOT="$(dirname "$NRE_ROOT")"
done
# The workflow lives at the bundle root, its steps under steps/.
BUNDLE_DIR="$NUREC_ONBOARD_DIR/nurec-onboard"
STEPS_DIR="$BUNDLE_DIR/steps"

# Runs a NuRec Onboard module (e.g. `lib.parse_input`); modules import as packages
# rooted at $NUREC_ONBOARD_DIR.
nurec_onboard_python() {
    PYTHONPATH="$NUREC_ONBOARD_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m "$@"
}

# Opt out of aspect_rules_lint telemetry (and other DO_NOT_TRACK-aware tools):
# each `bazel run //:format*` would POST to telemetry.aspect.build, and the
# sandbox's network prompt for it stalls the workflow.
export DO_NOT_TRACK=1

# ── Sandbox writable defaults ──
#
# Always-writable paths for both agents' sandboxes, plus NUREC_WORKDIR
# (applied by update_claude_config and build_codex_config_args). Each is
# resolved to its canonical path first: the sandbox checks writability against
# the resolved target, so a symlinked path would otherwise be read-only.
SANDBOX_DEFAULT_WRITABLE_PATHS=(
    "/tmp"                   # short-lived scratch
    "$HOME/.cache/bazel"     # Bazel repository and disk caches, pre-warmed by warm_bazel_cache
    "$HOME/.cache/uv"        # uv cache, for `bazel run //deps/pip:requirements` in converters
)

# ── Usage ──
usage() {
    echo "Usage: $0 <input.yaml> --agent <claude|codex> [options]"
    echo ""
    echo "Run the NuRec Onboard workflow end-to-end."
    echo ""
    echo "Arguments:"
    echo "  input.yaml    Path to the input YAML file"
    echo ""
    echo "Options:"
    echo "  --step <name>        Run a single step instead of the full workflow"
    echo "                       (one of the steps under steps/; see --list)"
    echo "  --agent <name>       Agent to use: claude or codex (required)"
    echo "  --model <name>       Model passed through to the selected agent"
    echo "                       (default: opus for Claude; configured default for Codex)"
    echo "  --preflight-only     Run pre-flight checks and exit (do not launch an agent)"
    echo "  --list               List the runnable steps"
    echo "  -h, --help           Show this help"
    echo "  --                   Pass every remaining argument to the selected"
    echo "                       agent's CLI. They are appended after the"
    echo "                       launcher's own arguments, so an argument that"
    echo "                       repeats one of those wins -- including the"
    echo "                       generated sandbox and permission settings"
    echo ""
    echo "Examples:"
    echo "  $0 path/to/input.yaml --agent claude"
    echo "  $0 path/to/input.yaml --agent codex --step audit-dataset"
    echo "  $0 path/to/input.yaml --agent claude --step build-ncore-converter --model sonnet"
    echo "  $0 path/to/input.yaml --agent claude -- --add-dir /scratch/shared"
    echo "  $0 path/to/input.yaml --agent codex -- --json"
    echo "  $0 --list"
    exit 1
}

# ── List steps ──
list_steps() {
    echo ""
    echo "Runnable steps (pass via --step <name>):"
    echo ""
    printf "  %-25s %-30s %s\n" "STEP" "AGENTS" "DESCRIPTION"
    printf "  %-25s %-30s %s\n" "--------" "------------------------------" "-----------"
    for manifest in "$STEPS_DIR"/*/manifest.yaml; do
        [[ -f "$manifest" ]] || continue
        dir=$(dirname "$manifest")
        name=$(basename "$dir")
        desc=""
        agents=""
        in_agents=false
        while IFS= read -r line; do
            if [[ "$line" =~ ^description:\ *(.*) ]]; then
                desc="${BASH_REMATCH[1]}"
            fi
            # Collect block-list agents (`agents:` then indented `- item` lines)
            # until the next top-level key; also handles the inline `agents: x` form.
            if [[ "$in_agents" == true ]]; then
                if [[ "$line" =~ ^[[:space:]]+-\ *(.+) ]]; then
                    agents="${agents:+$agents, }${BASH_REMATCH[1]}"
                    continue
                fi
                in_agents=false
            fi
            if [[ "$line" =~ ^agents:\ *(.*) ]]; then
                if [[ -n "${BASH_REMATCH[1]}" ]]; then
                    agents="${BASH_REMATCH[1]}"
                else
                    in_agents=true
                fi
            fi
        done < "$manifest"
        agents="${agents:-unspecified}"
        printf "  %-25s %-30s %s\n" "$name" "$agents" "$desc"
    done
    echo ""
    echo "With no --step, the full nurec-onboard workflow runs."
    echo ""
    exit 0
}

# ── Parse manifest ──
parse_manifest() {
    local manifest="$1"
    local field="$2"
    while IFS= read -r line; do
        if [[ "$line" =~ ^${field}:\ *(.*) ]]; then
            echo "${BASH_REMATCH[1]}"
            return
        fi
    done < "$manifest"
}

parse_manifest_prompt() {
    local manifest="$1"
    local in_prompt=false
    local prompt=""
    while IFS= read -r line; do
        if [[ "$in_prompt" == true ]]; then
            # End at the next unindented key; prompt lines may contain colons.
            if [[ ! "$line" =~ ^[[:space:]] && "$line" =~ ^[a-zA-Z0-9_-]+: ]]; then
                break
            fi
            local stripped="${line#  }"
            prompt+="${stripped}"$'\n'
        fi
        if [[ "$line" =~ ^prompt:\ *\| ]]; then
            in_prompt=true
        fi
    done < "$manifest"
    echo "$prompt"
}

# ── Handle --list and --help before positional args ──
for arg in "$@"; do
    case "$arg" in
        # Everything past the separator belongs to the agent's CLI, including
        # its own `--help`.
        --) break ;;
        --list) list_steps ;;
        -h|--help) usage ;;
    esac
done

if [[ $# -lt 1 ]]; then
    usage
fi

# ── Parse positional args ──
INPUT_YAML="$1"
shift 1

if [[ "$INPUT_YAML" == -* ]]; then
    echo "Error: expected <input.yaml> as the first argument, got option: $INPUT_YAML"
    usage
fi

# ── Parse optional flags ──
# STEP_NAME defaults to the whole workflow; --step selects one step. AGENT has
# no default: both agents run the workflow.
AGENT=""
STEP_NAME="nurec-onboard"
MODEL_OVERRIDE=""
PREFLIGHT_ONLY=false
AGENT_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            if [[ $# -lt 2 ]]; then
                echo "Error: --agent requires a value"
                usage
            fi
            AGENT="$2"
            shift 2
            ;;
        --step)
            if [[ $# -lt 2 ]]; then
                echo "Error: --step requires a value"
                usage
            fi
            if [[ "$2" == "nurec-onboard" ]]; then
                echo "Error: 'nurec-onboard' is the default workflow, not a step. Omit --step to run the full workflow."
                usage
            fi
            STEP_NAME="$2"
            shift 2
            ;;
        --model)
            if [[ $# -lt 2 ]]; then
                echo "Error: --model requires a value"
                usage
            fi
            MODEL_OVERRIDE="$2"
            shift 2
            ;;
        --preflight-only)
            PREFLIGHT_ONLY=true
            shift
            ;;
        --)
            shift
            AGENT_EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            echo "Error: Unknown option: $1"
            usage
            ;;
        *)
            echo "Error: Unexpected argument: $1"
            usage
            ;;
    esac
done

if [[ -z "$AGENT" ]]; then
    echo "Error: --agent is required (claude or codex)"
    usage
fi

case "$AGENT" in
    claude|codex) ;;
    *)
        echo "Error: Agent must be 'claude' or 'codex', got: $AGENT"
        usage
        ;;
esac

# ── Resolve target ──
if [[ "$STEP_NAME" == "nurec-onboard" ]]; then
    STEP_DIR="$BUNDLE_DIR"
else
    STEP_DIR="$STEPS_DIR/$STEP_NAME"
fi

if [[ ! -d "$STEP_DIR" ]]; then
    echo "Error: Step '$STEP_NAME' not found"
    echo ""
    echo "Available steps (for --step):"
    for d in "$STEPS_DIR"/*/; do
        [[ -f "$d/manifest.yaml" ]] && echo "  $(basename "$d")"
    done
    exit 1
fi

MANIFEST="$STEP_DIR/manifest.yaml"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: Step '$STEP_NAME' has no manifest.yaml"
    exit 1
fi

SKILL="$STEP_DIR/SKILL.md"

if [[ ! -f "$SKILL" ]]; then
    echo "Error: Step '$STEP_NAME' has no SKILL.md"
    exit 1
fi

if [[ ! -f "$INPUT_YAML" ]]; then
    echo "Error: Input file not found: $INPUT_YAML"
    exit 1
fi

# ── Read manifest ──
DESCRIPTION=$(parse_manifest "$MANIFEST" "description")

# ── Pre-flight checks ──
PREFLIGHT_FAIL=0

preflight_pass() { echo "  ✓ $1"; }
preflight_fail() { echo "  ✗ $1"; PREFLIGHT_FAIL=$((PREFLIGHT_FAIL + 1)); }

# State one pre-flight check establishes for a later one. Every check runs, even
# after the one feeding it failed, so these "not established" defaults make a
# consumer skip or fail rather than act on an unverified value.
DOCKER_AVAILABLE=false   # check_docker; gates the dev dependencies image pull and the GPU probe
CODEX_TRUSTED=false      # check_codex_setup; gates the Codex sandbox probes, which
                         # measure nothing until the checkout loads .codex
DATA_PATH=""             # check_input_yaml; feeds the path checks and, at launch,
                         # the generated permission overrides
NUREC_IMAGE=""           # check_input_yaml; the pull (or local build) and the container GPU probe
NUREC_DEV_BASE_IMAGE=""  # resolve_nurec_images; dev dependencies image the checkout's Dockerfile-dev.build extends
NUREC_SOURCE_DIR=""      # resolve_nurec_images; selected source checkout
NUREC_GIT_COMMON_DIR=""  # Linked worktree metadata may live outside the source mount
NUREC_BUILD_NETRC=""     # Only a development checkout uses host build credentials
NUREC_IMAGES_READY=false # check_nurec_images; the GPU probe runs the pulled image

# Cleanup runs on EXIT. Signal traps must exit: one that returned would resume
# the script and launch an agent with the lock already removed.
LAUNCHER_LOCK=""
MANAGED_BAZEL_CACHE_RC=""
BAZEL_WRAPPER_DIR=""
cleanup_on_exit() {
    [[ -z "$LAUNCHER_LOCK" ]] || rm -f "$LAUNCHER_LOCK"
    [[ -z "$MANAGED_BAZEL_CACHE_RC" ]] || rm -f "$MANAGED_BAZEL_CACHE_RC"
}
trap cleanup_on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Single-launcher lock via atomic O_EXCL open, holding the owner's PID to tell
# a live owner from a stale lock. The PID must be a positive integer, since
# `kill -0 -1` succeeds for any process the caller can signal.
check_launcher_state() {
    local lockfile="$NUREC_ONBOARD_DIR/.run.lock"
    if (set -o noclobber; echo "$$" > "$lockfile") 2>/dev/null; then
        LAUNCHER_LOCK="$lockfile"
        preflight_pass "Launcher state is clean"
        return
    fi
    local owner=""
    [[ -r "$lockfile" ]] && owner="$(cat "$lockfile" 2>/dev/null || true)"
    if [[ "$owner" =~ ^[1-9][0-9]*$ ]] && kill -0 "$owner" 2>/dev/null; then
        preflight_fail "Another run.sh launcher is active (PID $owner; lock: $lockfile)"
    else
        preflight_fail "Stale launcher lock at $lockfile (owner ${owner:-<empty>} not alive)"
        echo "      Remove and re-run:  rm '$lockfile'"
    fi
}

check_docker() {
    local docker_sock="/var/run/docker.sock"
    if [[ ! -S "$docker_sock" ]]; then
        preflight_fail "Docker socket not found at $docker_sock"
        return
    fi
    local docker_err
    docker_err="$(docker info 2>&1 >/dev/null || true)"
    if [[ -z "$docker_err" ]]; then
        DOCKER_AVAILABLE=true
        preflight_pass "Docker daemon is running"
    elif [[ "$docker_err" == *"permission denied"* ]]; then
        preflight_fail "Docker daemon is not reachable (permission denied on /var/run/docker.sock; check docker group membership)"
    else
        preflight_fail "Docker daemon is not reachable (is Docker installed and running?)"
    fi
}

resolve_nurec_images() {
    local resolved
    if ! resolved=$(nurec_onboard_python lib.runtime "$INPUT_YAML" "$NRE_ROOT" 2>&1); then
        preflight_fail "$resolved"
        return
    fi
    local fields
    mapfile -t fields <<< "$resolved"
    NUREC_SOURCE_DIR="${fields[0]}"
    NUREC_IMAGE="${fields[1]}"
    NUREC_BUILD_NETRC="${fields[2]:-}"
    NUREC_GIT_COMMON_DIR="${fields[3]:-}"
    # A development checkout runs in the development image it pins. A released checkout runs in a
    # development image built on the dev dependencies image selected by the skill.
    if [[ -z "$NUREC_BUILD_NETRC" ]]; then
        NUREC_DEV_BASE_IMAGE="$NUREC_IMAGE"
    fi
    if [[ -n "$NUREC_BUILD_NETRC" ]]; then
        preflight_pass "Build authentication: read-only $NUREC_BUILD_NETRC mount in the checkout-pinned development image"
    fi
    if [[ "$NUREC_SOURCE_DIR" == "${NUREC_WORKDIR:-}" ]]; then
        preflight_fail "NUREC_WORKDIR must differ from the source checkout (writable workdir, read-only source)"
    fi
    preflight_pass "NuRec source/image pair: $NUREC_SOURCE_DIR / $NUREC_IMAGE"
}

pull_nurec_image() {
    # Pull image $1 (labeled $2) and report pass/fail. Docker's progress goes to
    # a temp log shown only on failure; the "!" notice first keeps a long cold
    # pull from looking like a hang.
    local image="$1" label="$2" pull_log
    echo "  ! Pulling $label (docker pull $image); can take several minutes on a cold cache"
    pull_log="$(mktemp)"
    if docker pull "$image" >"$pull_log" 2>&1; then
        preflight_pass "$label is pullable: $image"
        rm -f "$pull_log"
        return 0
    fi
    preflight_fail "$label pull failed: $image"
    cat "$pull_log" >&2
    if [[ -n "$NUREC_BUILD_NETRC" ]]; then
        echo "  Authenticate Docker to the development image registry using the checkout README, then retry."
    fi
    rm -f "$pull_log"
    return 1
}

build_nurec_dev_image() {
    # Build the development image on the dev dependencies image $NUREC_DEV_BASE_IMAGE, then make
    # $NUREC_IMAGE its tag. The tag is per checkout, so runs on different checkouts cannot replace
    # each other's image. The build log is shown only on failure.
    local build_log tag
    if ! docker buildx version &>/dev/null; then
        preflight_fail "Docker Buildx is required to build the NuRec development image (docker buildx version failed)"
        return 1
    fi
    tag="nurec_dev:onboard-$(printf '%s' "$NUREC_SOURCE_DIR" | sha256sum | cut -c1-12)"
    echo "  ! Building the NuRec development image $tag on $NUREC_DEV_BASE_IMAGE; can take several minutes on a cold cache"
    build_log="$(mktemp)"
    if docker buildx build --network=host --build-arg "BASE_IMAGE=$NUREC_DEV_BASE_IMAGE" -t "$tag" \
        -f "$NUREC_SOURCE_DIR/deps/docker/Dockerfile-dev.build" --load "$NUREC_SOURCE_DIR/deps/docker" >"$build_log" 2>&1; then
        NUREC_IMAGE="$tag"
        preflight_pass "NuRec development image is built: $tag"
        rm -f "$build_log"
        return 0
    fi
    preflight_fail "NuRec development image build failed: $tag"
    cat "$build_log" >&2
    rm -f "$build_log"
    return 1
}

check_nurec_images() {
    [[ -n "$NUREC_IMAGE" && -n "$NUREC_SOURCE_DIR" ]] || return
    if [[ "$DOCKER_AVAILABLE" != true ]]; then
        echo "  ! Skipping NuRec image pull until Docker is available"
        return
    fi

    if [[ -n "$NUREC_DEV_BASE_IMAGE" ]]; then
        pull_nurec_image "$NUREC_DEV_BASE_IMAGE" "NuRec dev dependencies image" || return
        build_nurec_dev_image || return
    else
        pull_nurec_image "$NUREC_IMAGE" "NuRec development image" || return
    fi
    NUREC_IMAGES_READY=true
}

check_gpu() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        echo "  ! Host nvidia-smi diagnostic: available"
    else
        echo "  ! Host nvidia-smi diagnostic: unavailable (Docker GPU probe is authoritative)"
    fi

    # This only runs for requires_gpu, so a probe that cannot run fails.
    if [[ "$DOCKER_AVAILABLE" != true || "$NUREC_IMAGES_READY" != true ]]; then
        preflight_fail "Docker-scoped NVIDIA GPU probe could not run: it needs a running Docker daemon and a pulled NuRec image (see the Docker / NuRec image checks above)"
        return
    fi

    local attempt out
    for attempt in 1 2; do
        if out="$(docker run --rm --gpus all --entrypoint nvidia-smi "$NUREC_IMAGE" -L 2>&1)"; then
            preflight_pass "Docker-scoped NVIDIA GPU probe succeeded: ${out//$'\n'/; }"
            return
        fi
        echo "  ! Docker-scoped NVIDIA GPU probe attempt $attempt failed: $out"
    done
    preflight_fail "Docker-scoped NVIDIA GPU probe failed (docker run --rm --gpus all --entrypoint nvidia-smi $NUREC_IMAGE -L)"
}

check_bazel() {
    if command -v bazel &>/dev/null; then
        preflight_pass "Bazel is installed"
    else
        preflight_fail "Bazel is not installed (required to build and run the NCore converter)"
    fi
    if [[ -f "$NRE_ROOT/MODULE.bazel" ]]; then
        preflight_pass "Root Bazel workspace found at $NRE_ROOT"
    else
        preflight_fail "No MODULE.bazel found above $NUREC_ONBOARD_DIR (run.sh must stay inside the repository whose root workspace builds the validators)"
    fi
}

# Only the Codex checks read TOML; Claude's files are JSON.
check_python_parsers() {
    if ! python3 -c "import yaml" 2>/dev/null; then
        preflight_fail "PyYAML is not installed (required to parse the input YAML; run: pip install 'pyyaml>=6.0')"
        return
    fi
    if [[ "$AGENT" != "codex" ]]; then
        preflight_pass "PyYAML is installed"
        return
    fi
    if python3 -c "import tomllib" 2>/dev/null || python3 -c "import toml" 2>/dev/null; then
        preflight_pass "Python YAML and TOML parsers are installed"
    else
        preflight_fail "No Python TOML parser is available to read the Codex configuration (use Python 3.11+, or install 'toml>=0.10')"
    fi
}

# Print the git repository root that Claude Code and Codex record workspace
# trust under when started in the given directory: the main checkout's root
# for a linked worktree. Prints nothing outside a git repository.
resolve_git_trust_root() {
    local dir="$1" common_dir
    common_dir=$(git -C "$dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || return 0
    if [[ "$(basename "$common_dir")" == ".git" ]]; then
        dirname "$common_dir"
    else
        git -C "$dir" rev-parse --show-toplevel 2>/dev/null || true
    fi
}

check_claude_workspace_trust() {
    if ! command -v python3 &>/dev/null; then
        preflight_fail "Claude workspace trust could not be checked because python3 is not installed"
        return
    fi

    # Claude Code keys projects by physical path, so a symlinked checkout would
    # otherwise always look untrusted.
    local resolved_nurec_onboard_dir
    resolved_nurec_onboard_dir=$(realpath "$NUREC_ONBOARD_DIR" 2>/dev/null || echo "$NUREC_ONBOARD_DIR")

    # Started inside a git repository, Claude Code records trust for the
    # repository root, which then covers this checkout.
    local git_trust_root
    git_trust_root=$(resolve_git_trust_root "$resolved_nurec_onboard_dir")

    local claude_config="$HOME/.claude.json"
    local trust_state
    trust_state=$(python3 - "$claude_config" "$resolved_nurec_onboard_dir" "$git_trust_root" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
project_path = sys.argv[2]
git_root = sys.argv[3]

if not config_path.exists():
    print("missing_config")
    raise SystemExit(0)

# Read and parse separately to tell an unreadable file from invalid JSON;
# decoding bytes in json.loads reports non-UTF-8 as malformed.
try:
    raw = config_path.read_bytes()
except OSError as exc:
    print(f"unreadable_config: {exc}")
    raise SystemExit(0)

try:
    cfg = json.loads(raw)
except Exception as exc:
    print(f"malformed_config: {exc}")
    raise SystemExit(0)

# Wrong-shaped JSON still gets a remediation message, not a traceback.
if not isinstance(cfg, dict):
    print("malformed_config: top-level JSON value is not an object")
    raise SystemExit(0)

projects = cfg.get("projects", {})
if not isinstance(projects, dict):
    print("malformed_config: 'projects' is not a JSON object")
    raise SystemExit(0)

# An entry for the checkout itself wins over one for its repository root.
for candidate in (project_path, git_root):
    if not candidate or candidate not in projects:
        continue
    project_cfg = projects[candidate]
    if not isinstance(project_cfg, dict):
        print(f"malformed_config: projects[{candidate!r}] is not a JSON object")
        raise SystemExit(0)
    if "hasTrustDialogAccepted" not in project_cfg:
        continue
    if project_cfg["hasTrustDialogAccepted"] is not True:
        break
    print("trusted" if candidate == project_path else f"trusted_via_root:{candidate}")
    raise SystemExit(0)
print("untrusted")
PY
)

    local untrusted_scope="$resolved_nurec_onboard_dir"
    if [[ -n "$git_trust_root" && "$git_trust_root" != "$resolved_nurec_onboard_dir" ]]; then
        untrusted_scope="$resolved_nurec_onboard_dir or its repository root $git_trust_root"
    fi

    case "$trust_state" in
        trusted)
            preflight_pass "Claude workspace is trusted"
            ;;
        trusted_via_root:*)
            preflight_pass "Claude workspace is trusted (via repository root ${trust_state#trusted_via_root:})"
            ;;
        missing_config)
            preflight_fail "Claude workspace trust is not recorded for $resolved_nurec_onboard_dir (~/.claude.json not found)"
            echo "      Run 'claude' once in $resolved_nurec_onboard_dir and accept the workspace trust dialog, then re-run."
            echo "      If the dialog does not appear or does not persist, then with all Claude sessions closed set projects[\"$resolved_nurec_onboard_dir\"].hasTrustDialogAccepted: true in ~/.claude.json."
            ;;
        unreadable_config*)
            preflight_fail "Claude workspace trust could not be checked because ~/.claude.json is not readable (permission denied)"
            echo "      ${trust_state#unreadable_config: }"
            echo "      Fix the permissions on ~/.claude.json (or remove it with all Claude sessions closed), then run 'claude' once in $resolved_nurec_onboard_dir and accept the workspace trust dialog before re-running."
            ;;
        malformed_config*)
            preflight_fail "Claude workspace trust could not be checked because ~/.claude.json is malformed"
            echo "      ${trust_state#malformed_config: }"
            echo "      Repair or remove ~/.claude.json (with all Claude sessions closed), then run 'claude' once in $resolved_nurec_onboard_dir and accept the workspace trust dialog before re-running."
            ;;
        *)
            preflight_fail "Claude workspace trust is not recorded for $untrusted_scope"
            echo "      Run 'claude' once in $resolved_nurec_onboard_dir and accept the workspace trust dialog, then re-run."
            echo "      If the dialog does not appear or does not persist, then with all Claude sessions closed set projects[\"$resolved_nurec_onboard_dir\"].hasTrustDialogAccepted: true in ~/.claude.json."
            ;;
    esac
}

# Read paths the Codex CLI needs in its own sandbox: the runtime tree with the
# binary, and CODEX_HOME's argv[0] aliases, which include the sandbox helper.
codex_runtime_read_paths() {
    local entrypoint runtime_root codex_home codex_tmp
    entrypoint=$(realpath "$(command -v codex)")
    runtime_root=$(dirname "$entrypoint")
    if [[ "$entrypoint" == */bin/codex.js ]]; then
        runtime_root=$(dirname "$(dirname "$entrypoint")")
    fi
    codex_home="${CODEX_HOME:-$HOME/.codex}"
    codex_tmp="$codex_home/tmp"
    # Grant the resolved directory, which is what the sandbox reads.
    codex_tmp=$(realpath -m "$codex_tmp" 2>/dev/null || echo "$codex_tmp")
    printf '%s\n' "$runtime_root" "$codex_tmp"
}

# Inline TOML filesystem table giving a `codex sandbox` probe the runtime read
# grants an agent session gets. Grants only, like the tracked profile (see
# .codex/config.toml).
codex_probe_filesystem() {
    local table='{":minimal"="read"'
    local path quoted
    while IFS= read -r path; do
        # TOML-escape backslash, then double quote, so added escapes aren't re-escaped.
        quoted=${path//\\/\\\\}
        quoted=${quoted//\"/\\\"}
        table+=",\"$quoted\"=\"read\""
    done < <(codex_runtime_read_paths)
    printf '%s}' "$table"
}

check_codex_setup() {
    if ! command -v codex &>/dev/null; then
        preflight_fail "Codex CLI is not installed"
        echo "      Install Codex 0.138.0 or newer, then sign in and trust this checkout."
        return
    fi

    if ! command -v python3 &>/dev/null; then
        preflight_fail "Codex setup could not be checked because python3 is not installed"
        return
    fi

    local version_output version_state
    version_output=$(codex --version 2>/dev/null || true)
    version_state=$(python3 - "$version_output" <<'PY'
import re
import sys

match = re.search(r"(\d+)\.(\d+)\.(\d+)", sys.argv[1])
if match is None:
    print("unknown")
elif tuple(map(int, match.groups())) >= (0, 138, 0):
    print("supported")
else:
    print("outdated")
PY
)
    case "$version_state" in
        supported)
            preflight_pass "Codex CLI is supported ($version_output)"
            ;;
        outdated)
            preflight_fail "Codex CLI 0.138.0 or newer is required (found: $version_output)"
            ;;
        *)
            preflight_fail "Codex CLI version could not be determined (output: $version_output)"
            ;;
    esac

    local resolved_nurec_onboard_dir
    resolved_nurec_onboard_dir=$(realpath "$NUREC_ONBOARD_DIR" 2>/dev/null || echo "$NUREC_ONBOARD_DIR")

    # Trusting the enclosing repository's root covers a checkout nested below it.
    local git_trust_root
    git_trust_root=$(resolve_git_trust_root "$resolved_nurec_onboard_dir")

    local codex_home="${CODEX_HOME:-$HOME/.codex}"
    local user_config="$codex_home/config.toml"
    local project_config="$NUREC_ONBOARD_DIR/.codex/config.toml"
    local config_state
    config_state=$(python3 - "$user_config" "$project_config" "$resolved_nurec_onboard_dir" "$git_trust_root" <<'PY'
import sys
from pathlib import Path

user_path = Path(sys.argv[1])
project_path = Path(sys.argv[2])
project = sys.argv[3]
git_root = sys.argv[4]

try:
    import tomllib

    def parse(raw):
        return tomllib.loads(raw.decode())
except ModuleNotFoundError:
    import toml

    def parse(raw):
        return toml.loads(raw.decode())

def load(path):
    if not path.exists():
        return {}
    try:
        value = parse(path.read_bytes())
    except Exception as exc:
        print(f"malformed:{path}:{exc}")
        raise SystemExit(0)
    if not isinstance(value, dict):
        print(f"malformed:{path}:top-level TOML value is not a table")
        raise SystemExit(0)
    return value

user = load(user_path)
project_cfg = load(project_path)
system_path = Path("/etc/codex/config.toml")
system = load(system_path)

projects = user.get("projects", {})
if not isinstance(projects, dict):
    print(f"malformed:{user_path}:projects is not a table")
    raise SystemExit(0)
# An entry for the checkout itself wins over one for its repository root.
trusted_via = ""
for candidate in (project, git_root):
    entry = projects.get(candidate) if candidate else None
    if isinstance(entry, dict) and "trust_level" in entry:
        if entry["trust_level"] == "trusted":
            trusted_via = candidate
        break
trusted = bool(trusted_via)

legacy_paths = []
for path, cfg in (
    (system_path, system),
    (user_path, user),
    (project_path, project_cfg),
):
    if "sandbox_mode" in cfg or "sandbox_workspace_write" in cfg:
        legacy_paths.append(str(path))

print("trusted=" + ("true" if trusted else "false"))
print("trusted_via=" + trusted_via)
print("legacy=" + ",".join(legacy_paths))
PY
)

    if [[ "$config_state" == malformed:* ]]; then
        preflight_fail "Codex configuration could not be parsed"
        echo "      ${config_state#malformed:}"
        return
    fi

    local trusted="" trusted_via="" legacy=""
    while IFS='=' read -r key value; do
        case "$key" in
            trusted) trusted="$value" ;;
            trusted_via) trusted_via="$value" ;;
            legacy) legacy="$value" ;;
        esac
    done <<< "$config_state"

    if [[ "$trusted" == "true" ]]; then
        CODEX_TRUSTED=true
        if [[ "$trusted_via" == "$resolved_nurec_onboard_dir" ]]; then
            preflight_pass "Codex workspace is trusted"
        else
            preflight_pass "Codex workspace is trusted (via repository root $trusted_via)"
        fi
    elif [[ -n "$git_trust_root" && "$git_trust_root" != "$resolved_nurec_onboard_dir" ]]; then
        preflight_fail "Codex workspace trust is not recorded for $resolved_nurec_onboard_dir or its repository root $git_trust_root"
        echo "      Run 'codex' once in $resolved_nurec_onboard_dir and accept the workspace trust prompt."
    else
        preflight_fail "Codex workspace trust is not recorded for $resolved_nurec_onboard_dir"
        echo "      Run 'codex' once in $resolved_nurec_onboard_dir and accept the workspace trust prompt."
    fi

    if [[ -z "$legacy" ]]; then
        preflight_pass "Codex custom permission profile is not shadowed by legacy sandbox settings"
    else
        preflight_fail "Codex custom permission profile is shadowed by legacy sandbox settings"
        echo "      Remove sandbox_mode and sandbox_workspace_write from: $legacy"
    fi
}

# Verify the host can start the agents' sandbox: both use bubblewrap, and Claude
# also proxies network access through socat. This covers the host side only;
# the agent reports failures of the sandbox it actually builds.
check_sandbox_prerequisites() {
    if ! command -v bwrap &>/dev/null; then
        preflight_fail "bubblewrap (bwrap) is not installed; the $AGENT sandbox cannot start without it"
        echo "      Install it with your package manager (Debian/Ubuntu: sudo apt install bubblewrap)."
    elif bwrap --ro-bind / / --dev /dev --proc /proc true &>/dev/null; then
        preflight_pass "bubblewrap starts a sandbox on this host"
    else
        preflight_fail "bubblewrap is installed but cannot start a sandbox on this host"
        echo "      Enable unprivileged user namespaces, then retry."
    fi

    [[ "$AGENT" == "claude" ]] || return 0
    if command -v socat &>/dev/null; then
        preflight_pass "socat is installed for sandboxed network access"
    else
        preflight_fail "socat is not installed; the Claude sandbox cannot proxy network access"
        echo "      Install it with your package manager (Debian/Ubuntu: sudo apt install socat)."
    fi
}

# Fetch one allowlisted URL from inside the profile's sandbox.
#
# Codex's network proxy enforces `network.domains` and also resolves hostnames
# (sandboxed commands have no resolver), so without it every networked command
# fails with a name resolution error deep inside a build. `codex sandbox` runs
# the probe under the agent's policy without a model session. Skipped when the
# CLI or curl is missing, or without workspace trust (reported elsewhere).
check_codex_egress() {
    local probe_url="https://pypi.org/simple/"

    command -v codex &>/dev/null || return 0
    if [[ "$CODEX_TRUSTED" != true ]]; then
        echo "  ! Skipping the Codex egress probe until workspace trust is recorded"
        return
    fi
    if ! codex sandbox --help &>/dev/null; then
        echo "  ! Skipping the Codex egress probe (this Codex CLI has no 'sandbox' command)"
        return
    fi
    if ! command -v curl &>/dev/null; then
        echo "  ! Skipping the Codex egress probe (curl is not installed)"
        return
    fi

    # Codex finds project configuration from the working directory.
    if (cd "$NUREC_ONBOARD_DIR" && codex sandbox \
        -c 'default_permissions="nurec-onboard"' \
        -c "permissions.nurec-onboard.filesystem=$(codex_probe_filesystem)" \
        -- curl --silent --show-error --max-time 20 --output /dev/null "$probe_url") &>/dev/null; then
        preflight_pass "Codex sandbox reaches allowlisted domains"
    else
        preflight_fail "Codex sandbox cannot reach $probe_url (an allowlisted domain)"
        echo "      Domain rules and hostname resolution both depend on Codex's network proxy."
        echo "      Confirm 'features.network_proxy = true' is in effect (see .codex/config.README.md)."
    fi
}

# The sandboxed agent cannot reach the Docker socket (no /run), so
# container-backed steps rely on the `docker` rule resolving to `allow`, which
# runs the command on the host. A broken rule surfaces mid-workflow as a plain
# Docker connection error, so check it here. `codex sandbox` cannot: it always
# sandboxes.
check_codex_docker() {
    command -v codex &>/dev/null || return 0
    command -v docker &>/dev/null || return 0
    [[ "$CODEX_TRUSTED" == true ]] || return 0

    local rules="$NUREC_ONBOARD_DIR/.codex/rules/nurec-onboard.rules"
    if [[ ! -f "$rules" ]]; then
        preflight_fail "Codex Docker rule file is missing: $rules"
        return
    fi

    if ! codex execpolicy check --help &>/dev/null; then
        echo "  ! Skipping the Codex Docker rule check (this Codex CLI has no 'execpolicy' command)"
        return
    fi

    local decision
    if ! decision=$(cd "$NUREC_ONBOARD_DIR" && codex execpolicy check --rules "$rules" -- docker version 2>/dev/null | tail -1); then
        preflight_fail "'codex execpolicy check' failed on $rules"
        echo "      Run it to see the error: (cd $NUREC_ONBOARD_DIR && codex execpolicy check --rules $rules -- docker version)"
        return
    fi
    if [[ "$decision" != *'"decision":"allow"'* ]]; then
        preflight_fail "Docker commands are not allowed out of the Codex sandbox by $rules"
        echo "      Container-backed steps cannot reach the daemon without that rule."
        return
    fi

    # A denied read in any merged configuration layer makes Codex drop the
    # rule's escalation, since it is only enforceable inside the sandbox.
    local denied
    denied=$(python3 - "${CODEX_HOME:-$HOME/.codex}/config.toml" "$NUREC_ONBOARD_DIR/.codex/config.toml" <<'PY'
import sys
from pathlib import Path

try:
    import tomllib

    def parse(raw):
        return tomllib.loads(raw.decode())
except ModuleNotFoundError:
    import toml

    def parse(raw):
        return toml.loads(raw.decode())

denied = []
for candidate in [*sys.argv[1:], "/etc/codex/config.toml"]:
    path = Path(candidate)
    if not path.exists():
        continue
    try:
        config = parse(path.read_bytes())
    except Exception:
        continue
    profile = config.get("permissions", {}).get("nurec-onboard", {})
    entries = profile.get("filesystem", {})
    if not isinstance(entries, dict):
        continue
    denied += [f"{path}:{key}" for key, value in entries.items() if value == "deny"]

print(",".join(sorted(denied)))
PY
)
    if [[ -n "$denied" ]]; then
        preflight_fail "Codex permission profile denies reads ($denied), which blocks Docker"
        echo "      A deny entry suppresses the sandbox escalation the docker rule grants."
        return
    fi

    preflight_pass "Codex runs Docker commands on the host (execpolicy allow rule)"
}

check_input_yaml() {
    # Single authoritative parse of the input YAML. parse_input.py validates the
    # whole document (security-sensitive fields included, see _FIELD_VALIDATORS)
    # and prints the requested fields on stdout, one per line. Stdout and stderr
    # are captured separately so a stray warning cannot shift the fields; stderr
    # is shown only on failure.
    local __out __err __err_file
    __err_file="$(mktemp)"
    if __out=$(nurec_onboard_python lib.parse_input "$INPUT_YAML" \
        "data.path" 2>"$__err_file"); then
        local __fields
        mapfile -t __fields <<< "$__out"
        DATA_PATH="${__fields[0]:-}"
        preflight_pass "Input YAML is well-formed"
    else
        __err="$(cat "$__err_file")"
        preflight_fail "Input YAML failed validation: $__err"
    fi
    rm -f "$__err_file"
}

is_sensitive_root() {
    # Match roots whose wholesale exposure is almost certainly an operator typo.
    local canonical="$1"
    local home_canonical
    home_canonical="$(realpath -m -- "$HOME" 2>/dev/null || echo "$HOME")"
    local forbidden=("/" "$home_canonical" "/root" "/etc" "/usr" "/var")
    local path
    for path in "${forbidden[@]}"; do
        [[ "$canonical" == "$path" ]] && return 0
    done
    return 1
}

check_workdir_safety() {
    # Reject a NUREC_WORKDIR that would make a sensitive root writable: an exact
    # match, after resolving symlinks, on /, $HOME, /root, /etc, /usr or /var
    # (typo protection; subdirectories are fine). NUREC_WORKDIR_ALLOW_ANY=1
    # overrides. /tmp is allowed with a warning. Returns 1 after preflight_fail
    # on rejection.
    local workdir="$1"

    local canonical
    canonical="$(realpath -m -- "$workdir" 2>/dev/null || echo "$workdir")"

    if is_sensitive_root "$canonical"; then
        if [[ "${NUREC_WORKDIR_ALLOW_ANY:-}" == "1" ]]; then
            echo "  ! NUREC_WORKDIR resolves to a sensitive root ($canonical) -- allowed via NUREC_WORKDIR_ALLOW_ANY=1"
            return 0
        fi
        preflight_fail "NUREC_WORKDIR resolves to a sensitive root: $canonical (set NUREC_WORKDIR_ALLOW_ANY=1 to override; see .claude/settings.local.README.md)"
        return 1
    fi

    if [[ "$canonical" == "/tmp" ]]; then
        echo "  ! NUREC_WORKDIR is /tmp -- the sandbox already grants /tmp; consider a per-run subdirectory instead"
    fi
    return 0
}

check_input_docs() {
    # Codex reads every local data.docs entry, so the only rejection left is an
    # entry that resolves to a sensitive root and would expose its whole subtree.
    [[ "$AGENT" == "codex" ]] || return 0
    [[ -n "$DATA_PATH" ]] || return 0

    local command=(
        nurec_onboard_python lib.codex_config
        --check-docs
        --repo-root "$NUREC_ONBOARD_DIR"
        --input "$INPUT_YAML"
        --data-path "$DATA_PATH"
        --workdir "${NUREC_WORKDIR:-}"
    )

    local status doc_path
    local sensitive=0
    local docs_status
    if ! docs_status=$("${command[@]}"); then
        preflight_fail "Local data.docs paths could not be checked"
        return
    fi
    while IFS=$'\t' read -r status doc_path; do
        case "$status" in
            sensitive)
                preflight_fail "data.docs entry resolves to a sensitive root: $doc_path"
                sensitive=$((sensitive + 1))
                ;;
        esac
    done <<< "$docs_status"

    if (( sensitive == 0 )); then
        preflight_pass "Local data.docs paths are readable by Codex"
    fi
}

check_input_paths() {
    # data.path is bound by check_input_yaml's single parse.
    local data_path="$DATA_PATH"

    if [[ -z "$data_path" ]]; then
        preflight_fail "data.path not specified in input YAML"
    # YAML does not expand a leading "~"; flag it.
    elif [[ "${data_path:0:1}" == "~" ]]; then
        preflight_fail "data.path uses '~', which is not expanded inside YAML; use an absolute path instead: $data_path"
    elif [[ -d "$data_path" ]]; then
        DATA_PATH="$(realpath -- "$data_path")"
        preflight_pass "Data path exists: $DATA_PATH"
    elif [[ -e "$data_path" ]]; then
        preflight_fail "data.path must be a directory, but exists as a file: $data_path"
    else
        preflight_fail "Data path not found: $data_path"
    fi

    local workdir="${NUREC_WORKDIR:-}"
    if [[ -z "$workdir" ]]; then
        preflight_fail "NUREC_WORKDIR is not set"
    # The launcher does not expand a leading "~"; flag it.
    elif [[ "${workdir:0:1}" == "~" ]]; then
        preflight_fail "NUREC_WORKDIR uses '~', which the launcher does not expand; use an absolute path instead: $workdir"
    elif [[ ! -d "$workdir" ]]; then
        preflight_fail "NUREC_WORKDIR not found: $workdir"
    elif check_workdir_safety "$workdir"; then
        if [[ ! -w "$workdir" ]]; then
            preflight_fail "NUREC_WORKDIR not writable: $workdir"
        else
            NUREC_WORKDIR="$(realpath -- "$workdir")"
            export NUREC_WORKDIR
            preflight_pass "NUREC_WORKDIR exists and writable: $NUREC_WORKDIR"
        fi
    fi
    if [[ -n "${NUREC_WORKDIR:-}" && "$NUREC_WORKDIR" == "$DATA_PATH" ]]; then
        preflight_fail "NUREC_WORKDIR must differ from data.path (writable workdir, read-only raw data)"
    fi

}

REQUIRES_DOCKER=$(parse_manifest "$MANIFEST" "requires_docker")
REQUIRES_GPU=$(parse_manifest "$MANIFEST" "requires_gpu")
REQUIRES_BAZEL=$(parse_manifest "$MANIFEST" "requires_bazel")

# check_gpu probes inside the pulled development image.
NEED_NUREC_IMAGES=false
[[ "$REQUIRES_DOCKER" == "true" ]] && NEED_NUREC_IMAGES=true

echo ""
echo "Pre-flight checks:"

check_launcher_state
check_python_parsers
check_sandbox_prerequisites
if [[ "$AGENT" == "claude" ]]; then
    check_claude_workspace_trust
else
    check_codex_setup
    check_codex_egress
fi
check_input_yaml
check_input_paths
check_input_docs
if [[ "$REQUIRES_DOCKER" == "true" ]]; then
    check_docker
    if [[ "$AGENT" == "codex" ]]; then check_codex_docker; fi
fi
if [[ "$NEED_NUREC_IMAGES" == true ]]; then
    resolve_nurec_images
    check_nurec_images
fi
if [[ "$REQUIRES_GPU" == "true" ]]; then check_gpu; fi
if [[ "$REQUIRES_BAZEL" == "true" ]]; then check_bazel; fi

echo ""

if (( PREFLIGHT_FAIL > 0 )); then
    echo "Pre-flight failed -- fix the above issues and retry."
    exit 1
fi

if [[ "$PREFLIGHT_ONLY" == true ]]; then
    echo "Pre-flight passed."
    exit 0
fi

prepare_nurec_cache() {
    # Source builds run as the host user with the checkout mounted read-only.
    [[ -n "${NUREC_WORKDIR:-}" && -d "$NUREC_WORKDIR" && -w "$NUREC_WORKDIR" ]] || return
    is_sensitive_root "$(realpath -m -- "$NUREC_WORKDIR")" && return
    local cache="$NUREC_WORKDIR/nurec-bazel"
    mkdir -p "$cache/build-home" "$cache/ccache"
    {
        printf 'build --action_env="HOME=%s"\n' "$cache/build-home"
        printf 'build --host_action_env="HOME=%s"\n' "$cache/build-home"
        printf 'build --host_action_env="CCACHE_DIR=%s"\n' "$cache/ccache"
        printf 'test --test_env=USER\n'
        printf 'build --sandbox_writable_path="%s"\n' "$cache/build-home" "$cache/ccache"
        printf 'build --symlink_prefix="%s/bazel-"\n' "$cache"
    } > "$cache/nurec.bazelrc"
}

if [[ "$NEED_NUREC_IMAGES" == true ]]; then prepare_nurec_cache; fi

# ── Generate per-run Claude settings ──
#
# Regenerated from the committed template on every launch and git-ignored, so
# host paths never reach the tracked file.
update_claude_config() {
    local template="$NUREC_ONBOARD_DIR/.claude/settings.local.template.json"
    local settings="$NUREC_ONBOARD_DIR/.claude/settings.local.json"
    if [[ ! -f "$template" ]]; then
        echo ""
        echo "Error: $template not found; cannot generate Claude settings."
        exit 1
    fi

    local command=(
        nurec_onboard_python lib.claude_config
        --template "$template"
        --settings "$settings"
        --input "$INPUT_YAML"
    )
    local path
    for path in "${SANDBOX_DEFAULT_WRITABLE_PATHS[@]}" "${NUREC_WORKDIR:-}"; do
        [[ -n "$path" ]] || continue
        command+=(--write-path "$path")
    done
    "${command[@]}"
}

if [[ "$AGENT" == "claude" ]]; then
    update_claude_config
fi

# ── Verify sandbox settings are correct ──
#
# Runs before warm_bazel_cache, the first unsandboxed network action. Fails on
# each way the sandbox can silently vanish:
#   1. sandbox.enabled is false.
#   2. sandbox.failIfUnavailable is false, so a sandbox that cannot start on
#      this host falls back to none.
#   3. sandbox.allowUnsandboxedCommands is true (the default), letting Claude
#      retry outside the sandbox (`docker *` in excludedCommands is the one
#      explicit exception).
verify_sandbox_enabled() {
    local settings="$NUREC_ONBOARD_DIR/.claude/settings.local.json"
    if [[ ! -f "$settings" ]]; then
        echo ""
        echo "Error: $settings not found after update_claude_config"
        exit 1
    fi

    local state
    state=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    sandbox = json.load(f).get('sandbox', {})
print('enabled=' + ('true' if sandbox.get('enabled') is True else 'false'))
print('failIfUnavailable=' + ('true' if sandbox.get('failIfUnavailable') is True else 'false'))
print('allowUnsandboxedCommands=' + ('true' if sandbox.get('allowUnsandboxedCommands') is not False else 'false'))
" "$settings")

    local enabled="" fail_if_unavailable="" allow_unsandboxed=""
    while IFS='=' read -r key value; do
        case "$key" in
            enabled)                  enabled="$value" ;;
            failIfUnavailable)        fail_if_unavailable="$value" ;;
            allowUnsandboxedCommands) allow_unsandboxed="$value" ;;
        esac
    done <<< "$state"

    if [[ "$enabled" != "true" ]]; then
        echo ""
        echo "Error: sandbox.enabled is false in $settings; refusing to launch."
        echo "Re-enable the sandbox (sandbox.enabled: true) in .claude/settings.local.template.json before running."
        echo "See .claude/settings.local.README.md for the approved configuration."
        exit 1
    fi

    if [[ "$fail_if_unavailable" != "true" ]]; then
        echo ""
        echo "Error: sandbox.failIfUnavailable is not true in $settings; refusing to launch."
        echo "Without it, Claude Code falls back to unsandboxed execution if the sandbox cannot start."
        echo "Set sandbox.failIfUnavailable: true in .claude/settings.local.template.json and re-run."
        exit 1
    fi

    if [[ "$allow_unsandboxed" != "false" ]]; then
        echo ""
        echo "Error: sandbox.allowUnsandboxedCommands is not false in $settings; refusing to launch."
        echo "Without it, Claude Code can retry failed sandboxed commands outside the sandbox."
        echo "Set sandbox.allowUnsandboxedCommands: false in .claude/settings.local.template.json and re-run."
        exit 1
    fi
}

if [[ "$AGENT" == "claude" ]]; then
    verify_sandbox_enabled
fi

# ── Bazel routing for the agent ──
#
# Root-workspace validation uses literal Docker commands, because an agent's
# shell wrapper cannot inherit Docker's host-execution permission. The host
# wrapper below rejects root calls and keeps converter cache routing only.
setup_nurec_commands() {
    local cache="$NUREC_WORKDIR/nurec-bazel"
    NUREC_DOCKER=(docker run --rm --name "nurec-onboard-validate-$$"
        --user "$(id -u):$(id -g)"
        -e "HOME=$cache/build-home" -e "USER=$(id -un)" -e "CCACHE_DIR=$cache/ccache"
        -v "$NUREC_SOURCE_DIR:$NUREC_SOURCE_DIR:ro"
        -v "$NUREC_WORKDIR:$NUREC_WORKDIR"
        -w "$NUREC_SOURCE_DIR")
    if [[ "$DATA_PATH" != "$NUREC_SOURCE_DIR" ]]; then
        NUREC_DOCKER+=(-v "$DATA_PATH:$DATA_PATH:ro")
    fi
    if [[ "$NUREC_ONBOARD_DIR" != "$NUREC_SOURCE_DIR/skills/nurec-onboard" ]]; then
        NUREC_DOCKER+=(-v "$NUREC_ONBOARD_DIR:$NUREC_SOURCE_DIR/skills/nurec-onboard:ro")
    fi
    local build_mounts=()
    if [[ -n "$NUREC_BUILD_NETRC" ]]; then
        build_mounts+=(-v "$NUREC_BUILD_NETRC:$cache/build-home/.netrc:ro")
    fi
    if [[ -n "$NUREC_GIT_COMMON_DIR" ]]; then
        build_mounts+=(-v "$NUREC_GIT_COMMON_DIR:$NUREC_GIT_COMMON_DIR:ro")
    fi
    NUREC_DOCKER+=("${build_mounts[@]}")
    NUREC_DOCKER+=("$NUREC_IMAGE" bazelisk
        "--output_base=$cache/output_base" "--bazelrc=$cache/nurec.bazelrc")
    # The launcher waits in the foreground; agent-managed builds detach and keep logs.
    local agent_docker=(docker run -d "${NUREC_DOCKER[@]:3}")
    RUNTIME_COMMANDS="$NUREC_WORKDIR/nurec-runtime.md"
    {
        echo '# NuRec source commands for this run'
        echo "Source: $NUREC_SOURCE_DIR"
        echo "Development image: $NUREC_IMAGE"
        if (( ${#build_mounts[@]} )); then
            printf 'Additional build mounts (include in auxiliary-generation and reconstruction commands): '
            printf '%q ' "${build_mounts[@]}"
            printf '\n'
            echo 'These read-only mounts supply build authentication and/or linked-worktree metadata. Do not read, copy or print credential contents.'
        else
            echo 'Additional build mounts: none; omit the placeholder from application commands.'
        fi
        echo 'For ALL root-workspace Bazel calls in the skill (validators, NCore validate/eval, version checks), replace cd/bazel and any startup flags with this literal prefix:'
        printf '\n```bash\n'
        printf '%q ' "${agent_docker[@]}"
        printf '\n```\n\n'
        echo 'Append the documented Bazel command, target and arguments. Run Docker as one standalone command with literal paths; do not wrap it in a shell function, pipeline or cd command. Add same-path read-only mounts for any input stores outside the raw-data directory or workdir.'
        echo 'Poll the named container with separate docker inspect calls until it exits, then read docker logs and require State.ExitCode=0 before proceeding. Remove the stopped container with docker rm before reusing this prefix; a successful detached launch alone does not mean validation passed.'
        echo 'Example: append run //skills/nurec-onboard/validators:audit_dataset_validator -- INPUT REPORT PROFILE, using absolute artifact paths.'
        echo 'The prefix starts at the SOURCE ROOT, not the skill directory. Use root target labels unchanged. NuRec source/config reads use the Source path above; reconstructed workloads use the same source/image and cache as these validators.'
        echo 'Converter-workspace Bazel commands still run on the host. Do not copy dependency archives into host caches or use --override_module to work around root resolution.'
    } > "$RUNTIME_COMMANDS"
}

setup_nurec_commands

setup_bazel_wrapper() {
    local workdir="${NUREC_WORKDIR:-}"
    [[ "$REQUIRES_BAZEL" == "true" ]] || return 0
    [[ -n "$workdir" ]] || return 0

    local rc_flag=""
    if [[ "$AGENT" == "codex" ]]; then
        local cache_target="$NUREC_ONBOARD_DIR/.bazelrc.user.nurec-onboard-cache"
        local cache_root
        cache_root=$(realpath -m -- "$HOME/.cache/bazel" 2>/dev/null || echo "$HOME/.cache/bazel")
        # Register with the cleanup trap before writing, so an interrupted
        # write is still removed.
        MANAGED_BAZEL_CACHE_RC="$cache_target"
        cat > "$cache_target" <<EOF
# Managed by run.sh -- do not edit. See README.md "Bazel output routing".
startup --output_user_root="$workdir/.bazel_roots"
build --repository_cache="$cache_root/repository-cache"
build --disk_cache="$cache_root/disk-cache/nurec-onboard"
EOF
        rc_flag=$(printf -- '--bazelrc=%q' "$cache_target")
    fi

    BAZEL_WRAPPER_DIR="$workdir/.nurec-bin"
    local real_bazel root
    real_bazel=$(realpath "$(command -v bazel)")
    root=$(cd "$NRE_ROOT" && pwd -P)
    mkdir -p "$BAZEL_WRAPPER_DIR"
    cat > "$BAZEL_WRAPPER_DIR/bazel" <<EOF
#!/usr/bin/env bash
# Managed by run.sh -- do not edit. See README.md "Bazel output routing".
startup=($rc_flag)
workspace=\$(pwd -P)
while [[ "\$workspace" != / && ! -e "\$workspace/MODULE.bazel" ]]; do
    workspace=\$(dirname "\$workspace")
done
if [[ "\$workspace" == $(printf '%q' "$root") || "\$workspace" == $(printf '%q' "$NUREC_SOURCE_DIR") ]]; then
    echo "Root-workspace Bazel runs in the selected build image. Use the literal Docker prefix in $(printf '%q' "$RUNTIME_COMMANDS")." >&2
    exit 1
fi
exec $(printf '%q' "$real_bazel") "\${startup[@]}" "\$@"
EOF
    chmod 0755 "$BAZEL_WRAPPER_DIR/bazel"
}

setup_bazel_wrapper

# ── Pre-warm the Bazel repository cache ──
#
# Resolve source dependencies in the image before agent launch. Converter
# dependencies remain in the standalone host workspace and its existing cache.
warm_bazel_cache() {
    [[ "$REQUIRES_BAZEL" == "true" ]] || return 0
    local template_dir="$NUREC_ONBOARD_DIR/ncore_template"
    echo ""
    echo "Warming source Bazel dependencies in the matching image..."
    if ! "${NUREC_DOCKER[@]}" fetch //skills/nurec-onboard/...; then
        echo "Error: source dependency resolution failed; stopping before agent launch."
        return 1
    fi
    echo "Warming Bazel repository cache (converter template)..."
    if (cd "$template_dir" && bazel fetch //... 2>&1 | tail -10); then
        echo "  Template-workspace Bazel cache warmed."
    else
        echo "  WARNING: bazel fetch from $template_dir failed; the converter workspace's first sandboxed bazel run may hit network errors."
    fi
}

warm_bazel_cache

# Add the wrapper to PATH only after the warm, which reuses the template's
# extracted output base; the Codex rc would send it to this run's empty output
# root and re-extract every external repository.
[[ -z "$BAZEL_WRAPPER_DIR" ]] || export PATH="$BAZEL_WRAPPER_DIR:$PATH"

# ── Build Codex per-run permissions ──
#
# The tracked Codex profile has no host paths; each launch gets exact TOML
# overrides for the access it needs.
CODEX_CONFIG_ARGS=()
build_codex_config_args() {
    [[ "$AGENT" == "codex" ]] || return 0

    local bazelisk_cache
    local codex_runtime_paths=()
    mapfile -t codex_runtime_paths < <(codex_runtime_read_paths)
    if [[ -n "${BAZELISK_HOME:-}" ]]; then
        bazelisk_cache="$BAZELISK_HOME"
    elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
        bazelisk_cache="$XDG_CACHE_HOME/bazelisk"
    else
        bazelisk_cache="$HOME/.cache/bazelisk"
    fi

    local overrides
    local command=(
        nurec_onboard_python lib.codex_config
        --repo-root "$NUREC_ONBOARD_DIR"
        --input "$INPUT_YAML"
        --data-path "$DATA_PATH"
        --workdir "${NUREC_WORKDIR:-}"
        --read-path "$bazelisk_cache"
        # The root workspace's MODULE.bazel, .bazelrc and Starlark, which
        # every sandboxed Bazel call on the NuRec Onboard targets loads.
        --read-path "$NRE_ROOT"
        --read-path "$NUREC_SOURCE_DIR"
    )
    local path
    for path in "${codex_runtime_paths[@]}"; do
        command+=(--read-path "$path")
    done
    if [[ "$REQUIRES_DOCKER" == "true" ]]; then
        command+=(--requires-docker)
    fi
    for path in "${SANDBOX_DEFAULT_WRITABLE_PATHS[@]}"; do
        command+=(--write-path "$path")
    done

    if ! overrides=$("${command[@]}"); then
        echo "Error: failed to build Codex permission overrides"
        exit 1
    fi

    CODEX_CONFIG_ARGS=(
        -c 'approval_policy="never"'
        -c 'default_permissions="nurec-onboard"'
    )
    local override
    while IFS= read -r override; do
        [[ -n "$override" ]] || continue
        CODEX_CONFIG_ARGS+=(-c "$override")
    done <<< "$overrides"
}

build_codex_config_args

# ── Resolve model ──
if [[ -n "$MODEL_OVERRIDE" ]]; then
    MODEL="$MODEL_OVERRIDE"
elif [[ "$AGENT" == "claude" ]]; then
    MODEL="opus"
else
    MODEL=""
fi

# ── Build prompt ──
PROMPT_TEMPLATE=$(parse_manifest_prompt "$MANIFEST")
INPUT_CONTENT=$(cat "$INPUT_YAML")

PROMPT="${PROMPT_TEMPLATE//\{\{skill\}\}/$SKILL}"
PROMPT="${PROMPT//\{\{input\}\}/$INPUT_CONTENT}"
PROMPT+=$'\n\n'"Before running commands, read $RUNTIME_COMMANDS. Its root-workspace execution instructions replace the host Bazel prefixes in all referenced skill files."

# ── Launch ──
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "=========================================="
echo "  $STEP_NAME"
echo "=========================================="
echo "  $DESCRIPTION"
echo "------------------------------------------"
echo "  Started:  $TIMESTAMP"
echo "  Agent:    $AGENT"
echo "  Model:    ${MODEL:-configured default}"
echo "  Input:    $INPUT_YAML"
echo "  Skill:    $SKILL"
if (( ${#AGENT_EXTRA_ARGS[@]} > 0 )); then
    echo "  Args:     ${AGENT_EXTRA_ARGS[*]}"
    echo "------------------------------------------"
    echo "  ! These arguments follow the generated ones and can override them,"
    echo "    the sandbox and permission settings included."
fi
echo "=========================================="
echo ""

if [[ "$AGENT" == "claude" ]]; then
    # Claude Code loads its project (settings, CLAUDE.md, trust) from the
    # working directory. Extra arguments follow the prompt so a variadic option
    # like `--add-dir` cannot swallow it.
    (cd "$NUREC_ONBOARD_DIR" && claude --model "$MODEL" "$PROMPT" "${AGENT_EXTRA_ARGS[@]}")
else
    CODEX_ARGS=(
        exec
        --strict-config
        -C "$NUREC_ONBOARD_DIR"
        "${CODEX_CONFIG_ARGS[@]}"
    )
    if [[ -n "$MODEL" ]]; then
        CODEX_ARGS+=(--model "$MODEL")
    fi
    # Last, so an operator's argument overrides the generated one it repeats.
    if (( ${#AGENT_EXTRA_ARGS[@]} > 0 )); then
        CODEX_ARGS+=("${AGENT_EXTRA_ARGS[@]}")
    fi
    # Unpiped so Codex owns the terminal (colors, `--color auto`); the trailing
    # `-` reads the prompt from stdin.
    printf '%s' "$PROMPT" | codex "${CODEX_ARGS[@]}" -
fi
