<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Source setup

For software prerequisites, see
[Software Setup](https://docs.nvidia.com/nurec/basics/software.html).
Hosted NuRec links track the latest published docs and may differ from this
checkout.
Use a released NuRec checkout and the matching
`ghcr.io/nvidia/nurec/nurec_dev_deps` tag/digest published for that release.
Its tag is the checkout's `VERSION_MAJOR.VERSION_MINOR` from
`bazel/version/VERSION_FILE`, formatted as `YY.MM`.
Set `NUREC_DEV_DEPS_IMAGE` to that dependencies image. It supplies archives and
wheels under `/opt/nvidia/nurec-build-deps`. Build the development tools on top
using the checkout's `deps/docker/Dockerfile-dev.build`, then set `NUREC_IMAGE`
to the resulting development image. All `nurec_dev` commands use `NUREC_IMAGE`.
The development image builds the applications from source; it does not contain
a ready-to-run NuRec CLI. Public dependencies may still download on the first
build. A missing release image is a publication problem: report the
exact image and pull error rather than substituting another registry or runtime.

Docker's Buildx plugin is required (`docker buildx version`). The selected
checkout must include the Dockerfile's `BASE_IMAGE` argument. If it does not,
use a matching source release that supports this setup before proceeding.

From the host's Bash shell, replace `<release>` with your release branch or tag
and `<tag>` with the checkout's `YY.MM` dependency-image tag:

```bash
git clone --branch <release> https://github.com/NVIDIA/NuRec.git
cd NuRec
export NUREC_SRC="$PWD"
export NUREC_DEV_DEPS_IMAGE=ghcr.io/nvidia/nurec/nurec_dev_deps:<tag>
export NUREC_IMAGE=nurec_dev:local
docker buildx version
grep -q '^ARG BASE_IMAGE' "$NUREC_SRC/deps/docker/Dockerfile-dev.build" || {
  echo 'This checkout needs the BASE_IMAGE development Dockerfile.' >&2
  exit 1
}
docker pull "$NUREC_DEV_DEPS_IMAGE"
docker buildx build --network=host \
  --build-arg BASE_IMAGE="$NUREC_DEV_DEPS_IMAGE" \
  -t "$NUREC_IMAGE" -f "$NUREC_SRC/deps/docker/Dockerfile-dev.build" \
  --load "$NUREC_SRC/deps/docker"
python3 "$NUREC_SRC/skills/nurec/scripts/validate_setup.py" --source "$NUREC_SRC"
source "$NUREC_SRC/skills/nurec/scripts/nurec_dev.sh"
nurec_dev bazelisk build //:run //apps:nurec_tools
```

The helper follows Software Setup: same-path source and build-cache mounts, host
UID/GID, a writable build HOME, persistent ccache, GPU access, and 64 GiB shared
memory. Set `NUREC_OUTPUT_BASE` before sourcing to use a different build cache.
Do not share one output base between concurrently running Bazel commands.

```bash
# Additional host data, outside the source checkout:
NUREC_DOCKER_ARGS+=(--volume /data/nurec:/data/nurec)
nurec_dev bazelisk run //:run -- render --help
```

The sourced helper does not build or pull images, install drivers, modify Docker,
or download data. Run the image build above before using it. Rebuild the
development image when its Dockerfile or dependency-image pin changes, and use
distinct development-image tags for different source/dependency pairs. A checksum
mismatch under `/opt/nvidia/nurec-build-deps` means the source and dependency
image disagree; correct the pair before continuing.
Configure Docker and NVIDIA Container Toolkit before using it. Host Python 3 is
needed only for preflight; the build uses Python 3.13 in the dependency environment.
The standalone gRPC client requires its own Python 3.13 environment.

The public dependency image does not require an NVIDIA registry key. For gated datasets and
optional model weights, accept the repository's terms and export `HF_TOKEN` in
your shell. Do not put secrets in shell arguments, YAML, logs, or chat.
