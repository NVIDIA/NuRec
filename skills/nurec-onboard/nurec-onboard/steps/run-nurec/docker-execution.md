<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Docker execution rules for NuRec containers

These rules apply to **every `docker run`** in auxiliary-data generation and NuRec
reconstruction (see [SKILL.md](SKILL.md)). The short-lived `--rm --entrypoint
python|bash` recipes used only for config inspection are exempt where noted.

## Contents

- [One command, literal arguments](#one-command-literal-arguments)
- [Flag rules](#flag-rules)
- [Network egress check](#network-egress-check)
- [Run as the host user](#run-as-the-host-user)
- [Long-running containers: detach and poll](#long-running-containers-detach-and-poll)
- [Log-reading discipline](#log-reading-discipline)

## One command, literal arguments

Docker runs on the host rather than inside the agent's command sandbox. A runner
may decide that per command by inspecting the command it is handed, and then only
a single `docker ...` invocation built from literal words qualifies. Under such a
runner the forms below lose the host route and fail with `permission denied while
trying to connect to the docker API at unix:///var/run/docker.sock`, which reads
like a broken daemon rather than a rejected command. Write the right-hand form
regardless of runner; it works everywhere:

| Do not issue                              | Issue instead                                                  |
| ----------------------------------------- | -------------------------------------------------------------- |
| `cd /data && docker run ...`              | `docker run ...` with absolute paths                           |
| `docker logs <name> \| tail -50`          | `docker logs --tail 50 <name>`                                 |
| `docker logs <name> > log.txt`            | `docker logs <name>` and read the output                       |
| `docker run ... --user $(id -u):$(id -g)` | `docker run ... --user <uid>:<gid>` (for example, `1000:1000`) |
| `docker run ... -e HF_TOKEN=${HF_TOKEN}`  | `docker run ... -e HF_TOKEN`                                   |
| `docker run ... "$image"`                 | `docker run ... <dev_image>`                                   |
| `FOO=bar docker run ...`                  | `docker run ... -e FOO` after exporting `FOO` separately       |
| `docker cp <name>:/out/* /tmp/out`        | `docker cp <name>:/out/. /tmp/out`                             |

Every `<name>` in these recipes stands for a value resolved during pre-flight:
substitute the resolved text, and never pass a shell variable in its place.
`<dev_image>` is the `Development image` line of `$NUREC_WORKDIR/nurec-runtime.md`.
Obtain uid and gid with separate `id -u` and `id -g` commands, then write the
numbers into the Docker command. `-e HF_TOKEN` with no `=value` is the
pass-through form and takes the value from the environment.

Quoting is fine, because a quoted string with nothing to expand is still literal:
`--format '{{.Server.Version}}'` and `--entrypoint bash <image> -c "cat
configs/apps/<path>.yaml"` both keep the host route. It is expansion and
composition that cost it, not quotes. Wildcard words containing `*` or `?` and
environment assignments before `docker` are also ineligible for rule matching.

## Flag rules

Keep `--shm-size=64g`, `--gpus all`, and `--user <uid>:<gid>`.
Mount source, cache, data, and output at their absolute host paths, with the
source read-only. Use the source-built Bazel targets and startup flags shown in
[SKILL.md](SKILL.md). `HF_TOKEN` is optional; pass it by name only when available.
Reducing `--shm-size` below `64g` crashes PyTorch dataloaders with
`No space left on device`. Add `--network host` only when Docker's default
bridge cannot reach build dependencies or model-weight CDNs.

## Network egress check

The NuRec containers use Docker's default bridge network; they fetch a few
external model weights at runtime (e.g. the torchvision VGG16 backbone behind the
LPIPS metric). If a host's default bridge network cannot reach those weight CDNs,
add `--network host` to every aux-generation and reconstruction `docker run`.

To confirm up front whether this host needs `--network host`, the operator may
probe TCP/TLS reachability to the weight CDN host from inside the development image with
the default Docker bridge and then with host networking. The probe targets the CDN
host alone, not a specific weight file, so it stays valid as the weights NuRec
fetches change across versions:

```bash
docker run --rm --entrypoint python3 <dev_image> \
  -c "import socket; socket.create_connection(('download.pytorch.org', 443), timeout=10); print('reachable')"

docker run --rm --network host --entrypoint python3 <dev_image> \
  -c "import socket; socket.create_connection(('download.pytorch.org', 443), timeout=10); print('reachable')"
```

If both forms print `reachable`, keep the published recipes unchanged. If only the
`--network host` form succeeds, add `--network host` to every aux-generation and
reconstruction `docker run`. If both fail, the host itself cannot reach the
model-weight CDN and the fix is outside this step.

## Run as the host user

Add `--user <uid>:<gid>` to every long-running `docker run` (the aux-generation
and reconstruction recipes already include it). The container then writes outputs
(aux ITARs, checkpoints, validation renders, `metrics.yaml`) as the host user
instead of `root:root`, so later steps read and remove them without `sudo`.

Resolve the two numbers once with separate `id -u` and `id -g` commands and write
them literally into each Docker command, as
[One command, literal arguments](#one-command-literal-arguments) requires. A
`$(id -u)` substitution inside the Docker command costs it the host route.

The development-dependencies image runs as the host user when its build
paths are writable. Set `HOME=<nurec_cache>/build-home`, `USER=<username>`,
and `CCACHE_DIR=<nurec_cache>/ccache` as shown in the commands. The launcher
creates these directories and a Bazel rc that keeps build outputs and symlinks
in the cache, so the source mount can remain read-only. ITARs, checkpoints,
and validation outputs remain host-owned.

**Fallback.** Drop `--user` only if the workload itself exits mid-run with a
permission / root-required error (e.g. it tries to `chown`, write to `/root/...`,
or otherwise needs container-root capabilities). Outputs then land as
`root:root`, which the host-uid user cannot remove from the normal sandbox.
Recovering ownership is an **operator task**: the operator runs `sudo chown -R
$(id -u):$(id -g) <path>`, after which a normal user-mode `rm -rf <path>` works
inside the agent's sandbox. The agent must not invoke `sudo rm -rf <path>`
directly; chown-then-normal-rm keeps the destructive step under the same safety
heuristics as every other cleanup step.

## Long-running containers: detach and poll

Auxiliary-data generation and NuRec reconstruction are both long-running, and
their wall time scales with dataset size and hardware. Either can exceed the
per-invocation foreground-command cap that agent runners commonly enforce on
terminal subprocesses, so the recipes use `docker run -d --name <name> ...`.

Poll with **one Bash invocation per iteration -- NOT a shell `while` or `for`
loop.** A loop is a compound command, so it is not the single literal `docker`
invocation the agent runner needs in order to run Docker on the host (see
[One command, literal arguments](#one-command-literal-arguments)); it fails with
`permission denied while trying to connect to the Docker API at
unix:///var/run/docker.sock` even though a one-shot `docker inspect` of the same
container succeeds. Issue the launch, each poll, the final-log read, and the
cleanup as separate Bash invocations:

```bash
# Bash invocation 1 -- launch:
docker run -d --name <name> ... <image> <args...>

# Bash invocations 2..N -- poll (re-issue this whole command, one invocation
# at a time, until State.Status != "running"). Do NOT prepend `sleep N &&`:
# the agent runner owns the inter-poll delay (e.g. `block_until_ms`), so the
# `docker inspect` call must stay a single short-lived command. Bundling a
# `sleep` inside the Bash invocation hides the wait from the runner's
# cancellation / telemetry path and inflates the foreground budget of every
# poll iteration.
docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' <name>

# Bash invocation N+1 -- read final logs:
docker logs --tail 200 <name>

# Bash invocation N+2 -- clean up:
docker rm <name>
```

Use the same `<name>` across launch / poll / log / rm. If the launch fails because
`<name>` is already in use, the container is left over from an interrupted run in
the same workdir: confirm with `docker inspect -f '{{.State.Status}}' <name>` that
it is not `running`, `docker rm <name>`, then relaunch. Drive the inter-poll wait
through the runner's native delay parameter (e.g. a `block_until_ms` on the order
of tens of seconds); each `docker inspect` completes in under a second. The recipes
intentionally omit `--rm` so logs and exit state survive crashes until the
explicit `docker rm <name>` after diagnosis; the interpretation steps assume
both are available.

## Log-reading discipline

**No interstitial `docker logs --tail` peeks during the wait.** The `docker
inspect` poll line produces ~50 tokens per iteration; a reactive `docker logs
--tail 200` against a tqdm progress stream produces 200-2000 tokens of
carriage-return-rewritten progress bars that say nothing new (tqdm overwrites the
same line). Read logs once, _after_ `docker inspect` reports `Status !=
running`. If the wait feels long, lengthen the runner-side delay -- do not peek
logs.

**Cold-build startup.** The first source invocation downloads Bazel dependencies
and builds the target before the application starts. It is
human-readable but verbose; treat it as healthy startup, not stuck. The first
signal that conversion / training has actually started is dataset-specific (e.g.
`Loading sequence ...`, the first tqdm `it/s` line). Allow 15–20 minutes for a cold release build. A 67 GB per-workdir cache was
observed. These are estimates, not a timeout or disk-space guarantee. Keep polling container state while the build runs; compilation
output before application startup is normal, and a quiet 30-second interval
is not evidence of a hang.
