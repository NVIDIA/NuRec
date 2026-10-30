<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec — Teardown / disk cleanup

A source workflow leaves a published dependency image, a built development image, Bazel/build caches,
optional model weights, and a clip-sized
`<output_dir>/<RUN-ID>/` tree. Stop the task-owned service at session end.
The disk cleanup below is optional: preserve data, outputs, and build caches
unless the user requests their removal.

> **Set these first.** Every command below refuses to run unless both are
> set and non-empty. Do not substitute the placeholders by hand — an empty
> run id turns a targeted delete into a delete of the whole output tree.

```bash
OUTPUT_DIR=/path/to/output     # the --output-dir you trained with
RUN_ID=my-run-id               # the run you want to remove

# `${VAR:?}` alone only rejects empty/unset -- RUN_ID=../other-run would still
# escape OUTPUT_DIR. Reject anything that is not a single plain name.
#
# Defined as a function so it can be re-run AFTER each variable is assigned.
# Checking a variable before it is set validates nothing.
require_plain_name() {   # $1 = variable name
    case "${!1-}" in
        "")        echo "STOP: $1 is unset" >&2; return 1 ;;
        */*|.*)    echo "STOP: $1 must be a single directory name, not a path" >&2; return 1 ;;
    esac
}
require_plain_name RUN_ID || exit 1
```

## 1. Stop this session's serve-grpc container

Use the ownership-checking helper for a service started by this skill:

```bash
bash "$NUREC_SRC/skills/nurec/scripts/session_teardown.sh"
```

For a manually started service, stop only the process/container created for
this task. Do not select containers by their shared dependency-image ancestor.

## 2. Optionally remove the development and dependency images

```bash
docker image rm "${NUREC_IMAGE:?set the exact development image to remove}"
docker image rm "${NUREC_DEV_DEPS_IMAGE:?set the exact dependency image to remove}"
```

`docker image prune` and `docker builder prune` are **host-global** — they
delete unrelated images and build cache. Run them only if you intend that.

## 3. Remove NuRec caches (Fixer weights + Hydra-resolved configs)

```bash
rm -rf "${HOME:?}/.cache/nurec"
```

## 4. Remove one run's outputs

```bash
rm -rf "${OUTPUT_DIR:?set OUTPUT_DIR}/${RUN_ID:?set RUN_ID}"
```

The `:?` guards are the point: if either variable is unset or empty, bash
aborts before `rm` runs instead of deleting `${OUTPUT_DIR}` entire.

## 5. (optional) Remove NCore shards

```bash
DATASET_DIR=/path/to/dataset
NAME=my-clip-name

# NAME is only assigned here, so it must be validated HERE -- the check at the
# top of this document ran before it existed. NAME=../outside would otherwise
# resolve outside DATASET_DIR.
require_plain_name NAME || exit 1

rm -rf "${DATASET_DIR:?}/${NAME:?}".zarr.itar "${DATASET_DIR:?}/${NAME:?}".aux.*.zarr
```

## Outputs created without `-u` (root-owned)

The source helper preserves host UID/GID. Older runs may still have
root-owned files. Re-own only the run you are deleting, and stay on
one filesystem:

```bash
# GNU chown has no -x / --one-file-system, so the one-filesystem and
# root-owned-only constraints come from find, not chown:
#   -xdev    do not cross into another mounted filesystem
#   -user 0  touch only root-owned entries, leaving other users' files alone
#   -h       act on symlinks themselves rather than their targets
sudo find "${OUTPUT_DIR:?}/${RUN_ID:?}" -xdev -user 0 \
     -exec chown -h "$(id -u):$(id -g)" {} +
rm -rf "${OUTPUT_DIR:?}/${RUN_ID:?}"
```

Do **not** `chown -R` the parent `${OUTPUT_DIR}` — it may hold other users'
runs.

## Post-teardown verification

```bash
docker image ls "$NUREC_IMAGE"
du -sh "${HOME:?}/.cache/nurec" 2>/dev/null || echo "cache: clean"
ls -la "${OUTPUT_DIR:?}/${RUN_ID:?}" 2>/dev/null || echo "output: clean"
```
