<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# viser 0.2.23+nre5 - provenance and change record

`viser==0.2.23+nre5` is **not** built by `Dockerfile-wheels`. It is built by hand from the
NVIDIA-internal fork and published to the NRE package registry
(an NVIDIA-internal registry; wheel sha256
`e6f44d38d6b79539ea80052ece735784132cbd78b49bc62f48b59c1e618d5a3f`).

## Source

- Fork: NVIDIA-internal fork of upstream viser (branch `main`).
  The `+nre5` wheel is built from commit `594d497d` ("bump version to nre5 ...").
- Upstream: `https://github.com/nerfstudio-project/viser` (`LICENSE` in the wheel is Apache-2.0).
- Build recipe: `NRE_BUILD.md` at the root of the fork (its content is in the patch in this directory).

## What `+nre5` contains relative to upstream

The fork is upstream `v0.2.23` plus 59 commits. They fall in two groups:

1. **Unmodified upstream commits** (Brent Yi et al., Mar-Apr 2025, through upstream commit
   `5f3c1f98` "Fix point cloud shader for Firefox (#454)"): the wheel is therefore a snapshot of
   upstream `main` after `0.2.23`, not the PyPI `0.2.23` release (about 60 files differ from the
   PyPI wheel, including new `Html.tsx`, `csm/`, `mesh/` client sources and rebuilt JS bundles).
2. **NVIDIA changes** (Jiahui Huang, jiahuih@nvidia.com; one commit by liruilong940607), diff in
   `nvidia-delta-over-upstream-5f3c1f98.patch` (this directory):
   - remove the `plyfile` dependency (GPL-licensed) from `pyproject.toml` (also drops it from the
     wheel `METADATA` `Requires-Dist`);
   - raise the websocket `max_size` to 50 MB (`src/viser/infra/_infra.py`);
   - screenshots reuse the existing WebGL renderer instead of creating a new one
     (`src/viser/client/src/MessageHandler.tsx`);
   - add `NRE_BUILD.md`, a `dist/` `.gitignore` entry;
   - stamp `__version__` with `+nre5` at build time (`sed` in `NRE_BUILD.md`).

## License

viser is licensed under **Apache-2.0**, per the `LICENSE` file shipped in the wheel. The wheel
`METADATA` declares `License: MIT`; that is an upstream metadata error (the upstream
`pyproject.toml` says MIT, while its `LICENSE` file is Apache-2.0), and we follow the shipped
`LICENSE` file.

## Apache-2.0 section 4(b) status

Section 4(b) requires modified files to carry prominent notices that they were changed. The
`+nre5` wheel itself does **not** carry a change notice: the modified files in it have no
"modified by" comment, and the wheel ships no statement of changes. This directory (the change
list above and the patch) is the statement of changes for the published wheel.

A rebuilt wheel, `+nre6`, that includes the notice is planned. It will add a
`Modified by NVIDIA CORPORATION & AFFILIATES, <year>` comment to `_infra.py` and
`MessageHandler.tsx` in the fork, and ship this change list inside the wheel `dist-info` (for
example as `NUREC_CHANGES.txt`). Until then the `+nre5` wheel is not section 4(b) compliant on its
own.
