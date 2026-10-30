<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# How to Patch Python Wheels

## Introduction

This guide provides step-by-step instructions for patching Python wheels. This is useful when you need to modify a package without waiting for upstream changes.

## Prerequisites

### 1. Create a conda environment

```bash
conda create -n patch_wheel_env python=3.11 -y
```

### 2. Activate the environment

```bash
conda activate patch_wheel_env
```

## Steps

### 1. Download the original wheel

```bash
pip download package==version --python-version 3.11 --only-binary=:all: --no-deps
```

### 2. Extract the wheel contents

```bash
unzip package-version-py3-none-any.whl -d patched
```

### 3. Apply your patch

- Either copy files from your patched repository:
  ```bash
  rsync -avr ../patched-repo/path/to/module/* patched/path/to/module/
  ```
- Or manually edit the necessary files

### 4. Update version information

- Preferred for NuRec wheels: append the `+nre<N>` local version (see
  [The `+nre` convention](#the-nre-convention)):
  ```bash
  NEW_VERSION="version+nre1"
  ```
- Alternatively, take the commit hash of your patch:
  ```bash
  GIT_HASH=$(git -C patched-repo rev-parse --short HEAD)
  NEW_VERSION="version+git$GIT_HASH"
  ```
- Update the METADATA file:
  ```bash
  sed -i "s/original-version/$NEW_VERSION/" patched/package-version.dist-info/METADATA
  ```

### 5. Add the change notice (required for Apache-2.0 packages)

Apache-2.0 section 4(b) requires a statement that modified files were changed. Do this before
repacking; see [Change notices](#change-notices-apache-20-section-4b) for the details.

- Put the change list in `NUREC_CHANGES.txt` inside the wheel's `*.dist-info/`.
- Where practical, start each patch with a short plain-text `Modified by NVIDIA CORPORATION & AFFILIATES, <year>`
  preamble. Do not add notice hunks to the modified source files.

### 6. Rename the dist-info directory

```bash
mv patched/package-version.dist-info patched/package-$NEW_VERSION.dist-info
```

### 7. Repack the wheel

```bash
python -m wheel pack patched
```

### 8. Verify the patched wheel (optional)

```bash
unzip -l package-$NEW_VERSION-py3-none-any.whl
```

### 9. Publish the wheel

```bash
pip install twine
twine upload --repository your-repo-url package-$NEW_VERSION-py3-none-any.whl
```

## Example

```bash
# Create and activate conda environment
conda create -n package_patch_env python=3.11 -y
conda activate package_patch_env

# Install necessary tools
pip install wheel twine

# Download and patch the wheel
pip download setuptools==78.1.1 --python-version 3.11 --only-binary=:all: --no-deps
unzip setuptools-78.1.1-py3-none-any.whl -d patched
rsync -avr ../setuptools/setuptools/_vendor/jaraco/* patched/setuptools/_vendor/jaraco/

# Update version and repack
GIT_HASH=$(git -C ../setuptools rev-parse --short HEAD)
NEW_VERSION="78.1.1+git$GIT_HASH"
sed -i "s/78.1.1/$NEW_VERSION/" patched/setuptools-78.1.1.dist-info/METADATA
mv patched/setuptools-78.1.1.dist-info patched/setuptools-$NEW_VERSION.dist-info
python -m wheel pack patched

# Upload the patched wheel
twine upload --repository your-repo-url setuptools-$NEW_VERSION-py3-none-any.whl

# Deactivate the environment when done
conda deactivate
```

## The `+nre` convention

NuRec wheels that are patched or rebuilt are published to the NRE package registry with a PEP 440
local version `<upstream version>+nre<N>`. The local segment makes the resolver prefer the NuRec wheel
over the PyPI release of the same version, so `pyproject.toml`, `requirements_*.in` and
the Python dependency definitions pin the exact `+nre<N>`.

- Bump `<N>` on every rebuild that changes the wheel contents, including a change to a patch or to a
  change notice. Never republish different content under the same version.
- Wheels built by NVIDIA's custom wheel build take the number from the
  `*_LOCAL_VERSION` / `*_PATCH_VERSION` build argument of their stage. When it changes, update the
  stage comments, `custom_wheels/<wheel>/README.md`, any customer notice that names the version, and the
  pins above.
- Compiled CUDA wheels use `+<commit>pt<torch>cu<cuda>...` local versions, with a trailing `.nre<N>`
  on the ones that are also patched.
- Record in `NUREC_CHANGES.txt` whether a `+nre` bump is a source modification or a rebuild only (for
  example against a different FFmpeg), so reviewers can tell which obligations apply.
- The patches (and a copy of each wheel's change notice) live in `deps/custom_wheels/<wheel>/` (see
  `deps/custom_wheels/README.md`); the wheel build reads the patches from there.
- Add a `patch_note` for the wheel to the third-party notice generator inputs so that
  `THIRD_PARTY_NOTICES.md` gets a `Modified:` line. `THIRD_PARTY_NOTICES.md` is generated; never edit it by hand.

## Change notices (Apache-2.0 section 4(b))

Apache-2.0 section 4(b): _You must cause any modified files to carry prominent notices stating that You
changed the files._ It applies to every Apache-2.0 package we modify, including NVIDIA's own
(cosmos-predict2, TransformerEngine). It does not apply to a pure rebuild of unmodified source, but say
so in `NUREC_CHANGES.txt`, because a rebuild against another library still changes the binary.

NuRec states the changes in `NUREC_CHANGES.txt` and and, where practical, in a plain-text preamble at the top of the patch
(leading prose before the first diff header, which `patch` and `git apply` ignore). The patches do
not add notice comments to the modified files.

1. **Statement of changes.** List every modified file and what changed in `NUREC_CHANGES.txt`. Keep
   that file next to the patches and have the Dockerfile install it into the wheel's `*.dist-info/`
   before `wheel pack` (see the `decord`, `opencv_python`, `cosmos_predict2` and `transformer_engine`
   stages of `Dockerfile-wheels`).
2. **Patch preamble.** Begin each patch with a short plain-text note, for example:
   ```
   Modified by NVIDIA CORPORATION & AFFILIATES, 2026 (statement of changes, Apache-2.0
   section 4(b)): this patch changes <file> of <project>.
   ```
   Do not remove or alter the upstream copyright, license, attribution or NOTICE text.
3. **Keep the upstream `LICENSE` (and `NOTICE`, if any)** in the wheel's `dist-info/licenses/`.
4. **Verify the patch applies** to the pinned upstream source with
   `patch --dry-run --batch --forward --fuzz=0 -p1 -i <patch>` from the source root (for wheels, from
   the unpacked wheel directory), and check the finished wheel with `unzip -l` for `NUREC_CHANGES.txt`.
5. **Internal forks.** If the build clones an NVIDIA fork instead of upstream, the fork's own commits are
   modifications too. Diff the pinned fork commit against the upstream tag and record the changed
   files in `NUREC_CHANGES.txt` as well (see `custom_wheels/transformer_engine/NUREC_CHANGES.txt`).
6. **Patch hygiene.** Generate patches with `diff -u` (with context) instead of editing hunk headers by
   hand, so that `patch --fuzz=0` applies without offsets. An offset makes `patch` leave a `*.orig`
   file behind, which then ships in the wheel.

## Notes

- When using `python -m wheel pack`, the RECORD file is automatically updated
- The WHEEL file typically doesn't contain version information that needs updating
- For automation, consider creating a script or Dockerfile to handle the process
- Version naming convention: `<original_version>+git<git_hash>` (e.g., 78.1.1+gitb18a11365)
