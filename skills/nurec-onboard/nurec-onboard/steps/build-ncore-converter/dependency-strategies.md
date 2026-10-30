<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dependency-strategy ladder

Four sanctioned options for pulling the dataset's reader package into the
converter, in escalation order. Pick the first that works, record the choice as
`dependency_strategy` in `inferred_parameters`, and only escalate when the current
option _actually_ fails (not on a predicted conflict). See
[SKILL.md "Parse input and infer sensor properties"](SKILL.md#parse-input-and-infer-sensor-properties)
(step 3) for where this fits in scaffold mode and
[SKILL.md "Run the converter"](SKILL.md#run-the-converter) for the
`ModuleNotFoundError` / lockfile-failure escalation triggers.

Exploration scripts default to `bazel run`; system Python is allowed only under
Option 4, where the reader is not Bazel-resolvable.

**Option 1 -- direct pip dep (default).** Add the reader to
`deps/pip/requirements_3_X.in`, regenerate the lockfile
(`bazel run //deps/pip:requirements`), and add `requirement()`
entries in the converter `BUILD.bazel`. If the reader requires a different
Python ABI, bump `python_version` in `MODULE.bazel` (update toolchain, both
`pip.parse` calls, `use_repo` names, lockfile path, and requirements filename
to match; regenerate). Predicted conflicts are not grounds to skip Option 1
-- only escalate when the lockfile resolver _actually_ fails.

- **Pin the reader when the dataset ships serialized library objects** (pickle,
  joblib, `torch.save`). The writing library's version is part of the data
  format: loading resolves the writer's classes by module path, so a bare
  requirement that lets the resolver pick a newer major can fail the load or
  return different fields. Pin to the major that wrote the export and note the
  reason alongside `dependency_strategy`. Pin without waiting for a failure --
  an unnecessary pin costs one re-lock, a missing one costs a debugging detour
  mid-conversion.

**Option 2 -- Bazel dep + `sys.path` shim.** Use when Option 1 fails due to
a native-binding clash (most commonly **protobuf**: the reader's proto wheels
collide with another transitive). Pull only the **conflicting native library**
into `MODULE.bazel` via `bazel_dep` (Bazel Central Registry) or `http_archive`
(no module published). Add the reader itself to pip as in Option 1. Add a
small `deps.py` shim (~30 lines) that removes the clashing pip binding from
`sys.path` before importing the reader.

**Option 3 -- vendor a minimum viable slice.** Permissive license only
(Apache-2.0 / MIT / BSD). Use when Options 1-2 are blocked and the reader
is small. Copy only the imported modules into `vendored/<reader_name>/`,
preserve `LICENSE`, add a `vendored/BUILD.bazel` `py_library`. Record the
reader name, version/commit, license, and extracted files in
`critical_parameter_justifications`.

**Option 4 -- preprocess-step fallback (last resort).** Use only when
Options 1-3 are all blocked (lockfile actually failed, no `bazel_dep` for the
conflicting binding, reader too large or copyleft to vendor). Scaffold
`scripts/preprocess_<dataset>.py` under system Python emitting an intermediate
directory; the Bazel converter reads from that dir only. Record blockers and
surface long-term unblockers in `notes`.

- **Hard requirement**: the preprocess script MUST call the reader's canonical
  decoder functions for every correction it documents -- no hand-rolling
  point-cloud extraction, motion compensation, extrinsic application, or frame
  transforms. Hand-rolling any of these silently drops dataset-specific
  corrections and is treated as a converter bug at validation. List the specific
  reader functions called (one line each) under `notes` so the reviewer can
  audit that no decoder was reimplemented.

## HTTP 407 during lockfile regeneration

A proxy that requires credentials fails the `pip.parse` simple-API metadata
fetch; the `startup --host_jvm_args` line in `.bazelrc` fixes it, and its
comment there explains why. The converter workspace ships that line, so
reaching this section means the flag was removed or the proxy needs something
else.

Option 1 is where this surfaces: the committed lockfile and the warm repository
cache keep every other step off the network, so nothing reaches the proxy until
a new requirement forces module-extension re-evaluation. Read as a resolver
failure it looks like grounds to escalate to Option 4, but it is an environment
problem and escalating leaves a preprocess shim in place of a working pip dep.
