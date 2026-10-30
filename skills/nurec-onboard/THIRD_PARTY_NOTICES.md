# Third-Party Notices

This directory does not vendor or redistribute third-party OSS source code or
third-party binary dependencies.

NuRec Onboard's Bazel targets build in the enclosing repository's root
workspace and take their dependencies from its pins. The converter workspace
template under `ncore_template/` declares its own dependencies by name and
version. Bazel downloads them at build time; they are not part of this
directory and retain their own license terms. The template's complete pinned
dependency sets, including transitive dependencies, are in its lockfiles:

- [`ncore_template/deps/pip/requirements_3_13.txt`](ncore_template/deps/pip/requirements_3_13.txt)
- [`ncore_template/MODULE.bazel.lock`](ncore_template/MODULE.bazel.lock)

If third-party OSS code is added to this directory in the future, its copyright
notice, attribution, and license text must be added here before release.

## Direct dependencies

Python packages, which the template installs from PyPI, except for PyTorch's
CPU-only build, which it installs from PyTorch's package index:

- nvidia-ncore — Apache-2.0 — <https://github.com/NVIDIA/ncore>
- PyTorch — BSD-3-Clause — <https://pytorch.org/>
- click — BSD-3-Clause — <https://click.palletsprojects.com/>
- NumPy — BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 — <https://numpy.org/>
- PyYAML — MIT — <https://github.com/yaml/pyyaml>
- SciPy — BSD-3-Clause — <https://scipy.org/>
- universal-pathlib — MIT — <https://github.com/fsspec/universal_pathlib>

Bazel modules the template fetches from the Bazel Central Registry:

- aspect_rules_lint — Apache-2.0 — <https://github.com/aspect-build/rules_lint>
- aspect_bazel_lib — Apache-2.0 — <https://github.com/bazel-contrib/bazel-lib>
- buildifier_prebuilt — MIT — <https://github.com/keith/buildifier-prebuilt>
- rules_python — Apache-2.0 — <https://github.com/bazel-contrib/rules_python>
- rules_shell — Apache-2.0 — <https://github.com/bazelbuild/rules_shell>
- rules_uv — Apache-2.0 — <https://github.com/theoremlp/rules_uv>
- platforms — Apache-2.0 — <https://github.com/bazelbuild/platforms>
- rules_multirun — Apache-2.0 — <https://github.com/keith/rules_multirun>
- rules_multitool — Apache-2.0 — <https://github.com/theoremlp/rules_multitool>

## Build tools

The Bazel rules above download these prebuilt tools and toolchains to format,
lint, and build NuRec Onboard. They run on the developer's machine and are not
distributed with this directory:

- Ruff — MIT — <https://github.com/astral-sh/ruff>
- ShellCheck — GPL-3.0 — <https://github.com/koalaman/shellcheck>
- uv — Apache-2.0 OR MIT — <https://github.com/astral-sh/uv>
- buildifier — Apache-2.0 — <https://github.com/bazelbuild/buildtools>
- CPython, from python-build-standalone — PSF-2.0 (build scripts MPL-2.0) — <https://github.com/astral-sh/python-build-standalone>

## Runtime materials

The workflow also retrieves NVIDIA container images and NVIDIA CUDA wheels at
run time. They are not distributed with this directory; see the
[Notice and disclaimer](README.md#notice-and-disclaimer) section of the README.
