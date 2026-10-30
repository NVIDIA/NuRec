# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble a CUDA toolkit directory out of hermetic toolkit components."""

# The hermetic toolkit keeps each component in its own repository, laid out as
# "<component>/{bin,include,lib,nvvm}/...". This assembles a classic toolkit
# layout for consumers that need it.
_ASSEMBLE_SCRIPT = """#!/bin/bash
set -euo pipefail

output_dir="$1"
manifest="$2"

mkdir -p "$output_dir"
while read -r src dst; do
    target="$output_dir/$dst"
    mkdir -p "$(dirname "$target")"
    # Bazel stages inputs as symlinks. Keep them as such, since whatever takes
    # this tree out of the build reads through them.
    cp -a "$src" "$target"
done < "$manifest"

# Various consumers look for libraries under lib64, the components ship
# them under lib.
if [[ -d "$output_dir/lib" ]]; then
    mv "$output_dir/lib" "$output_dir/lib64"
fi
"""

def _keep(relative_path):
    """Whether a component file belongs in the tree.

    Static archives are over 2GB and nothing links CUDA statically, apart from
    libcudart_static.a, which nvcc's own link line needs.
    """
    basename = relative_path.rsplit("/", 1)[-1]
    if basename == "libcudart_static.a":
        return True
    return not (basename.endswith(".a") and "_static" in basename)

def _component_relative_path(file):
    """Strip the "../<component_repo>/<component>" prefix off a component file.

    What remains is already the path the classic toolkit layout uses.
    """
    parts = file.short_path.split("/", 3)
    if len(parts) != 4 or parts[0] != "..":
        fail("unexpected CUDA component file layout: {}".format(file.short_path))
    return parts[3]

def _cuda_toolkit_tree_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.attr.name)

    entries = []

    # Every component ships the same LICENSE at the same path. Keep the first
    # file to claim a destination and skip any later one.
    taken = {}
    for dep in ctx.attr.deps:
        for f in dep.files.to_list():
            dst = _component_relative_path(f)
            if _keep(dst) and dst not in taken:
                taken[dst] = True
                entries.append((f, dst))
    for dep in ctx.attr.header_only_deps:
        for f in dep.files.to_list():
            dst = _component_relative_path(f)
            if dst.startswith("include/") and dst not in taken:
                taken[dst] = True
                entries.append((f, dst))

    manifest = ctx.actions.declare_file(ctx.attr.name + "_manifest.txt")
    ctx.actions.write(
        output = manifest,
        content = "".join(["{} {}\n".format(f.path, dst) for f, dst in entries]),
    )

    script = ctx.actions.declare_file(ctx.attr.name + "_assemble.sh")
    ctx.actions.write(
        output = script,
        content = _ASSEMBLE_SCRIPT,
        is_executable = True,
    )

    args = ctx.actions.args()
    args.add(output_dir.path)
    args.add(manifest.path)

    ctx.actions.run(
        inputs = [f for f, _ in entries] + [manifest],
        outputs = [output_dir],
        executable = script,
        arguments = [args],
        mnemonic = "AssembleCudaToolkit",
        progress_message = "Assembling CUDA toolkit %{output}",
    )

    return [DefaultInfo(files = depset([output_dir]))]

cuda_toolkit_tree = rule(
    implementation = _cuda_toolkit_tree_impl,
    attrs = {
        "deps": attr.label_list(
            doc = "Components to assemble whole, as <component>_all_files targets",
            allow_files = True,
        ),
        "header_only_deps": attr.label_list(
            doc = "Components to take the headers of, leaving their libraries out",
            allow_files = True,
        ),
    },
    doc = "Assembles hermetic CUDA toolkit components into a classic toolkit directory",
)
