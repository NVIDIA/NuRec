# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch the CUDA compat apt package as a raw .deb and package it as an archive."""

load("@tar.bzl//tar:tar.bzl", "tar")

# The OS/version segment ("ubuntu2404") must match the run base image
# (deps/docker/Dockerfile-run.build's `FROM ubuntu:24.04`), since NVIDIA's apt
# repo is laid out per distro release.
_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404"

_BUILD_CONTENT = """
filegroup(
    name = "deb",
    srcs = ["{filename}"],
    visibility = ["//visibility:public"],
)
"""

def _cuda_compat_deb_impl(repository_ctx):
    if repository_ctx.os.arch in ("aarch64", "arm64"):
        apt_arch = "sbsa"
        deb_arch = "arm64"
        sha256 = repository_ctx.attr.sha256_sbsa
    else:
        apt_arch = "x86_64"
        deb_arch = "amd64"
        sha256 = repository_ctx.attr.sha256_x86_64

    filename = "cuda-compat-{}_{}_{}.deb".format(repository_ctx.attr.cuda_version, repository_ctx.attr.version, deb_arch)
    repository_ctx.download(
        url = "{}/{}/{}".format(_REPO_URL, apt_arch, filename),
        output = filename,
        sha256 = sha256,
    )
    repository_ctx.file("BUILD.bazel", _BUILD_CONTENT.format(filename = filename))

cuda_compat_deb = repository_rule(
    implementation = _cuda_compat_deb_impl,
    attrs = {
        "cuda_version": attr.string(
            mandatory = True,
            doc = "CUDA major-minor version the package is named for, e.g. '12-8'",
        ),
        "sha256_sbsa": attr.string(mandatory = True),
        "sha256_x86_64": attr.string(mandatory = True),
        "version": attr.string(mandatory = True, doc = "Package version, e.g. '570.211.01-0ubuntu1'"),
    },
    doc = "Raw cuda-compat-<cuda_version> .deb package for the host architecture",
)

# Copies the .deb's usr/local/cuda-*/compat/ directory out and writes an mtree
# describing it, symlinks included: libnvidia-container globs compat/lib*.so.*,
# resolves and dedupes the matches, then mounts each one over the driver
# libraries by basename. Intact, the SONAME symlinks collapse onto their
# targets and only the versioned libraries are mounted; flattened into regular
# files they are mounted too, putting the compat libcuda.so.1 over a newer host
# driver (CUDA error 803).
_EXTRACT_SCRIPT = """#!/bin/bash
set -euo pipefail

deb="$(readlink -f "$1")"
output_dir="$2"
mtree="$(readlink -f "$3")"
prefix="$4"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

(cd "$workdir" && ar x "$deb" data.tar.xz && tar xf data.tar.xz)
mkdir -p "$output_dir"
cp -a "$workdir"/usr/local/cuda-*/compat/. "$output_dir"/

{
  echo "#mtree"
  dir=""
  for part in $(echo "$prefix" | tr '/' ' '); do
    dir="$dir/$part"
    echo ".$dir type=dir uid=0 gid=0 mode=0755 time=0"
  done
  for path in $(cd "$output_dir" && find . -mindepth 1 -maxdepth 1 | sort); do
    name="${path#./}"
    if [[ -L "$output_dir/$name" ]]; then
      echo "./$prefix/$name type=link link=$(readlink "$output_dir/$name") uid=0 gid=0 mode=0755 time=0"
    else
      echo "./$prefix/$name type=file content=$output_dir/$name uid=0 gid=0 mode=0644 time=0"
    fi
  done
} > "$mtree"
"""

def _cuda_compat_libs_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.label.name)
    mtree = ctx.actions.declare_file(ctx.label.name + ".mtree")

    script = ctx.actions.declare_file(ctx.label.name + "_extract.sh")
    ctx.actions.write(output = script, content = _EXTRACT_SCRIPT, is_executable = True)

    ctx.actions.run(
        inputs = [ctx.file.deb],
        outputs = [output_dir, mtree],
        executable = script,
        arguments = [ctx.file.deb.path, output_dir.path, mtree.path, ctx.attr.prefix],
        mnemonic = "ExtractCudaCompatLibs",
        progress_message = "Extracting CUDA compat libraries",
    )

    return [
        DefaultInfo(files = depset([output_dir])),
        OutputGroupInfo(mtree = depset([mtree])),
    ]

cuda_compat_libs = rule(
    implementation = _cuda_compat_libs_impl,
    attrs = {
        "deb": attr.label(mandatory = True, allow_single_file = True),
        "prefix": attr.string(
            mandatory = True,
            doc = "Path the libraries are archived under, e.g. usr/local/cuda/compat",
        ),
    },
    doc = "Extracts the compat/ directory out of a cuda-compat .deb",
)

def cuda_compat_tar(name, deb, prefix, **kwargs):
    """Packages a cuda-compat .deb's compat/ directory as a tar archive.

    Args:
      name: name of the resulting archive target.
      deb: the cuda-compat .deb to unpack.
      prefix: path the libraries are archived under, e.g. usr/local/cuda/compat.
      **kwargs: passed to the archive target.
    """
    cuda_compat_libs(
        name = name + "_files",
        deb = deb,
        prefix = prefix,
        tags = kwargs.get("tags"),
    )
    native.filegroup(
        name = name + "_mtree",
        srcs = [name + "_files"],
        output_group = "mtree",
        tags = kwargs.get("tags"),
    )
    tar(
        name = name,
        srcs = [name + "_files"],
        compress = "gzip",
        compressor = "@pigz//:pigz",
        compressor_args = "-n",
        mtree = name + "_mtree",
        **kwargs
    )
