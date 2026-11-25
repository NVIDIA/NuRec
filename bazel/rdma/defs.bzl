# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetically vendor libibverbs (rdma-core) for the optional RDMA transport.

//nurec/libs/rdma_transport compiles against <infiniband/verbs.h> and links
libibverbs. Rather than requiring a system libibverbs-dev install -- which the
build/CI images deliberately lack -- ibverbs_repo unpacks the Ubuntu rdma-core
and libnl .debs that MODULE.bazel fetches as http_file and exposes:

  :ibverbs       headers (from libibverbs-dev) + the versioned libibverbs.so
                 (from libibverbs1), as a cc_library;
  :runtime_libs  everything the runtime .debs install under
                 usr/lib/x86_64-linux-gnu and etc/libibverbs.d, plus each
                 package's copyright file, for the run image layer
                 (//bazel/rdma:runtime_libs_tar).

Unpacking goes through repository_ctx.extract, which reads the .deb ar container
and its zstd-compressed members inside Bazel, so no host ar/tar/zstd is needed.

Only infiniband/ headers are vendored: the <rdma/...> kernel UAPI headers that
verbs.h includes come from the base image's linux-libc-dev.
"""

_BUILD_CONTENT = """\
load("@rules_cc//cc:defs.bzl", "cc_library")

# The glob picks the real ELF file (libibverbs.so.1.<abi>.<rdma-core version>), not
# the libibverbs.so.1 symlink beside it; linking it records DT_NEEDED against its
# soname, libibverbs.so.1, which the run image layer and host MOFED both provide.
cc_library(
    name = "ibverbs",
    srcs = glob(["usr/lib/x86_64-linux-gnu/libibverbs.so.*.*.*"], allow_empty = False),
    hdrs = glob(["dev/usr/include/infiniband/*.h"], allow_empty = False),
    includes = ["dev/usr/include"],
    target_compatible_with = ["@platforms//cpu:x86_64"],
    visibility = ["//visibility:public"],
)

# Each package's copyright file ships with its libraries, at usr/share/doc/<package>/copyright
# like every Ubuntu package in the image.
filegroup(
    name = "runtime_libs",
    srcs = glob(
        ["usr/lib/x86_64-linux-gnu/**", "etc/libibverbs.d/**", "usr/share/doc/*/copyright"],
        allow_empty = False,
    ),
    target_compatible_with = ["@platforms//cpu:x86_64"],
    visibility = ["//visibility:public"],
)
"""

# Only amd64 packages are pinned. Declaring the targets incompatible, rather than
# failing the fetch, keeps `bazel query //...` loadable on other hosts: //:format.check
# walks every package, so a fail() here would break unrelated aarch64 jobs.
_UNSUPPORTED_BUILD_CONTENT = """\
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "ibverbs",
    target_compatible_with = ["@platforms//cpu:x86_64"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "runtime_libs",
    target_compatible_with = ["@platforms//cpu:x86_64"],
    visibility = ["//visibility:public"],
)
"""

def _extract_deb(repository_ctx, deb, output):
    """Unpacks the files a .deb installs (its data.tar.* member) under output."""
    staging = "_deb"
    repository_ctx.extract(archive = repository_ctx.path(deb), output = staging)
    data = [f for f in repository_ctx.path(staging).readdir() if f.basename.startswith("data.tar.")]
    if len(data) != 1:
        fail("expected exactly one data.tar.* member in {}".format(deb))
    repository_ctx.extract(archive = data[0], output = output)
    repository_ctx.delete(staging)

def _ibverbs_repo_impl(repository_ctx):
    if repository_ctx.os.arch not in ("amd64", "x86_64"):
        repository_ctx.file("BUILD.bazel", _UNSUPPORTED_BUILD_CONTENT)
        return

    for deb in repository_ctx.attr.runtime_debs:
        _extract_deb(repository_ctx, deb, ".")

    # Kept apart so only its headers are exported: the globs above never reach dev/.
    _extract_deb(repository_ctx, repository_ctx.attr.dev_deb, "dev")

    repository_ctx.file("BUILD.bazel", _BUILD_CONTENT)

ibverbs_repo = repository_rule(
    implementation = _ibverbs_repo_impl,
    attrs = {
        "dev_deb": attr.label(
            mandatory = True,
            doc = "The libibverbs-dev .deb file (headers).",
        ),
        "runtime_debs": attr.label_list(
            mandatory = True,
            doc = "The runtime .deb files: libibverbs1 (the shared object linked against), plus " +
                  "everything the run image needs at load time (providers, libnl).",
        ),
    },
    doc = "Unpacks the pinned rdma-core/libnl .debs into a hermetic cc_library and runtime file set.",
)
