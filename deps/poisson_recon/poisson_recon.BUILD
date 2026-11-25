# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:defs.bzl", "cc_binary")

# The Makefile's flags, minus -Werror=strict-aliasing: MyAtomic.h type-puns its atomics, so the
# build disables strict aliasing instead.
cc_binary(
    name = "poisson_recon",
    srcs = ["Src/PoissonRecon.cpp"] + glob([
        "Src/*.h",
        "Src/*.inl",
    ]),
    copts = [
        "-DRELEASE",
        "-O3",
        "-fno-strict-aliasing",
        "-fopenmp",
        "-funroll-loops",
        "-ffast-math",
        "-w",
    ] + select({
        # SANITIZED_PR makes the level-set extraction use atomics for its cell, edge and face flags. Without
        # it, multi-threaded runs on aarch64 hit a data race and abort with "Failed to close loop"
        # (https://github.com/mkazhdan/PoissonRecon/issues/190).
        "@platforms//cpu:aarch64": ["-DSANITIZED_PR"],
        "//conditions:default": [],
    }),
    includes = ["."],
    linkopts = ["-fopenmp"],
    visibility = ["@nurec_repo//nurec/utils/io:__pkg__"],
)
