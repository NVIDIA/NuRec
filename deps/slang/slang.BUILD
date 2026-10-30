# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Platform-agnostic slang targets imported from external repo
# (supported by both x86_64 and aarch64 builds, aliased in BUILD.bazel)

cc_library(
    name = 'slang_headers',
    includes = ['include'],
    hdrs = glob(['include/*.h']),
)

# Glob the versioned .so so this file works for any pinned slang version (one per archive).
cc_import(
    name = 'lib_slang_compiler',
    shared_library = glob(["lib/libslang-compiler.so.0.*"], exclude = ["lib/*.dwarf"], allow_empty = False)[0],
)

cc_import(
    name = 'lib_slang_rtc',
    shared_library = glob(["lib/libslang-rt.so.0.*"], exclude = ["lib/*.dwarf"], allow_empty = False)[0],
)

filegroup(
    name = 'slangc',
    srcs = ['bin/slangc'],
)

# slangc plus its libs in the bin/ + lib/ layout so its RUNPATH ($ORIGIN/../lib)
# resolves in a container runfiles tree (shipping bin/slangc alone leaves lib/
# absent). Add to data deps of any target that JITs .slang at runtime.
filegroup(
    name = 'slangc_runfiles',
    srcs = ['bin/slangc'] + glob(['lib/*.so', 'lib/*.so.*'], exclude = ['lib/*.dwarf']),
)
