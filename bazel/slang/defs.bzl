# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Precompile a Slang module into a torch C++ CUDA extension """

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

def slangtorch_library(name, srcs, extra_cmd_args = "", deps = [], use_fast_math = True, **kwargs):
    """
    Precompile a Slang module into a torch C++ CUDA extension

    Args:
        name: The name of the target.
        srcs: The list of slang source files to compile. If not provided, the slang module name is used.
        extra_cmd_args: Additional arguments to pass to the slangc command.
        deps: The list of dependencies to add to the target additionaly to the srcs.
        use_fast_math: Whether to use fast math for the slang module.
        **kwargs: Additional arguments to pass to all generated targets.
    """

    # CUDA module source generation
    module_src_str = " ".join(["$(location " + s + ")" for s in srcs])
    slangc_cmd_base = "$(location //deps/slang:slangc) " + extra_cmd_args + " " + module_src_str
    slangc_cmd_coverage = slangc_cmd_base + " -g -line-directive-mode standard -O0 -o $@"
    slangc_cmd_prod = slangc_cmd_base + " -O3 -o $@"

    # slangc embeds an absolute path to slang-cuda-prelude.h in the generated .cu
    # output (resolved from the slangc binary's install location). This path includes
    # the Bazel repository-cache directory counter (e.g. .../contents/<hash>/2/include/...)
    # which can change across fetches/clean builds, causing stale action-cache hits to
    # reference a non-existent directory. We post-process the output to replace the
    # absolute #include with a portable one; the cuda_library then resolves the header
    # via //deps/slang:slang_headers on its include path.
    sed_fix_prelude = " && sed -i 's|#include \".*/slang-cuda-prelude\\.h\"|#include \"slang-cuda-prelude.h\"|' $@"

    combined_deps = deps + srcs

    prefix_name = name
    if name.endswith("_cc"):
        prefix_name = name[:-3]

    # CUDA module
    native.genrule(
        name = prefix_name + "_cuda_src",
        srcs = combined_deps,
        outs = [prefix_name + "_cuda.cu"],
        cmd = select({
            "//bazel/conditions:coverage_mode": slangc_cmd_coverage + " -target cuda" + sed_fix_prelude,
            "//conditions:default": slangc_cmd_prod + " -target cuda" + sed_fix_prelude,
        }),
        tools = ["//deps/slang:slangc"],
        **kwargs
    )

    cuda_library(
        name = prefix_name + "_cuda",
        srcs = [":" + prefix_name + "_cuda_src"],
        copts = [
            # Flags to be consistent with torch-cuda-extensions.
            # __CUDA_NO_HALF_OPERATORS__ and __CUDA_NO_HALF_CONVERSIONS__ are intentionally NOT set to allow fp16
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            # Enable Slang's fp16 prelude so generated CUDA code can resolve __half / half.
            "-DSLANG_CUDA_ENABLE_HALF=1",
            "-w",
            "--generate-line-info",
        ] + (["--use_fast_math"] if use_fast_math else []),
        deps = [
            "//deps/slang:slang_headers",
            "//deps/torch:torch_headers",
            "@rules_python//python/cc:current_py_cc_headers",
        ],
        **kwargs
    )

    # Torch C++ binding
    native.genrule(
        name = prefix_name + "_src",
        srcs = combined_deps,
        outs = [prefix_name + ".cpp"],
        cmd = select({
            "//bazel/conditions:coverage_mode": slangc_cmd_coverage + " -target torch-binding",
            "//conditions:default": slangc_cmd_prod + " -target torch-binding",
        }),
        tools = ["//deps/slang:slangc"],
        **kwargs
    )

    cc_binary(
        name = name,
        srcs = [":" + prefix_name + "_src"],
        copts = [
            # Flags to be consistent with torch-cuda-extensions and to build correct torch-extension bindings
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            "-DTORCH_EXTENSION_NAME=lib" + name,
            "-w",
        ] + select({
            # Slang generates `typedef _Float16 half;` which requires gcc-13+ on aarch64.
            # Map _Float16 to the supported __fp16 type for older compilers.
            "@platforms//cpu:aarch64": ["-D_Float16=__fp16"],
            "//conditions:default": [],
        }),
        linkopts = [
            "-Wl,--no-undefined",
            "-Wl,--as-needed",
        ],
        linkshared = True,
        deps = [
            ":" + prefix_name + "_cuda",
            "@nurec_pip_deps//torch:libc10",
            "@nurec_pip_deps//torch:libc10_cuda",
            "@nurec_pip_deps//torch:libtorch_cpu",
            "@nurec_pip_deps//torch:libtorch_python",
            "//deps/torch:torch_headers",
            "@rules_python//python/cc:current_py_cc_libs",
        ],
        **kwargs
    )
