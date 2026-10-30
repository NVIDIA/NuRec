# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("@nurec_pip_deps//:requirements.bzl", pip_requirement = "requirement")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "pylib",
    srcs = [
        "src/pix2pix_turbo_harmonizer.py",
    ] + glob([
        "src/model/**/*.py",
        "src/utils/**/*.py",
    ]),
    imports = ["src"],
    visibility = ["//visibility:public"],
    deps = [
        pip_requirement("cosmos-predict2"),
        pip_requirement("diffusers"),
        pip_requirement("einops"),
        pip_requirement("flash-attn"),
        pip_requirement("numpy"),
        pip_requirement("omegaconf"),
        pip_requirement("packaging"),
        pip_requirement("pillow"),
        pip_requirement("requests"),
        pip_requirement("torch"),
        pip_requirement("torchvision"),
        pip_requirement("tqdm"),
        pip_requirement("transformer_engine"),
    ],
)
