# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that Bazel packages the Python wrapper and selected native binding."""

import os

import nurec.libs.tracy_utils as tracy

from nurec.libs.tracy_utils import interface


def test_packaged_tracy_import():
    expected_binding = os.environ["EXPECTED_TRACY_BINDING"]
    binding = interface.tracy_utils_py
    assert binding is not None
    assert binding.__name__ == f"nurec.libs.tracy_utils.{expected_binding}"
    assert tracy.is_available() == (expected_binding != "tracy_utils_disabled_py")
