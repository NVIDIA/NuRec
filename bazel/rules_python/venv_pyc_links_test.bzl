# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests the `__pycache__` links that rules_python.patch adds to the venv link map.

rules_python's own `build_link_map` tests need `rules_testing`, a dev dependency
of rules_python that is not visible from this module, so the patched behaviour
is covered here instead.
"""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("@rules_python//python:py_info.bzl", "VenvSymlinkEntry", "VenvSymlinkKind")
load("@rules_python//python/private:venv_runfiles.bzl", "build_link_map")  # buildifier: disable=bzl-visibility

# `build_link_map` only reads the workspace name of its ctx.
_CTX = struct(workspace_name = "_main")

def _file(path):
    """A stand-in for a `File` of an external repository (`<repo>/<path in repo>`)."""
    return struct(short_path = "../" + path, basename = path.rsplit("/", 1)[-1])

def _entry(venv_path, link_to_path, files, link_to_file = None):
    return VenvSymlinkEntry(
        kind = VenvSymlinkKind.LIB,
        venv_path = venv_path,
        link_to_path = link_to_path,
        link_to_file = link_to_file,
        files = depset(files),
        package = None,
        version = None,
    )

def _venv_pyc_links_test_impl(ctx):
    env = unittest.begin(ctx)

    a = "+pypi_a/site-packages"
    b = "+pypi_b/site-packages"
    c = "+pypi_c/site-packages"
    a_common = _file(a + "/shared/common.py")
    a_common_pyc = _file(a + "/shared/__pycache__/common.cpython-313.pyc")
    a_z_pyc = _file(a + "/shared/__pycache__/z.cpython-313.pyc")
    b_a_pyc = _file(b + "/shared/__pycache__/a.cpython-313.pyc")
    b_common_pyc = _file(b + "/shared/__pycache__/common.cpython-313.pyc")
    c_ext = _file(c + "/solo/ext.so")

    # Wheels a and b both install modules into `shared/`, and both provide
    # `common.py`: a's is selected (first added wins). Wheel c alone installs
    # into `solo/`, file by file because of the shared library, and provides
    # `nopyc/n.py`, which has no bytecode in the runfiles.
    entries = [
        _entry("shared/common.py", a + "/shared/common.py", [a_common]),
        _entry("shared/z.py", a + "/shared/z.py", [_file(a + "/shared/z.py")]),
        _entry("shared/a.py", b + "/shared/a.py", [_file(b + "/shared/a.py")]),
        _entry("shared/common.py", b + "/shared/common.py", [_file(b + "/shared/common.py")]),
        _entry("solo/m.py", c + "/solo/m.py", [_file(c + "/solo/m.py")]),
        _entry("solo/ext.so", c + "/solo/ext.so", [c_ext], link_to_file = c_ext),
        _entry("nopyc/n.py", c + "/nopyc/n.py", [_file(c + "/nopyc/n.py")]),
    ]
    pyc_index = {
        a + "/shared/__pycache__": [a_common_pyc, a_z_pyc],
        b + "/shared/__pycache__": [b_a_pyc, b_common_pyc],
        c + "/solo/__pycache__": [_file(c + "/solo/__pycache__/m.cpython-313.pyc")],
    }

    actual = build_link_map(_CTX, entries, pyc_index = pyc_index)

    asserts.equals(
        env,
        {
            # Two wheels feed `shared/`: each selected `.py` exposes the pyc of
            # its own wheel, one by one, so b's `common` bytecode never stands
            # in for a's `common.py`, and no `__pycache__` directory is linked.
            "shared/a.py": b + "/shared/a.py",
            "shared/common.py": a_common,
            "shared/z.py": a + "/shared/z.py",
            "shared/__pycache__/a.cpython-313.pyc": b_a_pyc,
            "shared/__pycache__/common.cpython-313.pyc": a_common_pyc,
            "shared/__pycache__/z.cpython-313.pyc": a_z_pyc,
            # One wheel feeds `solo/`: its `__pycache__` is linked as one directory.
            "solo/__pycache__": c + "/solo/__pycache__",
            "solo/ext.so": c_ext,
            "solo/m.py": c + "/solo/m.py",
            # No bytecode in the runfiles, no link.
            "nopyc/n.py": c + "/nopyc/n.py",
        },
        actual[VenvSymlinkKind.LIB],
    )

    return unittest.end(env)

venv_pyc_links_test = unittest.make(_venv_pyc_links_test_impl)
