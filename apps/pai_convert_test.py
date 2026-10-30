# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PAI -> NCore converter wired into nurec_tools."""

from pathlib import Path

from python.runfiles import runfiles


def test_imports():
    """Verify the converter CLI and its external dependencies can be imported."""
    import tools.data_converter.pai.converter as converter  # noqa: PLC0415

    assert converter.cli is not None
    assert {"pai-v4", "pai-stream-v4"} <= set(converter.cli.commands)


def test_entrypoint_registers_unconditional_tools():
    """The generated entrypoint only registers tools named in --enabled-tools, so a tools.yaml
    entry alone is silently dropped. Guard the tools that are enabled in every flavour."""
    r = runfiles.Create()
    resolved = r.Rlocation("_main/apps/entrypoint.py", r.CurrentRepository())
    assert resolved is not None, "generated entrypoint not in runfiles"
    text = Path(resolved).read_text()
    for tool in ("pai-convert", "pai-clip-dl"):
        assert f'name="{tool}"' in text, f"{tool} missing from the generated entrypoint"
