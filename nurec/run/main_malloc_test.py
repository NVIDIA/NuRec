# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

from unittest.mock import Mock

import pytest


# nurec.run re-exports each command under its module's name, so fetch the modules through importlib.
main_module = importlib.import_module("nurec.run.main")
profile_dataloader_module = importlib.import_module("nurec.run.profile_dataloader")


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "module, command, next_step, kwargs",
    [
        (main_module, main_module.main, "assert_no_out_dir_override_in_resume", {}),
        (
            profile_dataloader_module,
            profile_dataloader_module.profile_dataloader,
            "parse_typed_config",
            {"cprofiler_output": None},
        ),
    ],
)
def test_entrypoint_tunes_malloc_before_anything_else(monkeypatch, module, command, next_step, kwargs):
    calls = Mock()
    calls.next_step.side_effect = _Stop
    monkeypatch.setattr(module, "tune_glibc_malloc", calls.tune)
    monkeypatch.setattr(module, next_step, calls.next_step)

    with pytest.raises(_Stop):
        command.callback(config_name="x", hydra_args=[], **kwargs)

    assert [name for name, _, _ in calls.mock_calls] == ["tune", "next_step"]
