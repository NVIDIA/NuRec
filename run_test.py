# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lazily loaded command group of the `run` entrypoint."""

import runpy

from pathlib import Path

import click
import pytest


_run = runpy.run_path(str(Path(__file__).with_name("run.py")))
cli = _run["cli"]
LazyGroup = _run["LazyGroup"]
LAZY_COMMANDS: dict[str, str] = _run["LAZY_COMMANDS"]


def test_every_lazy_command_is_listed_before_it_is_loaded() -> None:
    assert set(LAZY_COMMANDS) <= set(cli.list_commands(click.Context(cli)))


@pytest.mark.parametrize("name", sorted(LAZY_COMMANDS))
def test_each_lazy_command_loads_under_its_own_name(name: str) -> None:
    command = cli.get_command(click.Context(cli), name)
    assert command is not None
    assert command.name == name


def test_a_command_added_explicitly_wins_over_its_lazy_entry() -> None:
    """A composition root may install its own variant of a public export command under the public name."""
    group = LazyGroup(lazy_commands={"export-ego-mask": LAZY_COMMANDS["export-ego-mask"]})
    override = click.Command("export-ego-mask")
    group.add_command(override)
    assert group.get_command(click.Context(group), "export-ego-mask") is override


def test_a_command_added_explicitly_is_listed_and_found() -> None:
    group = LazyGroup(lazy_commands=LAZY_COMMANDS)
    command = click.Command("added-at-runtime")
    group.add_command(command)
    assert "added-at-runtime" in group.list_commands(click.Context(group))
    assert group.get_command(click.Context(group), "added-at-runtime") is command


def test_an_unknown_first_argument_loads_main_and_routes_to_it() -> None:
    group = LazyGroup(lazy_commands={"main": LAZY_COMMANDS["main"]}, default="main", default_if_no_args=True)
    assert "main" not in group.commands
    cmd_name, command, args = group.resolve_command(click.Context(group), ["--config-name=some.yaml", "mode=train"])
    assert cmd_name == "main"
    assert command is not None and command.name == "main"
    assert args == ["--config-name=some.yaml", "mode=train"]
