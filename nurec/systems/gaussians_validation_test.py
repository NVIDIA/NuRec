# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for distributed validation behavior in ``GaussiansSystem``."""

from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch

from nurec.systems.gaussians import GaussiansSystem


def _bare_system() -> GaussiansSystem:
    system = object.__new__(GaussiansSystem)
    torch.nn.Module.__init__(system)
    system.cpu()
    return system


def test_camera_rank_reports_camera_when_another_rank_has_lidar() -> None:
    system = _bare_system()
    batch: Any = SimpleNamespace(data=SimpleNamespace(camera=object()))

    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_world_size", return_value=2),
        mock.patch("torch.distributed.all_reduce") as all_reduce,
    ):
        assert system._any_rank_has_camera(batch)

    all_reduce.assert_called_once()


def test_lidar_rank_reports_camera_when_another_rank_has_camera() -> None:
    system = _bare_system()
    batch: Any = SimpleNamespace(data=SimpleNamespace(camera=None))

    def report_camera_peer(has_camera: torch.Tensor, **_kwargs: Any) -> None:
        has_camera.fill_(1)

    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_world_size", return_value=2),
        mock.patch("torch.distributed.all_reduce", side_effect=report_camera_peer) as all_reduce,
    ):
        assert system._any_rank_has_camera(batch)

    all_reduce.assert_called_once()


def test_lidar_rank_runs_selected_node_forwards_when_camera_peer_exists() -> None:
    system = _bare_system()
    system.val_out_types = ["rgb"]
    system_config: Any = SimpleNamespace(
        test=SimpleNamespace(
            val_render_selected_nodes={
                "road_only": ["road"],
                "background_only": ["background"],
            }
        )
    )
    system.config = system_config
    system.forward = mock.MagicMock(return_value=SimpleNamespace(rendered_cam=None))
    batch: Any = SimpleNamespace(data=SimpleNamespace(camera=None))

    with mock.patch.object(system, "_any_rank_has_camera", return_value=True):
        system.validation_selected_nodes_step(batch)

    assert system.forward.call_count == 2
