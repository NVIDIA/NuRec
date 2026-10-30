# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from nurec.models.gaussians.gaussians_model import BaseGaussianModel
    from nurec.models.nn_extensions import TypedModuleDict


class SelectedGaussianNodes:
    def __init__(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel], exclude_layer_ids: list[str]):
        self.gaussians_nodes = gaussians_nodes
        self.layer_ids = sorted(
            [
                layer_id
                for layer_id in gaussians_nodes.keys()
                if layer_id not in exclude_layer_ids and gaussians_nodes[layer_id].get_num_gaussians() > 0
            ]
        )

        layer_offset = 0
        self.layer_offsets: dict[str, int] = {}
        self.layer_num_gaussians: dict[str, int] = {}
        for layer_id in self.layer_ids:
            self.layer_offsets[layer_id] = layer_offset
            self.layer_num_gaussians[layer_id] = self.gaussians_nodes[layer_id].get_num_gaussians()
            layer_offset += self.layer_num_gaussians[layer_id]

    def get_densities(self) -> torch.Tensor:
        # Return empty tensor if no layers to concatenate
        if not self.layer_ids:
            return torch.empty(0)
        densities = torch.cat([self.gaussians_nodes[layer_id].get_densities() for layer_id in self.layer_ids])
        return densities
