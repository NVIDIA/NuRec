# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nurec.models.nn_extensions import TypedModuleDict


class DummyGaussian(BaseGaussianModel):
    def __init__(self, num_gaussians, densities):
        self._num = num_gaussians
        self._densities = densities
        self.optimizers = []  # no optimizers for this dummy

    def get_num_gaussians(self):
        return self._num

    def get_densities(self, preactivation=False):
        return self._densities

    def density_activation_inv(self, x):
        return x

    def scale_activation_inv(self, x):
        return x


def test_zero_layers():
    """Verify empty Gaussian layers are excluded without producing bogus tensors.

    Relocation and addition logic concatenate selected layers, so zero-sized layers must not
    create fake density entries.
    """
    nodes = cast(
        TypedModuleDict[BaseGaussianModel],
        TypedModuleDict({"layer0": DummyGaussian(0, torch.tensor([]))}),
    )
    sel = SelectedGaussianNodes(nodes, exclude_layer_ids=[])
    # No layers should be included
    assert sel.layer_ids == []
    # Densities should be empty
    dens = sel.get_densities()
    assert dens.numel() == 0


def test_mixed_layers():
    """Verify selection keeps non-empty layers and preserves their density values.

    This protects the layer filtering used by MCMC when some Gaussian nodes currently have no
    particles but others should still participate in sampling.
    """
    d0 = DummyGaussian(0, torch.tensor([]))
    d1 = DummyGaussian(2, torch.tensor([1.0, 2.0]))
    nodes = cast(TypedModuleDict[BaseGaussianModel], TypedModuleDict({"zero": d0, "nonzero": d1}))
    sel = SelectedGaussianNodes(nodes, exclude_layer_ids=[])
    # Only 'nonzero' should be present
    assert sel.layer_ids == ["nonzero"]
    assert torch.allclose(sel.get_densities(), torch.tensor([1.0, 2.0]))
