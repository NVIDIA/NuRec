# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_lightning.plugins.precision.amp import MixedPrecision

from nurec.run.main import MixedPrecisionNoCastPlugin


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GradScaler disables itself without CUDA and reports a scale of 1.0, hiding the init_scale contract",
)
def test_no_cast_plugin_keeps_grad_scaler_but_skips_global_autocast():
    """The 16-no-cast plugin is Lightning's MixedPrecision with a small-init-scale scaler and no autocast."""
    plugin = MixedPrecisionNoCastPlugin("16-mixed", "cuda")

    assert isinstance(plugin, MixedPrecision)
    assert plugin.scaler is not None
    assert plugin.scaler.get_scale() == 128.0

    with plugin.forward_context():
        assert not torch.is_autocast_enabled()
