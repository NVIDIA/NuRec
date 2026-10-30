# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nurec.config.harmonizer import HarmonizerConfig


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, True),
        ({"inference": {"use_color_transfer": False}}, False),
        ({"inference": {"use_color_transfer": True}}, True),
    ],
)
def test_harmonizer_color_transfer_supports_legacy_configs(config: dict, expected: bool) -> None:
    """Legacy model-only configs retain color transfer while explicit configs can override it."""
    assert HarmonizerConfig.model_validate(config).inference.use_color_transfer is expected
