# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-PPISP contract test for controller weight export.

The default ``weights_test`` target stays hermetic and CPU-only. This
target imports the real ``ppisp`` package and constructs a real
controller, so it is split into a separate manual Bazel target.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from nurec.utils.io.export.ppisp_export.controller.weights import (
    EXPECTED_CONTROLLER_WEIGHTS_LEN,
    flatten_controller_weights,
    select_camera_controller,
    validate_controller_architecture,
)


@unittest.skipUnless(torch.cuda.is_available(), "real PPISP requires CUDA")
class RealPPISPControllerRoundTripTest(unittest.TestCase):
    """Locks the contract against the real ``ppisp._PPISPController``."""

    def test_real_controller_validates_and_flattens(self) -> None:
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        config = PPISPConfig(use_controller=True)
        ppisp = PPISP(num_cameras=2, num_frames=4, config=config)
        controller = select_camera_controller(ppisp, 1)
        validate_controller_architecture(controller)
        flat = flatten_controller_weights(controller)
        self.assertEqual(flat.dtype, np.float32)
        self.assertEqual(flat.size, EXPECTED_CONTROLLER_WEIGHTS_LEN)


if __name__ == "__main__":
    unittest.main()
