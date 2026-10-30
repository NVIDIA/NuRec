# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

import numpy as np


class DepthEstimator(Protocol):
    max_depth_m: float
    method: str

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict from an RGB image and return one depth image with shape (1, H, W)."""
        ...
