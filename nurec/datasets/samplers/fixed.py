# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from omegaconf import DictConfig

from nurec.datasets.samplers.base import BaseFrameSampler, FrameSamplerReturn


if TYPE_CHECKING:
    from nurec.datasets.ncore import NCORETrainDataset


class FixedFrameSampler(BaseFrameSampler):
    """Always sample one configured local index from each sensor frame range."""

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset) -> None:
        del dataset
        self.frame_index = int(config.frame_index)
        assert self.frame_index >= 0

    def sample_frame(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        unique_sensor_id: str,
    ) -> FrameSamplerReturn:
        del rng, batch_idx, unique_sensor_id
        if self.frame_index >= len(frame_range):
            raise IndexError(
                f"Fixed local frame index {self.frame_index} is outside a frame range of length {len(frame_range)}"
            )
        return FrameSamplerReturn(sampled_frame_idx=frame_range[self.frame_index])

    @classmethod
    def training_frame_mask(cls, config: DictConfig, n_frames: int) -> npt.NDArray[np.bool_]:
        """Exactly the one configured frame — mirroring ``sample_frame``,
        which raises for ranges the index does not fit."""
        frame_index = int(config["frame_index"])  # config may be a raw dict here
        # Reject negatives here too: __init__ asserts frame_index >= 0, so a
        # negative index that Python would wrap to the last frame here would
        # make the mask claim a training frame the runtime sampler rejects.
        if frame_index < 0:
            raise ValueError(f"Fixed local frame index must be non-negative, got {frame_index}")
        if frame_index >= n_frames:
            raise IndexError(f"Fixed local frame index {frame_index} is outside a frame range of length {n_frames}")
        mask = np.zeros(n_frames, dtype=np.bool_)
        mask[frame_index] = True
        return mask


BaseFrameSampler.register_to_frame_sampler_factory("fixed", FixedFrameSampler)
