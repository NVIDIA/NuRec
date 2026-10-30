# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import ConfigDict

from nurec.config.base_schema import BaseConfigSchema, Field


class NovelViewPoseConfig(BaseConfigSchema):
    """Configuration for a single novel view pose with translation and rotation deltas."""

    translation: list[float] | None = None
    rotation: list[float] | None = None


class PSchedulerConfig(BaseConfigSchema):
    """Probability scheduler configuration for training-time harmonizer."""

    p_init: float = 0.5
    p_final: float | None = None
    milestones: list[int] = Field(default_factory=list)
    update_interval: Literal["step", "epoch"] = "step"
    gamma: float = 0.5


class TrainingHarmonizerConfig(BaseConfigSchema):
    """Configuration for training-time harmonizer."""

    enabled: bool = False
    # Training is non-temporal only. This field exists so a config that explicitly
    # requests temporal training is rejected loudly by TrainingHarmonizerController
    # rather than silently ignored. Training supports non-temporal only for now.
    temporal: bool = False
    p_scheduler: PSchedulerConfig = Field(default_factory=PSchedulerConfig)
    start_step: int = 0
    use_color_transfer: bool = True
    num_workers: int = 0
    novel_view_poses: list[NovelViewPoseConfig] = Field(default_factory=list)
    shuffle_novel_views: bool = False
    log_images: bool = False
    log_every_n_steps: int = 100
    log_buffer_images: bool = False
    debug_dir: str | None = None


class InferenceHarmonizerConfig(BaseConfigSchema):
    """Configuration for inference-time harmonizer."""

    enabled: bool = False
    use_color_transfer: bool = True


class HarmonizerModelConfig(BaseConfigSchema):
    """Configuration for the harmonizer model."""

    name: str = "harmonizer-nontemporal"
    cache_dir: str = "~/.cache/nurec/harmonizer"
    # Set model config to avoid pydantic warnings on protected namespaces, i.e. below class fields starting with model_
    model_config = ConfigDict(protected_namespaces=())
    model_url: str = ""
    model_resolution: tuple[int, int] = (576, 1024)
    model_filename: str = ""
    cosmos_base_model_url: str | None = None
    cosmos_tokenizer_url: str | None = None
    cosmos_cache_dir: str | None = None


class HarmonizerConfig(HarmonizerModelConfig):
    """Main harmonizer configuration with model parameters and training/inference settings."""

    training: TrainingHarmonizerConfig = Field(default_factory=TrainingHarmonizerConfig)
    inference: InferenceHarmonizerConfig = Field(default_factory=InferenceHarmonizerConfig)
