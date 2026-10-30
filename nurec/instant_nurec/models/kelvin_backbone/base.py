# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from torch import nn

from nurec.instant_nurec.supervision_pack import (
    BaseInstantNuRecSupervisionPack,
    KelvinInstantNuRecSupervisionPack,
    KelvinMotionSupervision,
)


def _tokengs_init_weights(m: nn.Module):
    """Initialize weights of the transformer backbone."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@dataclass(kw_only=True, slots=True)
class KelvinLatent(ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        Get the batch size of the latent.
        """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get the device of the latent.
        """

    @property
    @abstractmethod
    def deepest(self) -> torch.Tensor:
        """
        Get the deepest feature of the latent.
        Note this is not necessarily normalized via layer norm.
        Size will be (B, V, h, w, C)
        """


@dataclass(kw_only=True, slots=True)
class KelvinFeatureLatent(KelvinLatent):
    feature: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.feature.shape[0]

    @property
    def device(self) -> torch.device:
        return self.feature.device

    @property
    def deepest(self) -> torch.Tensor:
        return self.feature


@dataclass(kw_only=True, slots=True)
class KelvinMultiscaleFeaturesLatent(KelvinLatent):
    """
    Features means transformed queries (i.e. output of the attention block).
    """

    # (B, V, h, w, C)
    features: list[torch.Tensor]

    # (B, V, n_cls_tokens, C)
    cls_tokens: list[torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        return self.features[0].shape[0]

    @property
    def device(self) -> torch.device:
        return self.features[0].device

    @property
    def deepest(self) -> torch.Tensor:
        return self.features[-1]


@dataclass(kw_only=True, slots=True)
class KelvinVGGTFeaturesLatent(KelvinMultiscaleFeaturesLatent):
    """Kelvin multiscale latent from the frozen VGGT/VGGT encoder.

    Adds the (optional) VGGT depth prediction so a decoder can consume the
    encoder's own metric depth instead of predicting it. ``encoder_depth`` is
    ``None`` until ``use_vggt_depth`` is wired (Task 6); when present its
    shape is ``(B, V, H, W, 1)`` (image resolution, not patch resolution).
    """

    # (B, V, H, W, 1) or None. Populated only when config.use_vggt_depth is True (Task 6).
    encoder_depth: torch.Tensor | None = None


# Placed after the class definitions so reading top-to-bottom shows each name defined before
# it appears in ``__all__`` (the supervision-pack types come from the re-export above).
__all__ = [
    "BaseInstantNuRecSupervisionPack",
    "KelvinFeatureLatent",
    "KelvinLatent",
    "KelvinMotionSupervision",
    "KelvinMultiscaleFeaturesLatent",
    "KelvinInstantNuRecSupervisionPack",
    "KelvinVGGTFeaturesLatent",
]
