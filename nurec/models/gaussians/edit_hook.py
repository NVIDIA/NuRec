# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model layer's contract for a render-time collected-parameter override hook.

``GaussiansComposite`` can carry one hook that mutates the collected Gaussian
parameters just before rasterization. This module owns BOTH halves of that contract —
the protocol the hook satisfies and the layout record the composite hands it — so the
dependency runs from the implementation TO the model layer, never the other way. The
only implementation today is ``nurec.asset_edits.runtime.AspectOverrideRuntime``
(controllable SimReady assets); it satisfies :class:`GaussianEditOverrides`
structurally, so nothing here imports it and nothing here knows what an "asset edit" is.

Deliberately torch-only: both the composite and any implementation must be able to
depend on this without inheriting either side's dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, runtime_checkable

import torch


@dataclass
class CollectedLayerLayout:
    """Where ONE Gaussian layer landed in ONE sensor's collected tensors.

    Recorded by the composite during collect and handed to
    :meth:`GaussianEditOverrides.apply`, so a hook addressing baked row indices can map
    them onto the rows that actually got collected this frame.

    Index spaces (the whole point of this record):

    * ``layer_offset``/``layer_size`` are in the COLLECTED tensor: the layer occupies
      rows ``[layer_offset, layer_offset + layer_size)``.
    * ``kept_rows`` is in the layer's own UNFILTERED row space, which under distributed
      rendering is the ALL-GATHERED layer, not the rank-local one — it must be derived
      from the gathered parameters, or the remap misaddresses rows on multi-GPU only.
    """

    layer_offset: int
    layer_size: int
    layer_id: str = ""  # diagnostics only
    # Under edited-actors collection the layer is row-FILTERED: ``kept_rows[i]`` is the
    # unfiltered row of collected row ``i``. STRICTLY INCREASING (it is produced as
    # ``arange(n)[mask]``), which lets a consumer invert it with a ``searchsorted``
    # instead of a full-size lookup table. ``None`` = unfiltered (rows map 1:1).
    kept_rows: Optional[torch.Tensor] = None
    # The collector's own per-COLLECTED-ROW visibility mask (pose interpolation /
    # track-lifetime masking). A hook must never re-show a row the collect masked out.
    # ``None`` = nothing masked.
    collect_keep: Optional[torch.Tensor] = None


@runtime_checkable
class GaussianEditOverrides(Protocol):
    """A render-time hook that mutates one frame's collected Gaussian parameters."""

    def apply(
        self,
        gaussian_parameters: dict[str, torch.Tensor],
        frame_timestamp_us: int,
        layouts: Mapping[str, CollectedLayerLayout],
    ) -> None:
        """Mutate ``gaussian_parameters`` in place for this frame.

        ``layouts`` is what this sensor's collect recorded, keyed by layer id.
        """
