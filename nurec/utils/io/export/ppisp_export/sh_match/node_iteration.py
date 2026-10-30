# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-node SH-fit dispatch helper shared by the SH-match modules.

This module exists as a deliberate cycle breaker between
:mod:`nurec.utils.io.export.ppisp_export.sh_match.optimization` and
:mod:`nurec.utils.io.export.ppisp_export.sh_match.renderer_callable`.
Both modules call :func:`iter_gaussian_nodes` to walk the per-node
SH-fit surface, and ``renderer_callable`` already imports the view
context (:class:`TrainingViewContext`) from ``optimization`` at module
load time. Hosting :func:`iter_gaussian_nodes` here keeps that helper
on a leaf module with no siblings as imports, so neither
``renderer_callable`` nor ``optimization`` has to reach into the other
just to dispatch over gaussian nodes.

Keeping the helper here (rather than in ``renderer_callable`` with
``optimization`` doing local imports to defer it) avoids a sibling
import cycle between the two modules.
"""

from __future__ import annotations

from typing import List, Tuple


SINGLE_NODE_FALLBACK_KEY: str = "gaussians"
"""Name used for the single-node fallback in :func:`iter_gaussian_nodes`.

Mirrors the literal used by :mod:`nurec.utils.io.export.gaussian_usd_asset`
when the input model is a single ``SHGaussianModel`` rather than a
``GaussiansComposite``. Keeping the fallback key stable here lets test
fakes and production code identify the same single-node bucket.
"""


def iter_gaussian_nodes(model: object) -> List[Tuple[str, object]]:
    """Iterate ``(node_id, gaussian_node)`` pairs over a gaussian model.

    Returns ``list(model.gaussians_nodes.items())`` when the input is a
    :class:`nurec.models.gaussians.gaussians_composite.GaussiansComposite`
    (or any object exposing a truthy ``gaussians_nodes`` mapping with an
    ``items()`` method). Falls back to a single-node mapping
    ``[(SINGLE_NODE_FALLBACK_KEY, model)]`` for the historical
    single-node ``SHGaussianModel`` path that tests and a handful of
    legacy call sites still pass directly.

    The dispatch mirrors the one already used by
    :mod:`nurec.utils.io.export.gaussian_usd_asset` at the node-export
    loop. Factored here so every consumer of the per-node SH-fit
    surface (warm-start, optimizer setup, snapshot, render callables)
    routes through the same rule.

    Args:
        model: Either a :class:`GaussiansComposite` (production) or a
            single SH gaussian node (legacy / tests).

    Returns:
        Materialised list of ``(node_id, gaussian_node)`` tuples.
        Materialisation keeps the iteration order stable across
        repeated calls and avoids surprising behaviour when callers
        mutate the underlying mapping mid-iteration (e.g. the
        optimizer's freeze/unfreeze passes).
    """
    nodes = getattr(model, "gaussians_nodes", None)
    if nodes and hasattr(nodes, "items"):
        return list(nodes.items())
    return [(SINGLE_NODE_FALLBACK_KEY, model)]
