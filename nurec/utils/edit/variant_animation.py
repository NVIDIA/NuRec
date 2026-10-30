# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extraction of variant-state animation tables from SimReady assets.

An asset's animated behaviors are authored as variantSets on its root prim
(e.g. ``appearanceMode`` with states ``off`` / ``alternate`` / ``sequence``). An animated
state carries element prims with standard time-sampled USD — ``visibility`` for
visibility changes, plus optional continuous gain attributes::

    custom float nurec:gain      # gain timeSamples over the cycle (lerped)
    custom color3f nurec:coldRgb # reference color at gain 0

:func:`read_variant_animation` selects each state in turn (in-memory only) and
extracts every animated element's visibility on-windows and gain curve, normalized
over one cycle (the stage's timeCode range). The compiler matches baked Gaussian
parts to elements by root-relative prim path and emits the result as the manifest's
``ControlAnimation`` tables, which external animator processes evaluate at render
time. The same samples play as-is in usdview (elements carry proxy gprims; splat
payloads are purpose=render).

The module is intentionally pure (``pxr`` only, no ``torch``) so it can be
unit-tested cheaply and imported by non-renderer processes.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from pxr import Usd, UsdGeom

from nurec.utils.edit.simready_gaussians import in_animation_preview_subtree, open_stage_root


log = logging.getLogger(__name__)

# Continuous gain on a variant-animation element: float timeSamples over the
# cycle (linear), plus the reference color at gain 0.
_ATTR_GAIN = "nurec:gain"
_ATTR_COLD_RGB = "nurec:coldRgb"


class VariantAnimationError(ValueError):
    """Raised for malformed variant-animation authoring on an asset."""


def _invisible_at(prim: Usd.Prim, root: Usd.Prim, tc: Usd.TimeCode) -> bool:
    """True when ``prim`` is hidden at ``tc`` by an ``invisible`` opinion on itself or
    any ancestor up to (and including) the control ``root``.

    USD visibility is pruning — one ``invisible`` anywhere on the chain hides the whole
    subtree — so a state may hide an entire assembly with a single opinion on a parent
    Xform. Walks the RAW ``visibility`` attribute up the chain, mirroring the sibling
    importer (``simready_gaussians``), instead of
    ``UsdGeom.Imageable.ComputeVisibility()``: with the ParticleField schema not
    registered (any stock pxr build), computed visibility silently degrades to
    ``inherited`` on authored-invisible prims.
    """
    p = prim
    while p and p.IsValid():
        attr = p.GetAttribute("visibility")
        if attr and str(attr.Get(tc)) == UsdGeom.Tokens.invisible:
            return True
        if p.GetPath() == root.GetPath():
            break
        p = p.GetParent()
    return False


def _chain_sample_times(prim: Usd.Prim, root: Usd.Prim) -> set[float]:
    """Union of ``visibility`` timeSample positions on the prim's chain up to ``root``."""
    times: set[float] = set()
    p = prim
    while p and p.IsValid():
        attr = p.GetAttribute("visibility")
        if attr:
            times.update(attr.GetTimeSamples())
        if p.GetPath() == root.GetPath():
            break
        p = p.GetParent()
    return times


def _visibility_on_windows(
    prim: Usd.Prim, root: Usd.Prim, start_tc: float, cycle_tc: float
) -> list[tuple[float, float]]:
    """Normalized [start, end) windows where the prim's INHERITED visibility is visible.

    Visibility is inherited: the prim is visible at time ``t`` iff no prim on its
    ancestor chain up to the control ``root`` (inclusive) holds an ``invisible``
    opinion at ``t`` (see :func:`_invisible_at`) — a parent Xform hiding a whole
    assembly clips every descendant's windows. Tokens are held between samples (standard
    USD semantics), so windows are exact: every boundary is a sample position from the
    chain's union of ``visibility`` timeSamples. A chain with no samples is treated as
    always-visible (or never, if statically invisible) while its master is on.
    """
    times = _chain_sample_times(prim, root)

    def visible_at(tc: float) -> bool:
        """Whether this element is visible at timeCode ``tc`` (inherited up the chain)."""
        return not _invisible_at(prim, root, Usd.TimeCode(tc))

    if not times:
        return [(0.0, 1.0)] if visible_at(start_tc) else []

    edges = sorted({0.0, 1.0}.union(min(max((t - start_tc) / cycle_tc, 0.0), 1.0) for t in times))
    windows: list[tuple[float, float]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        if visible_at(start_tc + 0.5 * (a + b) * cycle_tc):
            if windows and windows[-1][1] == a:
                windows[-1] = (windows[-1][0], b)
            else:
                windows.append((a, b))
    return windows


@dataclass
class VariantAnimation:
    """Animation authored INSIDE a variantSet's states.

    A variant is a bundle of scene description, so an animated state (e.g. ``sequence``)
    carries element prims with time-sampled ``visibility`` — standard USD that plays in
    usdview once the state is selected. This extraction turns each state's samples into
    normalized on-windows keyed by the element's ROOT-RELATIVE prim path; the compiler
    matches baked Gaussian parts to elements by path prefix and schedules them. A state
    with no samples (e.g. ``off``) has an empty mapping: a plain static swap.
    """

    variant_set: str  # canonical variantSet name on the asset root
    cycle_s: float  # one animation cycle in seconds (stage timeCode range / tps)
    windows_by_state: dict[str, dict[str, list[tuple[float, float]]]]  # state (lower) -> rel path -> windows
    # state (lower) -> element keys COMPOSED by that state (with or without samples
    # there). An element present without samples is statically visible while the state
    # holds — e.g. a panel present throughout 'off'.
    members_by_state: dict[str, list[str]] = field(default_factory=dict)
    # Continuous gain: float ``nurec:gain`` timeSamples on an element become a
    # piecewise-linear envelope (state -> rel path -> [(pos, gain)]); ``nurec:coldRgb``
    # is the reference color at gain 0 (state -> rel path -> [r, g, b]).
    gains_by_state: dict[str, dict[str, list[tuple[float, float]]]] = field(default_factory=dict)
    cold_rgb_by_state: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    @property
    def is_animated(self) -> bool:
        """True when any state carries visibility windows (a static variantSet has none)."""
        return any(self.windows_by_state.values()) or any(self.gains_by_state.values())


def read_variant_animation(
    usd_path: Union[str, Path],
    variant_set: str,
    root_prim_path: Optional[str] = None,
) -> Optional[VariantAnimation]:
    """Extract the per-state visibility animation of one root variantSet, if any.

    ``variant_set`` is matched case-insensitively against the root prim's variantSets;
    returns ``None`` when the set does not exist (the caller validates control names
    elsewhere). States are selected in turn (in-memory only) and every prim with
    authored ``visibility`` timeSamples under the root contributes its on-windows.
    Windows and membership reflect INHERITED visibility (raw ``visibility`` walked up
    the ancestor chain to the root, see :func:`_invisible_at`), so an opinion on a
    parent Xform clips or hides every element beneath it.
    """
    usd_path = str(usd_path)
    stage, root = open_stage_root(usd_path, root_prim_path, VariantAnimationError)
    canonical = {n.lower(): n for n in root.GetVariantSets().GetNames()}
    name = canonical.get(str(variant_set).strip().lower())
    if name is None:
        return None
    vset = root.GetVariantSet(name)

    tps = float(stage.GetTimeCodesPerSecond() or 24.0)
    start_tc = float(stage.GetStartTimeCode())
    cycle_tc = float(stage.GetEndTimeCode()) - start_tc
    root_prefix = str(root.GetPath())

    authored = vset.GetVariantSelection()
    windows_by_state: dict[str, dict[str, list[tuple[float, float]]]] = {}
    members_by_state: dict[str, list[str]] = {}
    gains_by_state: dict[str, dict[str, list[tuple[float, float]]]] = {}
    cold_rgb_by_state: dict[str, dict[str, list[float]]] = {}
    try:
        for state in vset.GetVariantNames():
            if not vset.SetVariantSelection(state):
                raise VariantAnimationError(f"could not select {name}={state!r} on {root_prefix}")
            wins: dict[str, list[tuple[float, float]]] = {}
            gains: dict[str, list[tuple[float, float]]] = {}
            colds: dict[str, list[float]] = {}
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if in_animation_preview_subtree(prim):
                    continue  # preview clips reference the same payloads; not control animation
                attr = prim.GetAttribute("visibility")
                gattr = prim.GetAttribute(_ATTR_GAIN)
                has_vis = bool(attr) and attr.GetNumTimeSamples() > 0
                has_gain = bool(gattr) and gattr.GetNumTimeSamples() > 0
                if not has_vis and not has_gain:
                    continue
                if cycle_tc <= 0.0:
                    raise VariantAnimationError(
                        f"{usd_path}: {prim.GetPath()} has animated visibility/gain but the stage "
                        "has no timeCode range (author startTimeCode/endTimeCode)"
                    )
                if prim.GetPath() == root.GetPath():
                    # A root-level sample would yield the empty element key '', which no
                    # baked part can ever match — the animation would silently be lost.
                    raise VariantAnimationError(
                        f"{usd_path}: visibility/gain timeSamples are authored on the control root "
                        f"{root.GetPath()} itself; author them on a prim BELOW the root."
                    )
                rel = str(prim.GetPath())[len(root_prefix) :].lstrip("/")
                if has_vis:
                    wins[rel] = _visibility_on_windows(prim, root, start_tc, cycle_tc)
                if has_gain:
                    # Piecewise-linear intensity over the cycle (USD float samples lerp).
                    pts = sorted(
                        (min(max((t - start_tc) / cycle_tc, 0.0), 1.0), float(gattr.Get(Usd.TimeCode(t))))
                        for t in gattr.GetTimeSamples()
                    )
                    gains[rel] = pts
                    cattr = prim.GetAttribute(_ATTR_COLD_RGB)
                    if cattr and cattr.Get() is not None:
                        colds[rel] = [float(x) for x in cattr.Get()]
            windows_by_state[state.lower()] = wins
            gains_by_state[state.lower()] = gains
            cold_rgb_by_state[state.lower()] = colds

        # Keys animated IDENTICALLY in every state are not driven by THIS variantSet:
        # their samples compose from elsewhere (base level, or another control's authored
        # state). Mirroring the importer's union-minus-intersection attribution, drop
        # them — recording them would ship aspects no baked cluster carries and mark
        # static controls as animated.
        state_keys = [s.lower() for s in vset.GetVariantNames()]
        union_keys = (
            set().union(*windows_by_state.values()).union(*gains_by_state.values())
            if (windows_by_state or gains_by_state)
            else set()
        )
        for k in sorted(union_keys):
            per_state = [(windows_by_state[s].get(k), gains_by_state[s].get(k)) for s in state_keys]
            if all(p == per_state[0] for p in per_state):
                for s in state_keys:
                    windows_by_state[s].pop(k, None)
                    gains_by_state[s].pop(k, None)
                    cold_rgb_by_state[s].pop(k, None)

        # Membership pass: which of the (union of) element keys does each state compose?
        # An element sampled in the state is a member (its windows/gains govern when it
        # shows). An element UNSAMPLED in the state but gated by a sampled ANCESTOR there
        # inherits that ancestor's windows (a parent Xform animating a whole assembly).
        # An element with no samples anywhere on its chain is a member only when its
        # INHERITED visibility (raw-attribute walk up to the root at Default time, see
        # _invisible_at — never ComputeVisibility, which degrades on unregistered
        # schemas) is statically visible — states share element prims across variants
        # (only the visibility overrides differ), so mere prim existence would wrongly
        # admit every state's statically-hidden halves (e.g. alternate geometry hidden in "off"),
        # and a statically-invisible ANCESTOR must hide the whole subtree's elements.
        all_keys = (
            sorted(set().union(*windows_by_state.values()).union(*gains_by_state.values()))
            if (windows_by_state or gains_by_state)
            else []
        )
        if all_keys:
            for state in vset.GetVariantNames():
                if not vset.SetVariantSelection(state):
                    raise VariantAnimationError(f"could not select {name}={state!r} on {root_prefix}")
                key = state.lower()
                sampled = set(windows_by_state.get(key) or {}) | set(gains_by_state.get(key) or {})
                present = []
                for k in all_keys:
                    prim = stage.GetPrimAtPath(f"{root_prefix}/{k}")
                    if not prim.IsValid():
                        continue
                    if k in sampled:
                        present.append(k)
                    elif _chain_sample_times(prim, root):
                        windows_by_state.setdefault(key, {})[k] = _visibility_on_windows(prim, root, start_tc, cycle_tc)
                        present.append(k)
                    elif not _invisible_at(prim, root, Usd.TimeCode.Default()):
                        present.append(k)
                members_by_state[key] = present
    finally:
        if authored:
            vset.SetVariantSelection(authored)
        else:
            vset.ClearVariantSelection()

    anim = VariantAnimation(
        variant_set=name,
        cycle_s=cycle_tc / tps if cycle_tc > 0.0 else 0.0,
        windows_by_state=windows_by_state,
        members_by_state=members_by_state,
        gains_by_state=gains_by_state,
        cold_rgb_by_state=cold_rgb_by_state,
    )
    if anim.is_animated:
        log.info(
            "variant animation %s on %s: cycle %.3fs, %s",
            name,
            Path(usd_path).name,
            anim.cycle_s,
            {s: sorted(w) for s, w in windows_by_state.items() if w},
        )
    return anim


__all__ = [
    "VariantAnimationError",
    "VariantAnimation",
    "read_variant_animation",
]
