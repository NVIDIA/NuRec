# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render-time runtime for NuRec asset edits (controllable SimReady assets).

**How the whole feature fits together is documented once, in
``nurec/utils/edit/README.md``.** This module implements the render-time end of
it: the baked :class:`EditManifest`, the :class:`EditSchedule` / :class:`AspectScheduleRegistry`
that say which aspects are visible, and :class:`AspectOverrideRuntime`, the per-frame hook
that reveals the clusters whose aspects are active and hides that track's others.

The runtime is intentionally light: torch, the stdlib-only ``nurec.utils.renamed_keys``,
and the model layer's torch-only hook contract (``nurec.models.gaussians.edit_hook``),
nothing else. That is what lets both
``GaussiansComposite`` and ``nurec.grpc.serve`` reach it, and lets it be unit-tested in
isolation and shared with the compiler.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import zipfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from nurec.models.gaussians.edit_hook import CollectedLayerLayout
from nurec.utils.renamed_keys import normalize_renamed_key


# JSON key of the layer-local cuboid track index in manifests and schedules.
TRACK_ID_KEY = "nurec_track_id"
# Spelling written by files that predate the nre -> nurec rename.
LEGACY_TRACK_ID_KEY = "nre_track_id"


def normalize_track_id_key(d: Mapping[str, Any]) -> dict[str, Any]:
    """Accept `nre_track_id` or `nurec_track_id`; conflicting duplicates raise ValueError."""
    return normalize_renamed_key(d, TRACK_ID_KEY, LEGACY_TRACK_ID_KEY)


log = logging.getLogger(__name__)


# Filename used inside the USDZ archive to carry the baked geometry manifest.
EDIT_MANIFEST_FILENAME = "asset_edits_manifest.json"

# Post-activation density (opacity) targets. The collected ``densities`` tensor at
# the render hook has already been sigmoid-activated, so we write 0..1.
_VISIBLE_OPACITY = 0.999
_HIDDEN_OPACITY = 0.0

# Zeroth-order spherical-harmonics constant (Y_0_0). SH DC for an RGB color c is
# (c - 0.5) / _SH_C0, matching nurec.models.gaussians.utils.RGB2SH. Inlined here to keep
# this runtime free of heavy NuRec imports.
_SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------------
# Baked geometry manifest (embedded in the USDZ; produced by compile-asset-edits)
# ---------------------------------------------------------------------------------
@dataclass
class BakedCluster:
    """One controllable cluster: the baked rows an aspect switches.

    Indices are LAYER-LOCAL (row indices into this cluster's ``layer_id`` Gaussian tensors).
    The runtime adds that layer's offset within the collected composite tensor when
    applying overrides, so a single manifest can carry clusters in several layers.
    """

    cluster_id: str
    edit_id: str
    layer_id: str  # Gaussian layer these indices are local to (e.g. dynamic_rigids)
    nurec_track_id: int  # layer-local cuboid index the cluster is bound to
    aspect: str  # opaque aspect string this cluster represents
    on_indices: list[int]  # cluster rows shown when this aspect is active
    centroid_world: list[float]  # informational: cluster center in world frame
    shape: dict[str, Any] = field(default_factory=dict)  # informational shape spec
    # Per-row post-activation opacities for ``on_indices`` (parallel arrays). Imported
    # SimReady clusters carry real per-gaussian alpha (soft edges, translucency); when present the
    # runtime restores THESE values on show instead of the flat ``_VISIBLE_OPACITY`` (which
    # would turn a translucent surface into an opaque blob). Empty -> flat fill.
    on_opacities: list[float] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "BakedCluster":
        """Parse one manifest cluster entry (inverse of :meth:`to_dict`)."""
        d = normalize_track_id_key(d)
        return BakedCluster(
            cluster_id=str(d["cluster_id"]),
            edit_id=str(d["edit_id"]),
            layer_id=str(d["layer_id"]),
            nurec_track_id=int(d[TRACK_ID_KEY]),
            aspect=str(d["aspect"]),
            on_indices=[int(x) for x in d.get("on_indices", [])],
            centroid_world=[float(x) for x in d.get("centroid_world", [])],
            shape=dict(d.get("shape", {})),
            on_opacities=[float(x) for x in d.get("on_opacities", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        """Manifest JSON entry for this cluster (omits empty optional fields)."""
        out = {
            "cluster_id": self.cluster_id,
            "edit_id": self.edit_id,
            "layer_id": self.layer_id,
            TRACK_ID_KEY: self.nurec_track_id,
            "aspect": self.aspect,
            "on_indices": list(self.on_indices),
            "centroid_world": list(self.centroid_world),
            "shape": dict(self.shape),
        }
        if self.on_opacities:
            out["on_opacities"] = list(self.on_opacities)
        return out


@dataclass
class EditManifest:
    """Baked geometry for every controllable cluster.

    There is no single manifest-level layer: each :class:`BakedCluster` carries its own
    ``layer_id``, so one manifest can mix clusters across several Gaussian layers.

    ``assets`` is informational provenance only (the per-frame runtime never reads
    it): one record per baked edit — its asset, parts (prim path -> baked row
    ranges), native variantSet controls, placement, and how many captured Gaussians
    were hidden by ``replace_original``.
    """

    clusters: list[BakedCluster]
    assets: list[dict[str, Any]] = field(default_factory=list)
    # Animation tables consumed by EXTERNAL animator processes driving the renderer
    # over gRPC (see ControlAnimation). Optional; absent when nothing is controllable.
    controls: list["ControlAnimation"] = field(default_factory=list)
    # Layer sizes (total gaussians per manifest layer) at bake time: binds the
    # manifest's row indices to the checkpoint they were baked against. Empty on
    # manifests from older bakes (validate_against_layers then skips the check).
    layer_row_counts: dict[str, int] = field(default_factory=dict)
    # Content identity of each manifest layer at bake time: a digest of that layer's
    # ``gaussian_cuboid_ids`` (see :func:`layer_track_digest`). A layer can be
    # REORDERED or retrained without changing its row count, which ``layer_row_counts``
    # alone cannot see; the digest closes that gap. Empty on older bakes.
    layer_track_digest: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "EditManifest":
        """Parse an embedded manifest JSON document."""
        return EditManifest(
            clusters=[BakedCluster.from_dict(c) for c in data.get("clusters", [])],
            assets=[dict(v) for v in data.get("assets", [])],
            controls=[ControlAnimation.from_dict(c) for c in data.get("controls", [])],
            layer_row_counts={str(k): int(v) for k, v in (data.get("layer_row_counts") or {}).items()},
            layer_track_digest={str(k): str(v) for k, v in (data.get("layer_track_digest") or {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Manifest JSON document (clusters + asset records + control tables)."""
        out: dict[str, Any] = {
            "clusters": [c.to_dict() for c in self.clusters],
            "assets": [dict(v) for v in self.assets],
        }
        if self.controls:
            out["controls"] = [c.to_dict() for c in self.controls]
        if self.layer_row_counts:
            out["layer_row_counts"] = dict(self.layer_row_counts)
        if self.layer_track_digest:
            out["layer_track_digest"] = dict(self.layer_track_digest)
        return out

    def validate_against_layers(self, layers: Mapping[str, Any]) -> None:
        """Refuse a manifest that does not address the loaded checkpoint.

        THE install-time gate, shared by every runtime entry point (gRPC serve and the
        offline render CLI) — a manifest is only meaningful for the exact bake that
        produced it, and every failure mode here writes baked opacities into unrelated
        gaussians rather than erroring.

        ``layers``: layer_id -> the loaded Gaussian node (anything exposing
        ``get_num_gaussians()`` and a ``gaussian_cuboid_ids`` tensor).

        Checks, cheapest first:

        1. every layer the manifest names is present AND carries per-row cuboid ids
           (clusters are track-bound by definition, so a layer without them — e.g. a
           static SH layer — can never be a valid cluster target);
        2. cluster records are internally coherent (unique ``cluster_id``; parallel
           ``on_opacities``), since the runtime's per-cluster caches key on them;
        3. row counts match :attr:`layer_row_counts` (catches a grown/shrunk layer);
        4. each layer's ``gaussian_cuboid_ids`` digest matches
           :attr:`layer_track_digest` (catches a REORDERED or retrained layer of
           unchanged size, which check 3 cannot see);
        5. every cluster's baked rows are in range AND carry that cluster's own
           ``nurec_track_id`` — the direct statement of what the indices mean, so it
           also covers manifests from bakes predating checks 3 and 4.

        Raises ``ValueError`` naming the offending layer or cluster.
        """
        ids_by_layer: dict[str, torch.Tensor] = {}
        for layer_id in self.layer_ids():
            if layer_id not in layers:
                raise ValueError(
                    f"asset-edits manifest addresses layer {layer_id!r}, which the loaded "
                    f"checkpoint does not have (it has {sorted(layers)}). Wrong artifact?"
                )
            ids = getattr(layers[layer_id], "gaussian_cuboid_ids", None)
            if not (isinstance(ids, torch.Tensor) and ids.ndim == 1):
                raise ValueError(
                    f"asset-edits manifest addresses layer {layer_id!r}, which has no per-row "
                    "gaussian_cuboid_ids: clusters bind to cuboid tracks, so this layer cannot "
                    "be a cluster target (wrong layer_id or wrong artifact?)."
                )
            ids_by_layer[layer_id] = ids

        seen_ids: set[str] = set()
        for cluster in self.clusters:
            if cluster.cluster_id in seen_ids:
                raise ValueError(
                    f"asset-edits manifest carries duplicate cluster_id {cluster.cluster_id!r}; "
                    "cluster ids key the runtime's per-cluster caches and must be unique."
                )
            seen_ids.add(cluster.cluster_id)
            if cluster.on_opacities and len(cluster.on_opacities) != len(cluster.on_indices):
                raise ValueError(
                    f"asset-edits cluster {cluster.cluster_id!r} has {len(cluster.on_opacities)} "
                    f"on_opacities for {len(cluster.on_indices)} on_indices; the arrays are "
                    "parallel and must match."
                )

        for layer_id, expected_rows in self.layer_row_counts.items():
            node = layers.get(layer_id)
            if node is None or not hasattr(node, "get_num_gaussians"):
                continue
            actual_rows = int(node.get_num_gaussians())
            if actual_rows != int(expected_rows):
                raise ValueError(
                    f"asset-edits manifest was baked against layer {layer_id!r} with {expected_rows} "
                    f"gaussians, but the loaded checkpoint has {actual_rows}: the baked row indices do "
                    "not address this checkpoint (stale manifest or re-trained artifact). Re-bake."
                )

        for layer_id, expected in self.layer_track_digest.items():
            ids = ids_by_layer.get(layer_id)
            if ids is None:
                continue
            actual = layer_track_digest(ids)
            if actual != expected:
                raise ValueError(
                    f"asset-edits manifest was baked against a different layer {layer_id!r}: "
                    f"its gaussian_cuboid_ids digest is {actual} but the manifest records "
                    f"{expected}. The rows the manifest addresses are not the rows it was "
                    "baked from (re-trained, re-ordered or wrong artifact). Re-bake."
                )

        for cluster in self.clusters:
            if not cluster.on_indices:
                continue
            ids = ids_by_layer[cluster.layer_id]
            rows = torch.tensor(cluster.on_indices, dtype=torch.long, device=ids.device)
            n = int(ids.shape[0])
            if int(rows.min()) < 0 or int(rows.max()) >= n:
                raise ValueError(
                    f"asset-edits cluster {cluster.cluster_id!r} addresses rows outside layer "
                    f"{cluster.layer_id!r} (size {n}). Stale manifest for this checkpoint."
                )
            if not bool((ids[rows] == cluster.nurec_track_id).all()):
                raise ValueError(
                    f"asset-edits cluster {cluster.cluster_id!r} claims track "
                    f"{cluster.nurec_track_id} in layer {cluster.layer_id!r}, but those rows now "
                    "belong to other tracks: the manifest does not address this checkpoint "
                    "(layer re-ordered, re-baked or edited since). Re-bake."
                )

    def validate_schedule(self, schedule: "EditSchedule") -> None:
        """Raise ``ValueError`` when a schedule addresses something this manifest cannot show.

        :meth:`AspectOverrideRuntime.apply` iterates the manifest, not the registry, so a
        schedule entry keyed to a track with no baked clusters is never visited: a
        one-digit ``nurec_track_id`` typo renders nothing and warns nowhere. The same holds
        for an aspect string no cluster carries.
        """
        known = self.by_layer_track()
        for track in schedule.tracks:
            key = (track.layer_id, track.nurec_track_id)
            if key not in known:
                raise ValueError(
                    f"schedule drives track {key}, which has no baked clusters in this manifest "
                    f"(drivable: {sorted(known)}). Nothing would render."
                )
            baked = {c.aspect for c in known[key]}
            named = {a for s in track.switches for a in s.aspects} | set(track.aspect_gains)
            unknown = sorted(named - baked)
            if unknown:
                raise ValueError(
                    f"schedule for track {key} names aspect(s) {unknown} that no baked cluster "
                    f"carries; they would render as nothing. Baked aspects: {sorted(baked)}."
                )

    def drop_tracks(self, keys: Iterable[tuple[str, int]]) -> None:
        """Remove every cluster (and control table) bound to one of ``keys``, in place.

        Used when a manifest track cannot be resolved in the loaded composite: the
        cluster is genuinely dropped rather than merely omitted from the live
        addressing index, so the "ignored" the log claims is what actually happens.
        """
        drop = set(keys)
        if not drop:
            return
        self.clusters = [c for c in self.clusters if (c.layer_id, c.nurec_track_id) not in drop]
        self.controls = [c for c in self.controls if (c.layer_id, c.nurec_track_id) not in drop]

    def by_layer_track(self) -> dict[tuple[str, int], list[BakedCluster]]:
        """Clusters grouped by their ``(layer_id, nurec_track_id)`` binding."""
        out: dict[tuple[str, int], list[BakedCluster]] = {}
        for c in self.clusters:
            out.setdefault((c.layer_id, c.nurec_track_id), []).append(c)
        return out

    def layer_ids(self) -> list[str]:
        """Distinct layers referenced by clusters, in first-seen order."""
        out: list[str] = []
        for c in self.clusters:
            if c.layer_id not in out:
                out.append(c.layer_id)
        return out


@dataclass
class ControlAnimation:
    """One controllable variantSet of a baked asset: per-state animation windows,
    gain curves and naming, extracted from the asset's own variant animation at bake.

    Pure transport data — NuRec never evaluates it. It is embedded in the manifest for
    external animator processes, which evaluate the instantaneous aspect set per
    frame and send it in each render request's ``AspectState`` (the renderer just
    shows/hides the named clusters). State keys are lowercase (the aspect
    namespace); ``cycle_us == 0`` means a purely static control (e.g. panelState).
    Aspect naming convention the animator must reproduce:
    ``"<control>:<state>"`` for the state's static remainder,
    ``"<control>:shared:<key>"`` for cross-state shared elements,
    ``"<control>:<state>:<key>"`` for state-unique elements.
    The finished mapping is also carried explicitly (``state_aspects`` /
    ``element_aspects``) so senders can read the vocabulary straight off the
    manifest instead of re-deriving it.
    """

    edit_id: str
    layer_id: str
    nurec_track_id: int
    control: str  # normalized lowercase variantSet name, e.g. "appearancemode"
    states: list[str]  # lowercase, asset order (first = asset default ordering)
    default_state: str  # lowercase; what renders when undriven
    cycle_us: int = 0
    # state -> element key -> [(start, end)] normalized [0, 1) visibility windows.
    windows_by_state: dict[str, dict[str, list[tuple[float, float]]]] = field(default_factory=dict)
    # state -> [element keys present] (window-less members are statically visible).
    members_by_state: dict[str, list[str]] = field(default_factory=dict)
    # Element keys baked once across states -> aspect "<control>:shared:<key>".
    shared_elements: list[str] = field(default_factory=list)
    # state -> element -> [(pos, gain)] piecewise-linear cycle gain curve.
    gains_by_state: dict[str, dict[str, list[tuple[float, float]]]] = field(default_factory=dict)
    # state -> element -> [r, g, b] reference color at gain 0.
    cold_rgb_by_state: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # Explicit semantic bindings (proto enum value names, e.g. "PANEL_VISIBILITY").
    semantics: list[str] = field(default_factory=list)
    # The bound track's GLOBAL id string (cleaned scene tracks_id) — lets external
    # animators address tracks by the same track id strings the render wire uses, without
    # access to the model's layer layout. Empty when absent from the manifest.
    track_id_str: str = ""
    # The explicit SimReady->wire dictionary, recorded at bake from the clusters
    # actually emitted. state_aspects: state (lowercase variant name) -> its
    # "<control>:<state>" aspect — the state's non-animated REMAINDER (possibly an
    # empty cluster; it still rides every frame as the state marker). element_aspects:
    # element key (prim path under the asset root) -> every aspect that shows a copy
    # of it (a single ":shared:" entry when shared, else one "<control>:<state>:<key>"
    # per state that carries its own payload) — the individually drivable elements.
    state_aspects: dict[str, str] = field(default_factory=dict)
    element_aspects: dict[str, list[str]] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ControlAnimation":
        """Parse one control-animation table from manifest JSON."""

        def _windows(raw: Any) -> dict[str, dict[str, list[tuple[float, float]]]]:
            """Normalise a state->element->[(a, b), ...] window/gain mapping from JSON."""
            return {
                str(s): {str(k): [(float(a), float(b)) for a, b in ws] for k, ws in (elems or {}).items()}
                for s, elems in (raw or {}).items()
            }

        d = normalize_track_id_key(d)
        return ControlAnimation(
            edit_id=str(d["edit_id"]),
            layer_id=str(d["layer_id"]),
            nurec_track_id=int(d[TRACK_ID_KEY]),
            track_id_str=str(d.get("track_id_str", "")),
            control=str(d["control"]),
            states=[str(s) for s in d.get("states", [])],
            default_state=str(d["default_state"]),
            cycle_us=int(d.get("cycle_us", 0)),
            windows_by_state=_windows(d.get("windows_by_state")),
            members_by_state={str(s): [str(k) for k in ks] for s, ks in (d.get("members_by_state") or {}).items()},
            shared_elements=[str(k) for k in d.get("shared_elements", [])],
            gains_by_state=_windows(d.get("gains_by_state")),
            cold_rgb_by_state={
                str(s): {str(k): [float(c) for c in rgb] for k, rgb in (elems or {}).items()}
                for s, elems in (d.get("cold_rgb_by_state") or {}).items()
            },
            semantics=[str(s) for s in d.get("semantics", [])],
            state_aspects={str(s): str(a) for s, a in (d.get("state_aspects") or {}).items()},
            element_aspects={str(k): [str(a) for a in v] for k, v in (d.get("element_aspects") or {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Manifest JSON entry for this control table (omits empty sections)."""
        out: dict[str, Any] = {
            "edit_id": self.edit_id,
            "layer_id": self.layer_id,
            TRACK_ID_KEY: self.nurec_track_id,
            "track_id_str": self.track_id_str,
            "control": self.control,
            "states": list(self.states),
            "default_state": self.default_state,
            "cycle_us": self.cycle_us,
        }
        if self.windows_by_state:
            out["windows_by_state"] = {
                s: {k: [[a, b] for a, b in ws] for k, ws in elems.items()} for s, elems in self.windows_by_state.items()
            }
        if self.members_by_state:
            out["members_by_state"] = {s: list(ks) for s, ks in self.members_by_state.items()}
        if self.shared_elements:
            out["shared_elements"] = list(self.shared_elements)
        if self.gains_by_state:
            out["gains_by_state"] = {
                s: {k: [[p, g] for p, g in pts] for k, pts in elems.items()} for s, elems in self.gains_by_state.items()
            }
        if self.cold_rgb_by_state:
            out["cold_rgb_by_state"] = {
                s: {k: list(rgb) for k, rgb in elems.items()} for s, elems in self.cold_rgb_by_state.items()
            }
        if self.semantics:
            out["semantics"] = list(self.semantics)
        if self.state_aspects:
            out["state_aspects"] = dict(self.state_aspects)
        if self.element_aspects:
            out["element_aspects"] = {k: list(v) for k, v in self.element_aspects.items()}
        return out


def layer_track_digest(cuboid_ids: torch.Tensor) -> str:
    """Content identity of a Gaussian layer: a digest of its per-row track ids.

    Recorded at bake time (:attr:`EditManifest.layer_track_digest`) and re-derived at
    install, so a layer that was re-ordered or re-trained WITHOUT changing its row
    count is still caught. One device-to-host copy of an int64 vector per layer, at
    install only — never on the frame path.
    """
    ids = cuboid_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if ids.untyped_storage().nbytes() != ids.numel() * ids.element_size():
        # A contiguous VIEW of a larger tensor shares its storage; hashing that storage
        # would cover unrelated bytes at the wrong offset. Clone to own exactly our bytes.
        ids = ids.clone()
    # Via the storage rather than ``.numpy()``: this module is torch-only by design and
    # numpy is not in its dependency closure. Byte-identical either way.
    return hashlib.sha256(bytes(ids.untyped_storage())).hexdigest()


def load_manifest_from_path(path: Path) -> EditManifest:
    """Load a geometry manifest from a standalone JSON file."""
    with open(path, "r") as f:
        return EditManifest.from_dict(json.load(f))


def load_manifest_from_usdz(usdz_path: Path) -> Optional[EditManifest]:
    """Load the baked geometry manifest from a USDZ if present; else ``None``."""
    with zipfile.ZipFile(usdz_path, "r") as zf:
        if EDIT_MANIFEST_FILENAME not in zf.namelist():
            return None
        with zf.open(EDIT_MANIFEST_FILENAME) as f:
            data = json.load(f)
    return EditManifest.from_dict(data)


# ---------------------------------------------------------------------------------
# Resolved schedule (timing). Supplied at render time; never baked into the USDZ.
# ---------------------------------------------------------------------------------
def validate_aspect_gain(
    aspect: str, gain: Any, cold_rgb: Optional[Iterable[Any]], ctx: str
) -> tuple[float, Optional[list[float]]]:
    """Validate one sender-supplied aspect gain; raise ``ValueError`` on misuse.

    ``gain`` scales baked (post-sigmoid) opacities, so it must be a finite value in
    [0, 1]: a negative gain would flip SH band signs, values above 1 violate the
    density contract, and NaN poisons every raster tile it touches. ``cold_rgb`` is
    the 3-channel reference color at gain 0, each channel in [0, 1].
    """
    g = float(gain)
    if not math.isfinite(g) or not 0.0 <= g <= 1.0:
        raise ValueError(f"{ctx}: aspect {aspect!r} gain must be a finite value in [0, 1], got {gain!r}")
    if cold_rgb is None:
        return g, None
    rgb = [float(c) for c in cold_rgb]
    if len(rgb) != 3 or not all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in rgb):
        raise ValueError(f"{ctx}: aspect {aspect!r} cold_rgb must be 3 channels in [0, 1], got {list(cold_rgb)!r}")
    return g, rgb


@dataclass
class AspectSwitch:
    """One timed *state*: the set of aspects visible simultaneously from ``t_us`` on.

    A single-aspect switch is just a one-element set; an empty ``aspects`` list is
    an explicit all-hidden state (zero opacity: the clusters do not rasterize).
    """

    aspects: list[str]
    t_us: int  # absolute microseconds (matches rendering_data frame timestamps)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "AspectSwitch":
        """Parse one timed aspect switch from schedule JSON."""
        aspects = [str(a) for a in (d.get("aspects") or [])]
        if "t_us" not in d:
            raise ValueError(f"schedule switch is missing 't_us': {dict(d)!r}")
        return AspectSwitch(aspects=list(dict.fromkeys(aspects)), t_us=int(d["t_us"]))

    def to_dict(self) -> dict[str, Any]:
        """Schedule JSON entry for this switch."""
        return {"aspects": list(self.aspects), "t_us": self.t_us}


@dataclass
class TrackSchedule:
    """Ordered switch schedule for one bound track.

    Bound to a ``(layer_id, nurec_track_id)`` pair so a schedule can drive edits across
    several Gaussian layers (the same numeric track id is distinct per layer).
    """

    layer_id: str
    nurec_track_id: int
    edit_id: str
    switches: list[AspectSwitch]
    # Per-aspect intensity for live driving: the sender evaluates any fade curve
    # itself and ships the instantaneous value (aspect -> (gain, cold_rgb|None)).
    aspect_gains: dict[str, tuple[float, Optional[list[float]]]] = field(default_factory=dict)

    def active_state_at(self, t_us: int) -> Optional[set[str]]:
        """Set of aspects active at ``t_us``: the last switch with ``t_us <= frame``.

        Before the first switch we hold the first switch's state (the schedule
        fully defines the clip). Returns the active set (possibly empty, an explicit
        all-hidden state) or ``None`` only when there are no switches at all.
        """
        if not self.switches:
            return None
        # Linear scan instead of a per-frame sort: pick the latest switch with
        # t_us <= frame; before the first switch, hold the earliest switch's state
        # (the schedule fully defines the clip).
        best: Optional[AspectSwitch] = None
        earliest: Optional[AspectSwitch] = None
        for s in self.switches:
            if earliest is None or s.t_us < earliest.t_us:
                earliest = s
            if s.t_us <= t_us and (best is None or s.t_us > best.t_us):
                best = s
        active = (best if best is not None else earliest).aspects  # type: ignore[union-attr]
        return set(active)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TrackSchedule":
        """Parse one per-track schedule from schedule JSON."""
        d = normalize_track_id_key(d)
        if "layer_id" not in d or TRACK_ID_KEY not in d:
            raise ValueError(f"schedule track entry needs 'layer_id' and '{TRACK_ID_KEY}': {dict(d)!r}")
        layer_id = str(d["layer_id"])
        track_id = int(d[TRACK_ID_KEY])
        ctx = f"schedule track {layer_id}:{track_id}"
        gains: dict[str, tuple[float, Optional[list[float]]]] = {}
        for a, g in (d.get("aspect_gains") or {}).items():
            if "gain" not in g:
                raise ValueError(f"{ctx}: aspect_gains[{a!r}] is missing 'gain'")
            gains[str(a)] = validate_aspect_gain(str(a), g["gain"], g.get("cold_rgb") or None, ctx)
        return TrackSchedule(
            layer_id=layer_id,
            nurec_track_id=track_id,
            edit_id=str(d.get("edit_id", f"{layer_id}:{track_id}")),
            switches=[AspectSwitch.from_dict(s) for s in d.get("switches", [])],
            aspect_gains=gains,
        )

    def to_dict(self) -> dict[str, Any]:
        """Schedule JSON entry for this track (omits empty gains)."""
        out = {
            "layer_id": self.layer_id,
            TRACK_ID_KEY: self.nurec_track_id,
            "edit_id": self.edit_id,
            "switches": [s.to_dict() for s in sorted(self.switches, key=lambda s: s.t_us)],
        }
        if self.aspect_gains:
            out["aspect_gains"] = {
                a: ({"gain": g, "cold_rgb": cold} if cold is not None else {"gain": g})
                for a, (g, cold) in self.aspect_gains.items()
            }
        return out


@dataclass
class EditSchedule:
    """A full schedule document: timed aspect switches for any number of tracks."""

    # The only implemented time base: switch t_us values are absolute frame-clock
    # microseconds, matching render request timestamps. Validated on parse.
    time_base: str
    tracks: list[TrackSchedule]

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "EditSchedule":
        """Parse a full schedule document (missing keys default sanely)."""
        time_base = str(data.get("time_base", "absolute_us"))
        if time_base != "absolute_us":
            raise ValueError(
                "schedule time_base must be 'absolute_us' (switch t_us values are absolute "
                f"frame-clock microseconds, matching render request timestamps); got {time_base!r}. "
                "No other time base is implemented."
            )
        tracks = [TrackSchedule.from_dict(t) for t in data.get("tracks", [])]
        keys = [(t.layer_id, t.nurec_track_id) for t in tracks]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            # The registry is keyed by (layer_id, nurec_track_id); a duplicate would
            # silently keep only the last entry.
            raise ValueError(f"schedule lists track(s) {dupes} more than once")
        return EditSchedule(time_base=time_base, tracks=tracks)

    def to_dict(self) -> dict[str, Any]:
        """Schedule JSON document."""
        return {
            "time_base": self.time_base,
            "tracks": [t.to_dict() for t in self.tracks],
        }


def load_schedule_from_path(path: Path) -> EditSchedule:
    """Load an aspect schedule from a JSON file (see :class:`EditSchedule`)."""
    with open(path, "r") as f:
        return EditSchedule.from_dict(json.load(f))


# ---------------------------------------------------------------------------------
# Schedule registry (swappable at runtime; gRPC-ready via set_many / set_schedule)
# ---------------------------------------------------------------------------------
def _active_aspects_at(
    schedules: Mapping[tuple[str, int], TrackSchedule], frame_timestamp_us: int
) -> dict[tuple[str, int], set[str]]:
    """Per-track active aspect set for one schedule generation (the caller's snapshot)."""
    out: dict[tuple[str, int], set[str]] = {}
    for key, ts in schedules.items():
        state = ts.active_state_at(frame_timestamp_us)
        if state is not None:
            out[key] = state
    return out


@dataclass
class AspectScheduleRegistry:
    """Holds the active per-track schedules, keyed by ``(layer_id, nurec_track_id)``.

    A gRPC server can call :meth:`set_many` to swap the schedule between frames without
    rebaking geometry; the render CLI calls :meth:`set_schedule` once at startup. The key
    includes the layer because the same numeric track id is a distinct cuboid in each
    Gaussian layer.

    Concurrency: :meth:`set_many` REBINDS :attr:`tracks` rather than mutating it, so any
    single read of that attribute sees one whole schedule generation, never a half-written
    one. That is the entire guarantee — it does NOT make a swap safe to interleave with a
    render that reads the registry more than once. Consumers that need one consistent view
    must take their own snapshot (``schedules = registry.tracks``) and use it throughout;
    :meth:`AspectOverrideRuntime.apply` does exactly that, once per frame.
    """

    tracks: dict[tuple[str, int], TrackSchedule] = field(default_factory=dict)

    def set_schedule(self, schedule: EditSchedule) -> None:
        """Replace the registry contents with a full schedule document."""
        self.set_many(schedule.tracks)

    def set_many(self, tracks: Iterable[TrackSchedule]) -> None:
        """Atomically replace ALL schedules (a track absent from ``tracks`` is dropped)."""
        self.tracks = {(t.layer_id, t.nurec_track_id): t for t in tracks}


# ---------------------------------------------------------------------------------
# Per-frame executor
# ---------------------------------------------------------------------------------
class AspectOverrideRuntime:
    """Applies the schedule to the baked overlays at render time.

    Owned by ``GaussiansComposite`` (installed via ``set_edit_overrides``) and
    called once per render pass from ``_render_collected`` before the rasterizer.

    For every scheduled track this frame: each cluster whose aspect is **in** the
    track's active aspect set is shown; every other cluster of that track is
    hidden. The active set may hold one aspect
    (a simple switch), several (a combination like UK ``red+amber``), or none (an
    explicit all-hidden state). Tracks with no schedule entry are left exactly as
    baked (their overlays stay hidden, so the original scene shows through).
    """

    def __init__(self, manifest: EditManifest, registry: Optional[AspectScheduleRegistry] = None) -> None:
        """Bind the runtime to a manifest; ``registry`` defaults to an empty one.

        The manifest is trusted to address the loaded checkpoint: every entry point
        (gRPC serve and the offline render CLI) calls
        :meth:`EditManifest.validate_against_layers` before constructing the runtime,
        which rejects out-of-range rows and rows belonging to other tracks. The frame
        path re-checks none of it.
        """
        self.manifest = manifest
        self.registry = registry if registry is not None else AspectScheduleRegistry()
        self._layer_ids: list[str] = manifest.layer_ids()
        self._by_layer_track = manifest.by_layer_track()
        self._logged_layout: bool = False
        self._logged_first_apply: bool = False
        # (layer, track) pairs already warned about scheduled aspects no cluster carries.
        self._warned_unknown_aspects: set[tuple[str, int, str]] = set()
        # Per-cluster device caches for the baked row-index / opacity tensors — the
        # manifest is immutable for the runtime's lifetime, so building these from Python
        # lists once per device keeps that host-to-device conversion off the frame path.
        # Keyed by cluster OBJECT IDENTITY (safe because the manifest is immutable), so
        # the cache can never alias two clusters whatever their cluster_id strings are.
        self._row_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._opac_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _map_rows(
        self, layout: CollectedLayerLayout, rows: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Map unfiltered layer-local ``rows`` to this frame's collected rows.

        Returns ``(mapped_rows, valid_mask)`` — ``valid_mask`` is ``None`` when the layer
        was collected unfiltered (the mapping is the identity); otherwise ``mapped_rows``
        is meaningful only where the mask is True.

        Cost contract (this runs once per cluster per sensor per frame):

        * unfiltered layer — no allocation, no kernel, no host sync. ``rows`` was
          validated against the layer at install (``validate_against_layers``), so
          re-testing it per frame on the device would buy nothing and cost a sync.
        * row-filtered layer (edited-actors requests only) — ``kept_rows`` is strictly
          increasing, so the inverse is a ``searchsorted`` allocating ``len(rows)``, not
          the full-layer boolean mask plus ``cumsum`` a lookup table would need. The
          caller's one mask-index of the result is a host sync, but the collect that
          produced ``kept_rows`` already synced on this same path
          (``RigidGaussianModel._filter_edited_gaussians``), so it adds no new class of
          cost; it disappears if that collect ever stops materializing indices.
        """
        kept = layout.kept_rows
        if kept is None:
            return rows, None
        kept = kept.to(device=rows.device, dtype=rows.dtype)
        n_kept = int(kept.numel())
        if n_kept == 0:
            return rows, torch.zeros_like(rows, dtype=torch.bool)
        pos = torch.searchsorted(kept, rows).clamp_(max=n_kept - 1)
        # A baked row survived the filter iff it is exactly where searchsorted lands.
        # This also rejects rows outside the layer entirely, so no separate bound test
        # (and no ``.all()`` sync) is needed.
        return pos, kept[pos] == rows

    # ------------------------------------------------------------------
    # Render-time hook
    # ------------------------------------------------------------------
    @torch.no_grad()
    def apply(
        self,
        gaussian_parameters: dict[str, torch.Tensor],
        frame_timestamp_us: int,
        layouts: Mapping[str, CollectedLayerLayout],
    ) -> None:
        """Write this frame's aspect state into the collected opacity/color tensors.

        ``layouts`` is what THIS sensor's collect recorded (layer_id -> where that layer
        landed in these tensors), so the frame path is a pure function of its arguments.
        """
        wanted = set(self._layer_ids)
        layouts = {lid: lo for lid, lo in layouts.items() if lid in wanted}
        if not self._logged_layout and layouts:
            filtered = [lid for lid, lo in layouts.items() if lo.kept_rows is not None]
            log.info(
                "AspectOverrideRuntime: using collected layer layouts %s (manifest layers=%s%s)",
                {lid: (lo.layer_offset, lo.layer_size) for lid, lo in layouts.items()},
                self._layer_ids,
                f"; row-filtered: {filtered}" if filtered else "",
            )
            self._logged_layout = True
        if not layouts:
            return

        # ONE registry snapshot for the whole frame (the concurrency contract lives on
        # AspectScheduleRegistry: rebind-not-mutate makes this single read atomic).
        schedules = self.registry.tracks
        if not schedules:
            return

        densities = gaussian_parameters.get("densities")
        if densities is None:
            return

        active = _active_aspects_at(schedules, frame_timestamp_us)
        if not active:
            return

        if not self._logged_first_apply:
            sample = ", ".join(f"{lid}:{tid}={sorted(asp)}" for (lid, tid), asp in list(active.items())[:4])
            log.info(
                "AspectOverrideRuntime: applying %d active track aspect(s) at t=%dus (first: %s); densities shape=%s",
                len(active),
                frame_timestamp_us,
                sample,
                tuple(densities.shape),
            )
            self._logged_first_apply = True

        squeezed = densities.view(densities.shape[0]) if densities.ndim == 2 and densities.shape[1] == 1 else densities
        device = squeezed.device

        # SH (color) tensor; DC term is features[:, :3]. Used for sender-evaluated
        # color interpolation (_blend_color).
        features = gaussian_parameters.get("features")

        def _cluster_rows(cluster: BakedCluster) -> torch.Tensor:
            """The cluster's baked rows as a device tensor, cached per device.

            The rows are trusted as-is: the install-time gate
            (:meth:`EditManifest.validate_against_layers`) already refused any manifest
            whose indices fall outside the layer or on another track's rows.
            """
            key = (id(cluster), str(device))
            rows = self._row_cache.get(key)
            if rows is None:
                rows = torch.tensor(cluster.on_indices, device=device, dtype=torch.long)
                self._row_cache[key] = rows
            return rows

        def _cluster_opacities(cluster: BakedCluster) -> torch.Tensor:
            """The cluster's baked per-row opacities as a device tensor, cached per device."""
            key = (id(cluster), str(device))
            values = self._opac_cache.get(key)
            if values is None or values.dtype != squeezed.dtype:
                values = torch.tensor(cluster.on_opacities, device=device, dtype=squeezed.dtype)
                self._opac_cache[key] = values
            return values

        def _gidx(
            cluster: BakedCluster, layout: CollectedLayerLayout
        ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            """Baked unfiltered rows -> global collected indices, plus a survivor mask over
            the cluster's own indices when rows were dropped (by the edited-actors row
            filter, or masked out by the collector). ``(None, ...)`` = nothing to touch.

            The mask is applied ONCE here rather than tested with ``.any()``/``.all()``
            first: each of those tests is a device-to-host sync on the render path, and
            they only save work in the case where there is no work to do.
            """
            if not cluster.on_indices:
                return None, None
            if layout.layer_size == 0:
                # The row filter kept nothing from this layer: no collected row to write,
                # and indexing the zero-length collect_keep below would raise.
                return None, None
            mapped, valid = self._map_rows(layout, _cluster_rows(cluster))
            # Never re-show rows the collector masked out (track outside its keyframe
            # window, or otherwise deactivated): AND the collector's own keep mask in.
            # It is per COLLECTED ROW, so it indexes the same space as ``mapped``.
            keep = layout.collect_keep
            if keep is not None:
                safe = mapped if valid is None else torch.where(valid, mapped, torch.zeros_like(mapped))
                alive = keep.to(device=mapped.device, dtype=torch.bool)[safe]
                valid = alive if valid is None else (valid & alive)
            if valid is not None:
                mapped = mapped[valid]
                if mapped.numel() == 0:
                    return None, valid
            return mapped + layout.layer_offset, valid

        def _fill(cluster: BakedCluster, value: float, layout: CollectedLayerLayout) -> None:
            """Write one opacity value into every surviving collected row of the cluster."""
            idx, _ = _gidx(cluster, layout)
            if idx is not None:
                squeezed.index_fill_(0, idx, value)

        def _show(cluster: BakedCluster, layout: CollectedLayerLayout, gain: float = 1.0) -> None:
            """Reveal a cluster: per-row baked opacities when recorded
            (imported SimReady clusters keep their real alpha profile), else flat fill.
            ``gain`` scales the alphas (a sender-evaluated fade mid-cycle)."""
            if len(cluster.on_opacities) == len(cluster.on_indices):
                idx, valid = _gidx(cluster, layout)
                if idx is None:
                    return
                values = _cluster_opacities(cluster)
                if valid is not None:
                    values = values[valid]
                squeezed.index_copy_(0, idx, values * gain if gain != 1.0 else values)
            else:
                _fill(cluster, _VISIBLE_OPACITY * gain, layout)

        def _blend_color(
            cluster: BakedCluster, cold_rgb: list[float], gain: float, layout: CollectedLayerLayout
        ) -> None:
            """Lerp the cluster's SH DC from ``cold_rgb`` toward its BAKED color by
            ``gain`` (stateless: the collected features are rebuilt from the baked
            values every frame). Higher-order SH scales with gain so view-dependent
            appearance fades in with the same gain."""
            if features is None or features.ndim != 2 or features.shape[1] < 3:
                return
            idx, _ = _gidx(cluster, layout)
            if idx is None:
                return
            reference_dc = torch.tensor(
                [(float(c) - 0.5) / _SH_C0 for c in cold_rgb], device=features.device, dtype=features.dtype
            )
            baked_dc = features.index_select(0, idx)[:, :3]
            features[idx, :3] = reference_dc + (baked_dc - reference_dc) * gain
            if features.shape[1] > 3:
                features[idx, 3:] *= gain

        for (layer_id, track_id), clusters in self._by_layer_track.items():
            active_state = active.get((layer_id, track_id))
            if active_state is None:
                continue
            layout = layouts.get(layer_id)
            if layout is None:
                continue  # cluster's layer is not present in this composite
            # Aspects the schedule names but no cluster of this track carries would
            # otherwise vanish silently (sender typo / stale manifest): warn once.
            unknown = active_state - {c.aspect for c in clusters}
            for aspect in sorted(unknown):
                warn_key = (layer_id, track_id, aspect)
                if warn_key not in self._warned_unknown_aspects:
                    self._warned_unknown_aspects.add(warn_key)
                    log.warning(
                        "scheduled aspect %r names no baked cluster on (%s, %d); it will not render "
                        "(track has %d cluster(s)). Sender typo or stale manifest?",
                        aspect,
                        layer_id,
                        track_id,
                        len(clusters),
                    )
            # Sender-evaluated per-aspect gains for this track (aspect -> (gain, cold)).
            track_schedule = schedules.get((layer_id, track_id))
            gains = track_schedule.aspect_gains if track_schedule is not None else {}
            for cluster in clusters:
                is_on = cluster.aspect in active_state
                if is_on:
                    gain, cold = gains.get(cluster.aspect, (1.0, None))
                    _show(cluster, layout, gain=gain)
                    if cold is not None and gain < 1.0:
                        _blend_color(cluster, cold, gain, layout)
                else:
                    _fill(cluster, _HIDDEN_OPACITY, layout)


__all__ = [
    "EDIT_MANIFEST_FILENAME",
    "BakedCluster",
    "ControlAnimation",
    "EditManifest",
    "AspectSwitch",
    "TrackSchedule",
    "EditSchedule",
    "AspectScheduleRegistry",
    "AspectOverrideRuntime",
    "load_manifest_from_path",
    "load_manifest_from_usdz",
    "layer_track_digest",
    "load_schedule_from_path",
    "validate_aspect_gain",
]
