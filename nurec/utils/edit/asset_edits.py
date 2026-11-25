# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The asset-edits bake: :class:`AssetEditsBaker` turns edits into baked Gaussian rows.

The CLI wrapper and USDZ repack live in ``asset_edits_cli.py``, the authoring schema in
``asset_edits_schema.py``, the tensor-append utilities in ``layer_append.py`` and the
render-time runtime in ``nurec/asset_edits/runtime.py``. Given a parsed ``AssetEditsDoc`` and a
materialized ``GaussiansComposite``, the baker applies each edit by its ``op``: ``add`` and
``replace`` append an asset's clusters to the bound layer (``replace`` also hides the
captured actor), ``exclude`` hides a captured actor, and ``clone`` copies a captured track's
trained rows (or bakes another edit's asset again) onto a new track. Everything is recorded
in the geometry manifest. An asset is a composed SimReady USD or, for a static prop, a 3DGS
``.ply`` (``ply_gaussians.py``).

The end-to-end pipeline is documented in ``nurec/utils/edit/README.md``.

Geometry frame
--------------
The whole asset is placed into the bound track's cuboid-local frame with one shared
transform (rotate, uniform scale, align), so the composed parts keep their layout and ride
the moving track.
"""

from __future__ import annotations

import dataclasses
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, cast

import numpy as np
import torch

from nurec.asset_edits.runtime import (
    BakedCluster,
    ControlAnimation,
    EditManifest,
)
from nurec.datasets.tracks import CuboidTracks
from nurec.models.composite import LayerTrackIds
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.utils.edit.asset_edits_schema import (
    AssetPlacement,
    ControlSpec,
    EditSpec,
    is_ply_asset,
)
from nurec.utils.edit.layer_append import (
    HIDDEN_DENSITY_LOGIT,
    CapturedSnapshot,
    LayerAppender,
    LayerTensorSpec,
    euler_xyz_to_matrix,
    matrix_to_quat_wxyz,
    quat_mul_wxyz,
    world_track_poses,
)
from nurec.utils.edit.ply_gaussians import load_ply_asset

# pxr is in this module's import graph regardless (variant_animation imports pxr, and
# run.py imports this module eagerly), so the SimReady loaders import at top level.
from nurec.utils.edit.simready_gaussians import (
    SimReadyPart,
    SimReadyVariantControl,
    load_simready_asset,
    payloads_equal,
)
from nurec.utils.edit.variant_animation import VariantAnimation
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.types import TrackFlags


log = logging.getLogger(__name__)


class AssetEditsBakeError(ValueError):
    """Raised when a bake cannot proceed (bad binding, unusable asset, layer mismatch).

    A domain error, not a CLI one: :class:`AssetEditsBaker` is meant to be callable from a
    gRPC handler or a notebook, so it must not depend on the CLI framework. The
    ``compile-asset-edits`` command translates this to a ``ClickException`` at its own
    boundary — see :mod:`nurec.utils.edit.asset_edits_cli`.
    """


# Keyframe padding around the clip window for synthetic world-bind tracks. The pose is
# constant, so interpolation is insensitive to the exact value; the padding only keeps the
# track defined for request timestamps landing just outside the clip window.
_WORLD_TRACK_PAD_US = 2_000_000
# Floor for derived world-bind cuboid dims: flat/thin assets still need a usable cuboid.
_MIN_WORLD_DIM_M = 0.05


def _element_token(elem: str) -> str:
    """An INJECTIVE flattening of an element key for use inside a cluster_id.

    A plain ``replace("/", "_")`` maps ``Detail/surface`` and ``Detail_surface`` to the same
    token; duplicate cluster ids are refused at install
    (``EditManifest.validate_against_layers``), which would make the bake's own output
    unloadable. Escaping ``_`` first keeps the mapping reversible.
    """
    return elem.replace("_", "__").replace("/", "_s_")


def _match_element(keys: list[str], prim_path: str, root_prefix: str) -> Optional[str]:
    """The element key (root-relative prim-path prefix) owning ``prim_path``, if any.

    The LONGEST match wins. An asset may animate both a parent Xform and a prim beneath it
    (an assembly gated as a whole, its panels animated individually), putting both
    ``AssemblyL`` and ``AssemblyL/part_2`` in ``keys``. Matching the ancestor would fold the
    nested element's parts into the parent's cluster while the manifest still advertises
    the nested aspect, leaving that aspect with no geometry to show.
    """
    rel = prim_path[len(root_prefix) :].lstrip("/") if prim_path.startswith(root_prefix) else prim_path
    matches = [k for k in keys if rel == k or rel.startswith(k + "/")]
    return max(matches, key=len) if matches else None


def _elements_identical(by_state: dict, all_states: Iterable[str]) -> bool:
    """True when an element contributes the SAME Gaussian payload in EVERY control state.

    Both halves matter, and the second is why ``all_states`` is a parameter. Only such an
    element may collapse to a single state-agnostic ``"<control>:shared:<key>"`` cluster; if
    it is absent from even one state, that cluster has no way to express "hidden in that
    state", and correctness moves out of the baked geometry and into the sender. An element
    present in exactly one state is the case that looks identical vacuously — it is not
    identical across states, it simply has nothing to disagree with, and it costs nothing
    to keep state-scoped (a single-state element bakes once either way).
    """
    states = list(by_state)
    if set(states) != set(all_states):
        return False
    ref = {p.prim_path: p.gaussians for p in by_state[states[0]]}
    for s in states[1:]:
        cur = {p.prim_path: p.gaussians for p in by_state[s]}
        if set(cur) != set(ref) or not all(payloads_equal(ref[q], cur[q]) for q in ref):
            return False
    return True


@dataclass(frozen=True)
class SyntheticTrack:
    """One NEW cuboid track synthesized by the bake: a world-bound prop or a clone.

    Everything the CLI needs to persist the track into the baked artifact: the
    checkpoint splice keys off ``layer_id``/``nurec_track_id``; ``datasource_summary.json``
    takes the NuRec-frame tquat rows; ``sequence_tracks.json`` the world-frame ones. A
    world bind has 2 constant keyframes; a clone inheriting its source's motion has the
    source's K keyframes; a parked clone has 2, or one constant keyframe per source frame
    when it copies a per-track-frame appearance table (``AssetEditsBaker._parked_keyframes``).
    """

    edit_id: str
    layer_id: str
    track_id_str: str
    nurec_track_id: int  # layer-local index — MUST equal the appended rows' cuboid id
    global_index: int  # index in the composite's full (global) track list
    timestamps_us: list[int]  # K keyframe timestamps (NuRec playback clock)
    pose_tquat_nurec: list[list[float]]  # K x [tx, ty, tz, qx, qy, qz, qw], NuRec frame
    pose_tquat_world: list[list[float]]  # same, world frame
    dims_nurec: list[float]  # cuboid dims in the layer's (NuRec) frame
    dims_world: list[float]
    label_class: str
    kind: str = "world"  # world | clone
    flags: str = "DYNAMIC"  # serialized TrackFlags name(s), '|'-joined
    # Per-track appearance state copied from a clone's source track: one CPU row per
    # tensor, keyed by the layer state-dict suffix ("time_embed.timestamps_us_ranges",
    # "time_embed.u", "track_albedos", "track_scales"). The CLI splices these instead of
    # identity rows. None for world binds (and for clones of freshly synthesized tracks).
    per_track_state: Optional[dict[str, torch.Tensor]] = None
    # The same kind of state for every dependent layer that also carries this track (layers
    # whose ``motion_source_layer`` is this track's layer), keyed by layer id; None entries
    # mean "identity / fresh rows" for that layer.
    dependent_state: dict[str, Optional[dict[str, torch.Tensor]]] = dataclasses.field(default_factory=dict)

    @property
    def n_poses(self) -> int:
        """Number of keyframes on the track."""
        return len(self.timestamps_us)

    @property
    def timestamp_range_us(self) -> tuple[int, int]:
        """First and last keyframe timestamps, in microseconds."""
        return (int(min(self.timestamps_us)), int(max(self.timestamps_us)))


def _tquat_rows(ct: CuboidTracks) -> list[list[float]]:
    """A track container's packed poses as plain ``[tx, ty, tz, qx, qy, qz, qw]`` rows."""
    tq = se3_matrix_to_tquat(ct.tracks_poses.matrix())
    return [[float(x) for x in row] for row in tq.detach().cpu().numpy().reshape(-1, 7)]


def _flag_names(flags: TrackFlags) -> str:
    """``TrackFlags`` as the ``'|'``-joined NAME string the serialized track members carry."""
    names = [f.name for f in TrackFlags if f.value and (flags & f) == f and f.name]
    return "|".join(names) if names else "NONE"


def _track_timestamps(ct: CuboidTracks, track_idx: int) -> np.ndarray:
    """Keyframe timestamps (int64 microseconds) of track ``track_idx`` of a container."""
    start, count = (int(v) for v in ct.tracks_packinfo[track_idx].tolist())
    return ct.tracks_timestamps_us[start : start + count].detach().cpu().numpy().astype(np.int64)


def _nearest_frame_slots(times_us: np.ndarray, frames_us: np.ndarray) -> np.ndarray:
    """For each time, the index of the nearest frame in a sorted frame timeline.

    The rule a per-track-frame appearance table is read with at render time
    (``nearest_track_frame_slots``): times outside the timeline clamp to its ends, a tie
    takes the earlier frame, and an empty or one-frame timeline always gives slot 0.
    """
    times = np.asarray(times_us, dtype=np.int64).reshape(-1)
    frames = np.asarray(frames_us, dtype=np.int64).reshape(-1)
    if frames.size < 2:
        return np.zeros(times.shape, dtype=np.int64)
    later = np.clip(np.searchsorted(frames, times, side="left"), 1, frames.size - 1)
    earlier = later - 1
    return np.where(frames[later] - times < times - frames[earlier], later, earlier).astype(np.int64)


@dataclass(frozen=True)
class _LayerContext:
    """Everything per-layer the bake derives once before touching any edit."""

    layer_id: str
    node: RigidGaussianModel
    ct: CuboidTracks
    n_tracks: int
    pose_world: np.ndarray  # (n_tracks, 4, 4) world-frame track poses (cuboid-local -> world)
    spec: LayerTensorSpec
    appender: LayerAppender
    snapshot: CapturedSnapshot


class _ResolvedControl(NamedTuple):
    """One YAML control resolved against the asset's canonical variantSets."""

    spec: ControlSpec
    control: SimReadyVariantControl
    default_state: str


class _PartPlan(NamedTuple):
    """One asset part scheduled for baking, with its role and target visibility."""

    role: str  # static | variant | variant_default | element
    control: Optional[str]
    state: Optional[str]
    element: Optional[str]
    part: SimReadyPart
    hidden: bool


@dataclass(frozen=True)
class _AssemblyTransform:
    """The ONE shared placement transform for a whole asset assembly.

    Never per-part: rotating/scaling parts individually would destroy their relative
    layout. ``rotated`` is order-paired with the part plan it was computed from — keep
    the two sequences aligned or rows land on the wrong prims.
    """

    placement: AssetPlacement
    euler_deg: tuple[float, float, float]
    rot: np.ndarray  # (3, 3) placement rotation
    q_rot: np.ndarray  # (4,) wxyz quaternion of ``rot``
    rotated: list[np.ndarray]  # per-part rotated asset-local positions, paired with the plan
    extent: np.ndarray  # (3,) rotated-assembly bbox extent
    scale: float  # uniform assembly scale
    shift: np.ndarray  # (3,) cuboid-local translation applied after scaling
    scale_log_delta: float  # log(scale), added to per-gaussian log scales
    up_ax: int  # cuboid-local axis most aligned with world up


def _dependent_layers(composite: GaussiansComposite) -> dict[str, list[str]]:
    """Layers that take their motion from another layer, keyed by that owner layer.

    ``config.motion_source_layer`` names the owner. Such a layer shares the owner's track
    list, so an edit on an owner track applies to it too (see ``AssetEditsBaker.bake_layer``).
    """
    out: dict[str, list[str]] = {}
    for layer_id, node in composite.gaussians_nodes.items():
        owner = getattr(getattr(node, "config", None), "motion_source_layer", None)
        if owner:
            out.setdefault(str(owner), []).append(str(layer_id))
    return out


class AssetEditsBaker:
    """Bakes a parsed ``AssetEditsDoc`` into a ``GaussiansComposite``'s layer tensors.

    One instance per compile run. ``bake`` walks the edits layer by layer and accumulates the
    manifest streams (clusters, asset records, control-animation tables); ``manifest()``
    assembles them. The command level has already grouped the edits by layer, checked for
    collisions and read each control's variant animation (``animations``). The asset loaders
    are injectable so unit tests can bake synthetic assets without USD or PLY files.
    """

    def __init__(
        self,
        composite: GaussiansComposite,
        world_to_nurec: Any,
        assets_root: Path,
        *,
        asset_loader: Optional[Callable[..., Any]] = None,
        clip_window_us: Optional[tuple[int, int]] = None,
        animations: Optional[Mapping[tuple[str, str], VariantAnimation]] = None,
    ) -> None:
        """Bind the baker to a materialized composite.

        ``assets_root`` resolves relative ``asset`` paths. ``clip_window_us`` (the artifact's rig
        playback window) is needed only for world binds and parked clones, whose synthetic tracks
        span it. ``animations`` maps ``(edit_id, control_name)`` to the variant animation the
        compiler read from the asset; controls without an entry are static swaps.
        """
        self.composite = composite
        self.world_to_nurec = world_to_nurec
        self.assets_root = assets_root
        self.clip_window_us = clip_window_us
        self._asset_loader = asset_loader
        self.baked_clusters: list[BakedCluster] = []
        self.asset_records: list[dict] = []  # informational manifest records, one per edit
        self.control_animations: list[ControlAnimation] = []  # runtime-consumed live-drive control tables
        # Captured actors hidden by exclude edits (manifest ``excluded_tracks``).
        self.excluded_tracks: list[dict] = []
        # Layers driven by another layer's motion, keyed by that owner (see _dependent_layers).
        self.dependents: dict[str, list[str]] = _dependent_layers(composite)
        # Every layer whose tensors this bake changed, in bake order: the target layers and
        # their dependents. The CLI splices and binds exactly these.
        self.baked_layers: list[str] = []
        # Synthetic-track bookkeeping: tracks synthesized by _materialize_synthetic_tracks
        # (world binds and clones; the CLI persists these into the artifact), the
        # layer-local index each such edit bakes against, the NuRec-frame container of each
        # (a clone may copy the motion of a world bind synthesized in the same bake), and
        # imported-asset reuse between world-bind dims derivation and the bake.
        self.synthetic_tracks: list[SyntheticTrack] = []
        self._synthetic_track_idx: dict[str, int] = {}
        self._synthetic_ct_nurec: dict[str, CuboidTracks] = {}
        # Per-track-frame appearance tables (see _frame_columns): each such layer's track
        # container as loaded, before synthetic tracks re-derive it (its columns are keyed to
        # these frames), and, per captured clone, the source time each clone keyframe shows.
        self._frame_timelines: dict[str, CuboidTracks] = {}
        self._keyframe_source_times: dict[str, np.ndarray] = {}
        self._specs_by_id: dict[str, EditSpec] = {}
        self._prepared: dict[str, tuple[Path, Any, list[_ResolvedControl], list[_PartPlan]]] = {}
        self._animations: dict[tuple[str, str], VariantAnimation] = dict(animations or {})
        # BAKE OUTPUT, keyed (edit_id, control_name): element keys whose payload is
        # identical in every state, baked once and addressed by the state-agnostic aspect
        # "<control>:shared:<key>". Computed in _collect_parts, read back when clusters and
        # animation tables emit.
        self._shared_elements: dict[tuple[str, str], list[str]] = {}
        # (edit id, control name) -> (state -> state aspect, element key -> [aspects]):
        # the explicit SimReady->wire dictionary, recorded as the clusters emit so it
        # can never drift from them; published on each ControlAnimation.
        self._control_vocab: dict[tuple[str, str], tuple[dict[str, str], dict[str, list[str]]]] = {}

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def bake(self, edits_by_layer: dict[str, list[EditSpec]]) -> int:
        """Bake every layer's edits; returns the total number of Gaussians appended."""
        self._specs_by_id = {fx.id: fx for edits in edits_by_layer.values() for fx in edits}
        # A dependent layer has no tracks or motion of its own: its owner's edits reach it
        # (see bake_layer). Editing it directly would bake rows its owner never sees.
        owner_of = {dep: owner for owner, deps in self.dependents.items() for dep in deps}
        for layer_id in edits_by_layer:
            if layer_id in owner_of:
                owner = owner_of[layer_id]
                raise AssetEditsBakeError(
                    f"edits target layer {layer_id!r}, which takes its tracks and motion from {owner!r} "
                    f"(motion_source_layer). Edit {owner!r} instead: a replace, exclude or captured clone there "
                    f"is applied to {layer_id!r} too."
                )
        # Validate captured-track references against the ORIGINAL track lists, BEFORE
        # synthetic-track synthesis appends new tracks: an out-of-range id that happens
        # to collide with a synthetic track's index must fail loud, not silently claim it.
        for layer_id, edits in edits_by_layer.items():
            node = self.composite.gaussians_nodes[layer_id]
            n_original = len(cast(CuboidTracks, node.cuboid_tracks).tracks_id)
            for fx in edits:
                if fx.bind is not None and not fx.bind.is_world:
                    what, idx = "binds", fx.bind.nurec_track_id
                elif fx.is_clone and fx.clone is not None and fx.clone.is_captured:
                    what, idx = "clones captured track", int(fx.clone.source_nurec_track_id or 0)
                else:
                    continue
                if not 0 <= idx < n_original:
                    raise AssetEditsBakeError(
                        f"edit {fx.id!r} ({fx.op}) {what} nurec_track_id {idx}, which is out of range "
                        f"(layer {layer_id!r} has {n_original} cuboid tracks: valid ids 0..{n_original - 1}; "
                        "tracks synthesized by this bake do not count)."
                    )
        # The bake changes track lists and row counts: retire the composite's cached gather
        # topology before the first change, as GaussiansComposite.insert_asset does.
        self.composite._invalidate_fused_gather_generation_plan()
        self._materialize_synthetic_tracks(edits_by_layer)
        total_appended = 0
        for layer_id, edits in edits_by_layer.items():
            total_appended += self.bake_layer(layer_id, edits)
        log.info(
            "Baked %d overlay Gaussians total across %d cluster(s) in %d layer(s): %s",
            total_appended,
            len(self.baked_clusters),
            len(edits_by_layer),
            list(edits_by_layer),
        )
        return total_appended

    def manifest(self) -> EditManifest:
        """The geometry manifest accumulated by ``bake`` (embedded in the baked USDZ)."""
        seen: set[str] = set()
        for cluster in self.baked_clusters:
            if cluster.cluster_id in seen:
                # The install gate refuses duplicate ids, so shipping one would make
                # this bake's own output unloadable. Fail the bake instead.
                raise AssetEditsBakeError(
                    f"bake produced duplicate cluster_id {cluster.cluster_id!r}; "
                    "rename the colliding edit/control/state/element."
                )
            seen.add(cluster.cluster_id)
        return EditManifest(
            clusters=self.baked_clusters,
            assets=self.asset_records,
            excluded_tracks=self.excluded_tracks,
            controls=self.control_animations,
        )

    def bake_layer(self, layer_id: str, edits: list[EditSpec]) -> int:
        """Bake every edit targeting ``layer_id``; returns the Gaussians appended to it.

        Layers that take their motion from ``layer_id`` share its track list, so a replace or
        exclude hides the same track's rows in them too, and a captured clone copies their rows
        of the source track onto the clone's track.
        """
        lc = self._make_layer_context(layer_id)
        deps = {dep: self._make_layer_context(dep) for dep in self.dependents.get(layer_id, [])}
        n_clusters_before = len(self.baked_clusters)

        for fx in edits:
            track_idx = self._check_binding(lc, fx)
            centroid_world = lc.pose_world[track_idx][:3, 3].astype(np.float64)
            log.debug(
                "Edit %s (%s) -> %s track %d centroid=(%.1f, %.1f, %.1f)",
                fx.id,
                fx.op,
                layer_id,
                track_idx,
                centroid_world[0],
                centroid_world[1],
                centroid_world[2],
            )
            if fx.is_exclude:
                hidden = self._hide_original_rows(lc, fx, track_idx)
                self.asset_records.append(self._exclude_record(lc, fx, track_idx, hidden))
                self._apply_to_dependents(deps, fx, track_idx, self.asset_records[-1], self.excluded_tracks[-1])
                continue
            if fx.is_clone and fx.clone is not None and fx.clone.is_captured:
                self._bake_captured_clone(lc, fx, track_idx)
                self._apply_to_dependents(deps, fx, track_idx, self.asset_records[-1])
                continue
            self._log_placement_frame(lc, fx, track_idx)
            self._bake_asset(lc, fx, track_idx, centroid_world)
            if fx.hides_original:
                self._apply_to_dependents(deps, fx, track_idx, self.asset_records[-1])

        # Apply every edit's staged rows in ONE cat per layer tensor (row indices are
        # already assigned arithmetically, so the result is identical to eager cats).
        lc.appender.flush()
        lc.node._validate_fields()
        for dep_lc in deps.values():
            dep_lc.appender.flush()
            dep_lc.node._validate_fields()
        self.baked_layers.extend(lid for lid in (layer_id, *deps) if lid not in self.baked_layers)
        log.info(
            "%s layer: baked %d overlay Gaussians across %d cluster(s) (layer now has %d Gaussians)",
            layer_id,
            lc.appender.appended,
            len(self.baked_clusters) - n_clusters_before,
            int(lc.node.get_num_gaussians()),
        )
        return lc.appender.appended

    def _apply_to_dependents(
        self,
        deps: dict[str, _LayerContext],
        fx: EditSpec,
        track_idx: int,
        record: dict,
        excluded_entry: Optional[dict] = None,
    ) -> None:
        """Repeat a hide (replace, exclude) or a captured-clone copy on the dependent layers.

        Writes what happened per layer into ``record["dependent_layers"]`` (and the
        ``excluded_tracks`` entry): original size, rows hidden, rows copied.
        """
        if not deps:
            return
        info: dict[str, dict[str, int]] = {}
        for dep, dep_lc in deps.items():
            entry = {"original_layer_size": dep_lc.appender.initial_n, "gaussians_hidden": 0, "gaussians_copied": 0}
            if fx.hides_original:
                entry["gaussians_hidden"] = self._hide_original_rows(dep_lc, fx, track_idx)
            elif fx.is_clone and fx.clone is not None and fx.clone.is_captured:
                src_idx = int(fx.clone.source_nurec_track_id or 0)
                rows = dep_lc.snapshot.rows_of_track(src_idx)
                if rows.size:
                    appended = dep_lc.appender.append_copied_rows(
                        rows,
                        track_idx,
                        densities_logit=dep_lc.snapshot.densities_logit[rows],
                        frame_columns=self._frame_columns(dep, fx, src_idx),
                    )
                    entry["gaussians_copied"] = len(appended)
                    entry["first_index"] = appended[0]
                log.info(
                    "  [clone %s] copied %d captured gaussian(s) of track %d onto track %d in dependent layer %s",
                    fx.id,
                    int(rows.size),
                    src_idx,
                    track_idx,
                    dep,
                )
            info[dep] = entry
        record["dependent_layers"] = info
        if excluded_entry is not None:
            excluded_entry["dependent_layers"] = {
                dep: {"gaussians_hidden": e["gaussians_hidden"]} for dep, e in info.items()
            }

    def _layer_flags(self, layer_id: str) -> TrackFlags:
        """The flags a new track needs to pass ``layer_id``'s track filter (DYNAMIC unless it says ``is_dynamic: false``)."""
        config = self.composite.obj_track_ids[layer_id].config
        is_dynamic = config.get("is_dynamic", True) if hasattr(config, "get") else True
        return TrackFlags.DYNAMIC if bool(is_dynamic) else TrackFlags.NONE

    def _clone_track_state(self, node: Any, src_idx: Optional[int], motion: Any) -> Optional[dict[str, torch.Tensor]]:
        """The per-track state a clone inherits from track ``src_idx`` of ``node``, adjusted to its motion."""
        if src_idx is None:
            return None
        state = self._copy_per_track_state(node, src_idx)
        if state is None:
            return None
        if motion.is_world:
            state.pop("time_embed.timestamps_us_ranges", None)  # a parked clone spans the clip
        elif motion.time_shift_us and "time_embed.timestamps_us_ranges" in state:
            state["time_embed.timestamps_us_ranges"] = state["time_embed.timestamps_us_ranges"] + int(
                motion.time_shift_us
            )
        return state

    # ------------------------------------------------------------------
    # Per-layer setup and binding checks
    # ------------------------------------------------------------------
    def _make_layer_context(self, layer_id: str) -> _LayerContext:
        """Derive everything per-layer the bake needs (tracks, poses, spec, appender)."""
        layer_node = cast(RigidGaussianModel, self.composite.gaussians_nodes[layer_id])
        ct = cast(CuboidTracks, layer_node.cuboid_tracks)
        n_tracks = len(ct.tracks_id)
        log.info(
            "%s layer: %d existing Gaussians across %d cuboid tracks",
            layer_id,
            layer_node.get_num_gaussians(),
            n_tracks,
        )
        # World-frame track poses (cuboid-local -> world), so authored world-axis offsets
        # land where the author measured them.
        pose_world = world_track_poses(ct, self.world_to_nurec)
        try:
            tensor_spec = LayerTensorSpec.from_layer(
                layer_node, int(self.composite.config.layers[layer_id].fourier_features_dim or 1)
            )
        except ValueError as exc:
            raise AssetEditsBakeError(f"layer {layer_id!r} cannot take baked overlays: {exc}") from exc
        # Snapshot the captured (trained) Gaussians BEFORE appending any overlays: lens
        # detection, replace / exclude hiding and captured clones must address ORIGINAL rows only.
        return _LayerContext(
            layer_id=layer_id,
            node=layer_node,
            ct=ct,
            n_tracks=n_tracks,
            pose_world=pose_world,
            spec=tensor_spec,
            appender=LayerAppender(layer_node, tensor_spec),
            snapshot=CapturedSnapshot.from_layer(layer_node),
        )

    # ------------------------------------------------------------------
    # Synthetic tracks: world binds (introduced props) and clones
    # ------------------------------------------------------------------
    def _materialize_synthetic_tracks(self, edits_by_layer: dict[str, list[EditSpec]]) -> None:
        """Synthesize one new cuboid track per world bind and per clone.

        Runs before any layer bake, in two rounds: world binds first, then clones (a clone may copy
        the motion of a world bind from the same bake). Each round appends its tracks to the
        composite's global track list and re-derives every affected layer's ``LayerTrackIds``, so
        the new tracks get exactly the layer-local indices the appended rows will be tagged with.
        """
        world_edits = [(lid, fx) for lid, edits in edits_by_layer.items() for fx in edits if fx.is_world]
        clone_edits = [(lid, fx) for lid, edits in edits_by_layer.items() for fx in edits if fx.is_clone]
        if not world_edits and not clone_edits:
            return
        window = self._clip_window_padded(world_edits + clone_edits)
        device = self.composite.cuboid_tracks.device
        # Taken before the first round re-derives any layer's track container.
        carrying = dict.fromkeys(lid for target, _ in clone_edits for lid in (target, *self.dependents.get(target, [])))
        self._frame_timelines = {
            lid: cast(CuboidTracks, cast(RigidGaussianModel, self.composite.gaussians_nodes[lid]).cuboid_tracks)
            for lid in carrying
            if self._frame_table_width(lid) is not None
        }
        # Clone motion comes from the PRE-BAKE layer containers, CALIBRATED: the
        # calibration's per-pose deltas are sized to those containers, and a captured
        # clone must ride the poses its source's trained rows were fit against (the
        # clone's own calibration rows are then identity).
        calibrated: dict[str, CuboidTracks] = {}
        for layer_id, _ in clone_edits:
            if layer_id not in calibrated:
                node = cast(RigidGaussianModel, self.composite.gaussians_nodes[layer_id])
                getter = getattr(node, "get_calibrated_cuboid_tracks", None)
                calibrated[layer_id] = getter() if callable(getter) else cast(CuboidTracks, node.cuboid_tracks)
        if world_edits:
            self._materialize_group(world_edits, lambda lid, fx: self._synthesize_world_track(lid, fx, window, device))
        if clone_edits:
            self._materialize_group(
                clone_edits, lambda lid, fx: self._synthesize_clone_track(lid, fx, calibrated[lid], window, device)
            )

    def _clip_window_padded(self, edits: list[tuple[str, EditSpec]]) -> Optional[tuple[int, int]]:
        """The padded clip window constant-pose synthetic tracks span; None when no edit needs it."""
        needs = any(
            fx.is_world or (fx.is_clone and fx.clone is not None and fx.clone.motion.is_world) for _, fx in edits
        )
        if not needs:
            return None
        if self.clip_window_us is None:
            raise AssetEditsBakeError(
                "document contains world binds or parked clones but the baker was built without "
                "clip_window_us (the artifact's rig playback window)"
            )
        t0 = int(self.clip_window_us[0]) - _WORLD_TRACK_PAD_US
        t1 = int(self.clip_window_us[1]) + _WORLD_TRACK_PAD_US
        if t1 <= t0:
            raise AssetEditsBakeError(f"clip_window_us is degenerate: {self.clip_window_us}")
        return (t0, t1)

    def _materialize_group(
        self,
        items: list[tuple[str, EditSpec]],
        synthesize: Callable[[str, EditSpec], tuple[CuboidTracks, CuboidTracks, Optional[dict[str, torch.Tensor]]]],
    ) -> None:
        """Append one round of synthetic tracks and register them as layer tracks.

        Synthetic tracks append at the end of the global list and ``LayerTrackIds`` keeps insertion
        order, so their layer-local indices are ``old_n, old_n+1, ...`` both now and after the baked
        artifact reloads. This is asserted anyway: a track captured by zero or by several layers
        would desynchronize row and track indexing, so the bake fails instead.
        """
        # Pre-append layer sizes: this round's tracks land at indices old_n, old_n+1, ... in
        # the target layers AND in their dependents (same filter, same track list).
        targets = list(dict.fromkeys(layer_id for layer_id, _ in items))
        affected = list(dict.fromkeys(lid for t in targets for lid in (t, *self.dependents.get(t, []))))
        old_layer_n = {
            layer_id: len(cast(RigidGaussianModel, self.composite.gaussians_nodes[layer_id]).cuboid_tracks.tracks_id)
            for layer_id in affected
        }
        pending: list[
            tuple[
                str,
                EditSpec,
                CuboidTracks,
                CuboidTracks,
                Optional[dict[str, torch.Tensor]],
                dict[str, Optional[dict[str, torch.Tensor]]],
            ]
        ] = []
        for layer_id, fx in items:
            tid = fx.synthetic_track_id or ""
            if tid in self.composite.cuboid_tracks.tracks_id:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: synthetic track_id {tid!r} collides with an existing track in the artifact"
                )
            ct_world, ct_nurec, per_track_state = synthesize(layer_id, fx)
            # A captured clone also inherits each dependent layer's per-track state of its source.
            dep_state: dict[str, Optional[dict[str, torch.Tensor]]] = {}
            for dep in self.dependents.get(layer_id, []):
                dep_state[dep] = None
                if fx.is_clone and fx.clone is not None and fx.clone.is_captured:
                    dep_node = cast(RigidGaussianModel, self.composite.gaussians_nodes[dep])
                    dep_state[dep] = self._clone_track_state(
                        dep_node, int(fx.clone.source_nurec_track_id or 0), fx.clone.motion
                    )
            self.composite.cuboid_tracks = CuboidTracks.Ops.concatenate([self.composite.cuboid_tracks, ct_nurec])
            self._synthetic_ct_nurec[fx.id] = ct_nurec
            pending.append((layer_id, fx, ct_world, ct_nurec, per_track_state, dep_state))

        # Re-derive every affected layer's track partition from the grown global list
        # (the same rail GaussiansComposite.insert_asset rides). Unaffected layers keep
        # their existing LayerTrackIds: appends never shift existing global indices.
        for layer_id in old_layer_n:
            node = cast(RigidGaussianModel, self.composite.gaussians_nodes[layer_id])
            new_ids = LayerTrackIds(config=self.composite.obj_track_ids[layer_id].config)
            new_ids.initialize_from_tracks(self.composite.cuboid_tracks)
            self.composite.obj_track_ids[layer_id] = new_ids
            node.cuboid_tracks = new_ids.get_layer_tracks(self.composite.cuboid_tracks)

        # Membership probe across ALL layers: each synthetic track must be captured by its
        # target layer and by that layer's dependents, and by nothing else.
        membership = {
            layer_id: set(self._probe_layer_track_ids(ids_obj.config))
            for layer_id, ids_obj in self.composite.obj_track_ids.items()
        }

        rank: dict[str, int] = {}
        for layer_id, fx, ct_world, ct_nurec, per_track_state, dep_state in pending:
            tid = fx.synthetic_track_id or ""
            label_class = str(ct_nurec.tracks_label_class[0])
            deps = self.dependents.get(layer_id, [])
            capturing = sorted(lid for lid, ids in membership.items() if tid in ids)
            if capturing != sorted([layer_id, *deps]):
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: synthetic track {tid!r} (label_class={label_class!r}) must be captured by "
                    f"its target layer {layer_id!r}"
                    + (f" and that layer's dependents {deps}" if deps else "")
                    + f", but is captured by {capturing or 'no layer'}. Pick a label_class the target layer's "
                    "track filter accepts and no unrelated layer does."
                )
            layer_tracks = cast(RigidGaussianModel, self.composite.gaussians_nodes[layer_id]).cuboid_tracks
            idx = layer_tracks.tracks_id.index(tid)
            expected = old_layer_n[layer_id] + rank.get(layer_id, 0)
            rank[layer_id] = rank.get(layer_id, 0) + 1
            if idx != expected:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: synthetic track {tid!r} resolved to layer-local index {idx}, expected "
                    f"{expected} — the layer's track ordering is not append-stable; baking would desynchronize "
                    "row/track indexing"
                )
            for dep in deps:
                dep_tracks = cast(RigidGaussianModel, self.composite.gaussians_nodes[dep]).cuboid_tracks
                dep_idx = dep_tracks.tracks_id.index(tid)
                if dep_idx != idx:
                    raise AssetEditsBakeError(
                        f"edit {fx.id!r}: synthetic track {tid!r} is track {dep_idx} in dependent layer {dep!r} "
                        f"but track {idx} in its owner {layer_id!r}; the two layers' track lists have diverged"
                    )
            self._synthetic_track_idx[fx.id] = idx
            self.synthetic_tracks.append(
                SyntheticTrack(
                    edit_id=fx.id,
                    layer_id=layer_id,
                    track_id_str=tid,
                    nurec_track_id=idx,
                    global_index=self.composite.cuboid_tracks.tracks_id.index(tid),
                    timestamps_us=[int(t) for t in ct_nurec.tracks_timestamps_us.detach().cpu().numpy().reshape(-1)],
                    pose_tquat_nurec=_tquat_rows(ct_nurec),
                    pose_tquat_world=_tquat_rows(ct_world),
                    dims_nurec=[float(x) for x in ct_nurec.cuboids_dims[0].detach().cpu().numpy()],
                    dims_world=[float(x) for x in ct_world.cuboids_dims[0].detach().cpu().numpy()],
                    label_class=label_class,
                    kind="clone" if fx.is_clone else "world",
                    flags=_flag_names(TrackFlags(int(ct_nurec.tracks_flags[0].item()))),
                    per_track_state=per_track_state,
                    dependent_state=dep_state,
                )
            )

    def _synthesize_world_track(
        self, layer_id: str, fx: EditSpec, window: Optional[tuple[int, int]], device: torch.device
    ) -> tuple[CuboidTracks, CuboidTracks, Optional[dict[str, torch.Tensor]]]:
        """A world bind's constant-pose track: (world container, NuRec container, no per-track state)."""
        bind = fx.bind
        assert bind is not None and window is not None
        t0, t1 = window
        tid = fx.synthetic_track_id or ""
        dims_world = self._world_bind_dims(fx)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, float(bind.yaw_deg))
        pose[:3, 3] = np.asarray(bind.position, dtype=np.float64)
        if bind.anchor == "base":
            pose[2, 3] += dims_world[2] / 2.0  # position names the cuboid base center
        ct_world = CuboidTracks.Factory.from_numpy(
            tracks_id=[tid],
            tracks_poses=[np.stack([pose, pose]).astype(np.float32)],
            tracks_timestamps_us=[np.array([t0, t1], dtype=np.int64)],
            tracks_label_class=[bind.label_class],
            tracks_flags=[self._layer_flags(layer_id)],
            cuboids_dims=[dims_world.astype(np.float32)],
            device=device,
        )
        ct_nurec = CuboidTracks.Ops.transform_with_frame_conversion(ct_world, self.world_to_nurec, None)
        log.info(
            "  [add %s] synthetic track %r: label_class=%s dims=(%.2f, %.2f, %.2f) m "
            "at world (%.2f, %.2f, %.2f) yaw=%.1f deg over [%d..%d] us",
            fx.id,
            tid,
            bind.label_class,
            dims_world[0],
            dims_world[1],
            dims_world[2],
            pose[0, 3],
            pose[1, 3],
            pose[2, 3],
            float(bind.yaw_deg),
            t0,
            t1,
        )
        return ct_world, ct_nurec, None

    def _synthesize_clone_track(
        self,
        layer_id: str,
        fx: EditSpec,
        calibrated: CuboidTracks,
        window: Optional[tuple[int, int]],
        device: torch.device,
    ) -> tuple[CuboidTracks, CuboidTracks, Optional[dict[str, torch.Tensor]]]:
        """A clone's track: the source's calibrated trajectory offset in its own frame, or a parked pose.

        Returns (world container, NuRec container, copied per-track appearance state). Its
        keyframes are the ones every layer carrying it keeps on reload (``_reload_stable``);
        the source time each one shows is recorded for ``_frame_columns``.
        """
        clone = fx.clone
        assert clone is not None
        tid = clone.track_id
        node = cast(RigidGaussianModel, self.composite.gaussians_nodes[layer_id])
        src_idx, src_ct = self._clone_source(layer_id, fx, calibrated)
        src_tid = str(src_ct.tracks_id[0])
        label_class = str(src_ct.tracks_label_class[0])
        flags = TrackFlags(int(src_ct.tracks_flags[0].item()))
        dims_nurec = src_ct.cuboids_dims[0].detach().cpu().numpy().astype(np.float32)
        motion = clone.motion
        per_track_state = self._clone_track_state(node, src_idx, motion)

        if motion.is_world:
            assert window is not None
            times, source_times = self._parked_keyframes(layer_id, fx, window)
            src_world = CuboidTracks.Ops.transform_with_frame_conversion(src_ct, self.world_to_nurec.inverse(), None)
            dims_world = src_world.cuboids_dims[0].detach().cpu().numpy().astype(np.float64)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, float(motion.yaw_deg))
            pose[:3, 3] = np.asarray(motion.position, dtype=np.float64)
            if motion.anchor == "base":
                pose[2, 3] += dims_world[2] / 2.0
            ct_world = CuboidTracks.Factory.from_numpy(
                tracks_id=[tid],
                tracks_poses=[np.repeat(pose[None].astype(np.float32), times.size, axis=0)],
                tracks_timestamps_us=[times],
                tracks_label_class=[label_class],
                tracks_flags=[flags],
                cuboids_dims=[dims_world.astype(np.float32)],
                device=device,
            )
            ct_world = self._reload_stable(layer_id, ct_world)
            kept = _track_timestamps(ct_world, 0)
            self._keyframe_source_times[fx.id] = source_times[np.searchsorted(times, kept)]
            ct_nurec = CuboidTracks.Ops.transform_with_frame_conversion(ct_world, self.world_to_nurec, None)
            log.info(
                "  [clone %s] synthetic track %r parked at world (%.2f, %.2f, %.2f) yaw=%.1f deg, %d keyframe(s) over "
                "[%d..%d] us; source track %r (label_class=%s, dims=(%.2f, %.2f, %.2f) m)",
                fx.id,
                tid,
                pose[0, 3],
                pose[1, 3],
                pose[2, 3],
                float(motion.yaw_deg),
                int(kept.size),
                int(kept[0]),
                int(kept[-1]),
                src_tid,
                label_class,
                dims_world[0],
                dims_world[1],
                dims_world[2],
            )
            return ct_world, ct_nurec, per_track_state

        poses = src_ct.tracks_poses.matrix().detach().cpu().numpy().astype(np.float64).reshape(-1, 4, 4)
        ts = src_ct.tracks_timestamps_us.detach().cpu().numpy().astype(np.int64).reshape(-1)
        frames = self._source_appearance_frames(layer_id, fx)
        if getattr(node, "is_static", False) and frames.size >= 2:
            # A static layer calibrates every track to one frozen pose (two samples far outside
            # the clip), whatever its keyframes; hold that pose at the source's appearance frames
            # instead, so a per-track-frame table keeps its frames.
            poses = np.repeat(poses[:1], frames.size, axis=0)
            ts = frames
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, float(motion.yaw_deg))
        delta[:3, 3] = np.asarray(motion.offset, dtype=np.float64) * self._nurec_units_per_metre()
        # Right-multiplying offsets every keyframe in the SOURCE track's cuboid-local frame
        # (a lateral offset stays lateral around bends).
        poses = poses @ delta[None, :, :]
        ts = ts + int(motion.time_shift_us)
        ct_nurec = CuboidTracks.Factory.from_numpy(
            tracks_id=[tid],
            tracks_poses=[poses.astype(np.float32)],
            tracks_timestamps_us=[ts],
            tracks_label_class=[label_class],
            tracks_flags=[flags],
            cuboids_dims=[dims_nurec],
            device=device,
        )
        ct_nurec = self._reload_stable(layer_id, ct_nurec)
        kept = _track_timestamps(ct_nurec, 0)
        self._keyframe_source_times[fx.id] = kept - int(motion.time_shift_us)
        ct_world = CuboidTracks.Ops.transform_with_frame_conversion(ct_nurec, self.world_to_nurec.inverse(), None)
        log.info(
            "  [clone %s] synthetic track %r inherits %d keyframe(s) of source track %r (label_class=%s) "
            "with offset=(%.2f, %.2f, %.2f) m yaw=%.1f deg time_shift=%d us",
            fx.id,
            tid,
            int(kept.size),
            src_tid,
            label_class,
            motion.offset[0],
            motion.offset[1],
            motion.offset[2],
            float(motion.yaw_deg),
            int(motion.time_shift_us),
        )
        return ct_world, ct_nurec, per_track_state

    def _clone_source(
        self, layer_id: str, fx: EditSpec, calibrated: CuboidTracks
    ) -> tuple[Optional[int], CuboidTracks]:
        """The clone's source track: (its pre-bake layer-local index, or None for a track
        synthesized this bake; a one-track NuRec-frame container of its poses)."""
        clone = fx.clone
        assert clone is not None
        if clone.is_captured:
            idx = int(clone.source_nurec_track_id or 0)
            return idx, CuboidTracks.Ops.subset_from_indices(calibrated, [idx])
        src = self._specs_by_id.get(clone.source_edit_id or "")
        if src is None:
            raise AssetEditsBakeError(f"edit {fx.id!r}: clone source edit {clone.source_edit_id!r} is not being baked")
        if src.layer_id != layer_id:
            raise AssetEditsBakeError(
                f"edit {fx.id!r}: clone source {src.id!r} bakes into layer {src.layer_id!r}, not {layer_id!r}"
            )
        if src.is_world:
            ct = self._synthetic_ct_nurec.get(src.id)
            if ct is None:
                raise AssetEditsBakeError(f"edit {fx.id!r}: clone source {src.id!r} has no synthesized track yet")
            return None, ct
        idx = src.nurec_track_id
        return idx, CuboidTracks.Ops.subset_from_indices(calibrated, [idx])

    @staticmethod
    def _copy_per_track_state(node: RigidGaussianModel, src_idx: int) -> Optional[dict[str, torch.Tensor]]:
        """One CPU row per per-track appearance tensor of ``src_idx`` (time embedding,
        track albedo/scale), keyed by the layer state-dict suffix; None when the layer has none."""
        state: dict[str, torch.Tensor] = {}
        time_embed = getattr(node, "time_embed", None)
        sources = (
            (time_embed, "timestamps_us_ranges", "time_embed.timestamps_us_ranges"),
            (time_embed, "u", "time_embed.u"),
            (node, "track_albedos", "track_albedos"),
            (node, "track_scales", "track_scales"),
        )
        for owner, attr, key in sources:
            if owner is None:
                continue
            tensor = getattr(owner, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1 and int(tensor.shape[0]) > src_idx:
                state[key] = tensor[src_idx].detach().cpu().clone().unsqueeze(0)
        return state or None

    def _reload_stable(self, layer_id: str, ct: CuboidTracks) -> CuboidTracks:
        """A synthetic track's container cut to the keyframes every layer carrying it keeps on reload.

        Each rigid layer re-applies its own short-gap coalescing (``tracks_calib.short_gap_fusion``)
        to its tracks when the artifact loads, and the target layer and its dependents may differ.
        Coalescing only drops keyframes, which only widens the gaps left, so after one pass through
        each carrying layer's rule every one of them keeps all the remaining keyframes: the
        per-keyframe rows the CLI splices (identity calibration deltas) and a frame table's columns
        (``_frame_columns``) then line up with the reloaded track.
        """
        for lid in (layer_id, *self.dependents.get(layer_id, [])):
            node = cast(RigidGaussianModel, self.composite.gaussians_nodes[lid])
            ct = RigidGaussianModel._maybe_coalesce_short_timestamp_gaps(node.config, ct)
        return ct

    def _frame_table_width(self, layer_id: str) -> Optional[int]:
        """The column count of ``layer_id``'s per-track-frame appearance table; None if it has none.

        With a single Fourier feature a layer has no time embedding, so a third axis on its
        ``features_albedo`` (``[G, N, 3]``) indexes its tracks' frames: when the layer renders,
        each Gaussian shows the column of its track's keyframe nearest the frame time, counted on
        the layer's OWN track timeline (its datasource frames, which may be packed differently from
        its motion source's), and a track with fewer than N frames pads with its last column.
        """
        albedo = getattr(self.composite.gaussians_nodes[layer_id], "features_albedo", None)
        n_fourier = int(self.composite.config.layers[layer_id].fourier_features_dim or 1)
        if n_fourier == 1 and isinstance(albedo, torch.Tensor) and albedo.ndim == 3:
            return int(albedo.shape[1])
        return None

    def _source_appearance_frames(self, layer_id: str, fx: EditSpec) -> np.ndarray:
        """The frames of a captured clone's source in the per-track-frame tables it copies rows of.

        The sorted union over ``layer_id`` and its dependents, counting only layers with such a
        table (``_frame_table_width``) where the source track has rows; empty for other clones.
        """
        clone = fx.clone
        assert clone is not None
        if not clone.is_captured:
            return np.empty(0, dtype=np.int64)
        src_idx = int(clone.source_nurec_track_id or 0)
        per_layer = [
            _track_timestamps(self._frame_timelines[lid], src_idx)
            for lid in (layer_id, *self.dependents.get(layer_id, []))
            if lid in self._frame_timelines
            and bool(
                (cast(RigidGaussianModel, self.composite.gaussians_nodes[lid]).gaussian_cuboid_ids == src_idx).any()
            )
        ]
        return np.unique(np.concatenate(per_layer)) if per_layer else np.empty(0, dtype=np.int64)

    def _parked_keyframes(self, layer_id: str, fx: EditSpec, window: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """A parked clone's keyframe times, and the source time each one shows.

        The window's two ends, unless the clone copies rows of a per-track-frame appearance table
        whose source track has several frames there. Then there is one keyframe per source frame,
        the source's frames spread evenly over the window, so the clone plays its source's
        appearance once across the clip. Its time-embedded appearance does the same: a parked
        clone's time-embedding range is its own span (``_clone_track_state``).
        """
        t0, t1 = window
        frames = self._source_appearance_frames(layer_id, fx)
        if frames.size < 2:
            ends = np.array([t0, t1], dtype=np.int64)
            return ends, ends
        times = t0 + np.round((frames - frames[0]) * ((t1 - t0) / (frames[-1] - frames[0]))).astype(np.int64)
        # Strictly increasing even if rounding merged two frames (a source outliving the window).
        steps = np.arange(times.size)
        return np.maximum.accumulate(times - steps) + steps, frames

    def _frame_columns(self, layer_id: str, fx: EditSpec, src_idx: int) -> Optional[np.ndarray]:
        """The source column each column of a captured clone's copied frame-table rows takes.

        None for a layer without a per-track-frame appearance table (``_frame_table_width``).
        The table is keyed to the frames of the layer's own track, while the clone's track has
        its own keyframes: the owner's calibrated ones shifted by ``time_shift_us`` (possibly
        coalesced differently from this layer's; a static owner's frozen pose held at the
        source's frames), or a parked clone's (``_parked_keyframes``).
        Column ``k`` takes the source frame nearest the source time keyframe ``k`` shows, by the
        rule the layer renders with, so at every keyframe the clone shows what its source showed
        then; between keyframes the layer picks the nearest one. A source frame the clone has no
        keyframe for (one the owner coalesced into a neighbour closer than its short-gap
        threshold) shows that neighbour's column. Columns past the last keyframe repeat it, as
        the trained table pads a short track. A clone can have more keyframes than the table has
        columns (a parked clone's keyframes are the union of its source's frames in every such
        layer, which may pack them differently); the copy then widens the table
        (``LayerAppender.widen_frame_table``).
        """
        width = self._frame_table_width(layer_id)
        if width is None:
            return None
        source_frames = _track_timestamps(self._frame_timelines[layer_id], src_idx)
        columns = _nearest_frame_slots(self._keyframe_source_times[fx.id], source_frames)
        padding = max(width - columns.size, 0)
        return np.concatenate([columns, np.full(padding, columns[-1], dtype=np.int64)])

    def _synthetic_track_for(self, edit_id: str) -> SyntheticTrack:
        """The synthetic track an edit created (world bind or clone)."""
        return next(s for s in self.synthetic_tracks if s.edit_id == edit_id)

    def _probe_layer_track_ids(self, config: Any) -> list[str]:
        """The track ids a layer's filter captures from the CURRENT global track list."""
        probe = LayerTrackIds(config=config)
        probe.initialize_from_tracks(self.composite.cuboid_tracks)
        assert probe.track_ids is not None  # set by initialize_from_tracks
        return probe.track_ids

    def _nurec_units_per_metre(self) -> float:
        """NuRec units per authored metre (``FrameConversion.target_scale``; 1.0 when absent)."""
        scale = getattr(self.world_to_nurec, "target_scale", None)
        return float(scale) if scale is not None else 1.0

    def _world_bind_dims(self, fx: EditSpec) -> np.ndarray:
        """World-frame cuboid dims for a world bind: explicit ``dims``, else the placed asset's bbox.

        Uses the same default-visible part set and placement rotation as the assembly transform, so
        the assembly fills its synthetic cuboid exactly and ``align: bbox`` rests its base at the
        authored position. The imported asset is cached for the bake.
        """
        assert fx.bind is not None
        if fx.bind.dims is not None:
            return np.asarray(fx.bind.dims, dtype=np.float64)
        asset_path, asset = self._load_asset(fx)
        resolved = self._resolve_controls(fx, asset)
        parts = self._collect_parts(fx, asset, resolved)
        self._prepared[fx.id] = (asset_path, asset, resolved, parts)
        placement = fx.placement
        rx, ry, rz = placement.euler_xyz_deg()
        rot = euler_xyz_to_matrix(rx, ry, rz)
        visible = [p for p in parts if not p.hidden] or parts
        all_rot = np.concatenate([p.part.gaussians.positions @ rot.T for p in visible], axis=0)
        extent = (all_rot.max(axis=0) - all_rot.min(axis=0)) * float(placement.scale)
        return np.maximum(extent, _MIN_WORLD_DIM_M)

    def _check_binding(self, lc: _LayerContext, fx: EditSpec) -> int:
        """Validate the edit's track binding; returns the bound track index."""
        if fx.binds_synthetic_track:
            track_idx = self._synthetic_track_idx[fx.id]
            tid = lc.ct.tracks_id[track_idx]
            if tid != fx.synthetic_track_id:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: layer {lc.layer_id!r} track {track_idx} is {tid!r}, not the edit's "
                    "synthetic track — layer tracks changed between materialization and bake"
                )
            return track_idx
        track_idx = fx.nurec_track_id
        if track_idx < 0 or track_idx >= lc.n_tracks:
            raise AssetEditsBakeError(
                f"edit {fx.id!r} binds nurec_track_id={track_idx}, which is out of range "
                f"(layer {lc.layer_id!r} has {lc.n_tracks} cuboid tracks: valid ids 0..{lc.n_tracks - 1})."
            )
        return track_idx

    def _log_placement_frame(self, lc: _LayerContext, fx: EditSpec, track_idx: int) -> None:
        """Cuboid-frame authoring aid (logging only).

        Placement offsets are in the cuboid-LOCAL frame, so report that frame -- the
        box edge lengths and which local axis is length/width/height -- so an offset
        can be set deterministically instead of by trial and error.
        """
        try:
            dims_local = np.asarray(lc.ct.cuboids_dims[track_idx].detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            rmat = lc.pose_world[track_idx][:3, :3]  # cols = world dir of local +x/+y/+z
            up_ax = int(np.argmax(np.abs(rmat[2, :])))  # local axis most aligned with world up
            horiz = [a for a in range(3) if a != up_ax]
            length_ax = max(horiz, key=lambda a: dims_local[a])
            width_ax = horiz[0] if horiz[1] == length_ax else horiz[1]
            axis_names = "xyz"
            log.debug(
                "  [placement] track %d cuboid-local dims: x=%.2f y=%.2f z=%.2f m; "
                "local %s=LENGTH, local %s=WIDTH, local %s=HEIGHT(up). "
                "Offset is in this frame: rear face ~= local-%s = +/-%.2f, tail height ~= local-%s up to +%.2f. "
                "Local-axis world dirs: x->%s y->%s z->%s.",
                track_idx,
                dims_local[0],
                dims_local[1],
                dims_local[2],
                axis_names[length_ax],
                axis_names[width_ax],
                axis_names[up_ax],
                axis_names[length_ax],
                dims_local[length_ax] / 2.0,
                axis_names[up_ax],
                dims_local[up_ax] / 2.0,
                [round(float(x), 2) for x in rmat[:, 0]],
                [round(float(x), 2) for x in rmat[:, 1]],
                [round(float(x), 2) for x in rmat[:, 2]],
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            log.warning("  [placement] could not summarise cuboid frame for track %d: %s", track_idx, exc)

    # ------------------------------------------------------------------
    # The asset bake, decomposed into its pipeline steps
    # ------------------------------------------------------------------
    def _bake_asset(self, lc: _LayerContext, fx: EditSpec, track_idx: int, centroid_world: np.ndarray) -> None:
        """Bake one composed SimReady asset bound to ``track_idx``.

        The whole assembly gets one shared transform (rotate, uniform scale, align), so the parts
        keep their layout. Static parts bake visible. Each controllable control bakes one cluster
        per state, the default state visible and the rest hidden, with aspect ``"<control>:<state>"``,
        so the runtime is an N-way variant selector. Non-controllable controls bake only their
        default state, visible, without a cluster.
        """
        # A clone of an edit bakes the SOURCE edit's asset/placement/controls under its
        # own id; every other op is its own appearance.
        spec = self._appearance_spec(fx)
        prepared = self._prepared.pop(fx.id, None)
        if prepared is not None:
            asset_path, asset, resolved, parts = prepared
        else:
            asset_path, asset = self._load_asset(spec)
            resolved = self._resolve_controls(spec, asset)
            parts = self._collect_parts(spec, asset, resolved)
        xf = self._compute_assembly_transform(lc, spec, track_idx, parts)
        part_records, state_rows, state_opacities = self._append_parts(lc, track_idx, parts, xf)
        self.baked_clusters.extend(
            self._emit_clusters(lc, spec, track_idx, resolved, state_rows, state_opacities, centroid_world)
        )
        self.control_animations.extend(self._emit_control_animations(lc, spec, track_idx, resolved))
        replaced = self._hide_original_rows(lc, fx, track_idx) if fx.hides_original else 0
        self.asset_records.append(
            self._asset_record(lc, fx, spec, track_idx, asset_path, asset, xf, resolved, part_records, replaced)
        )

    def _appearance_spec(self, fx: EditSpec) -> EditSpec:
        """The spec whose asset/placement/controls/prim_path this edit bakes, under this edit's id."""
        if fx.is_clone and fx.clone is not None and not fx.clone.is_captured:
            src = self._specs_by_id.get(fx.clone.source_edit_id or "")
            if src is None:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: clone source edit {fx.clone.source_edit_id!r} is not being baked"
                )
            return dataclasses.replace(src, id=fx.id, op="add", layer_id=fx.layer_id, bind=None, clone=None)
        return fx

    def _load_asset(self, fx: EditSpec) -> tuple[Path, Any]:
        """Resolve and import the edit's asset: a composed SimReady USD, or a static ``.ply``."""
        if not fx.asset:
            raise AssetEditsBakeError(f"edit {fx.id!r} ({fx.op}) has no asset to import")
        asset_path = Path(fx.asset)
        if not asset_path.is_absolute():
            asset_path = (self.assets_root / asset_path).resolve()
        if not asset_path.exists():
            raise AssetEditsBakeError(f"asset not found: {asset_path} (edit {fx.id!r}, asset={fx.asset!r})")

        if self._asset_loader is not None:
            loader = self._asset_loader
        elif is_ply_asset(fx.asset):
            loader = load_ply_asset
        else:
            loader = load_simready_asset
        try:
            asset = loader(
                asset_path,
                control_sets=[c.name for c in fx.controls],
                root_prim_path=fx.prim_path,
            )
        except Exception as exc:
            raise AssetEditsBakeError(f"edit {fx.id!r}: could not import asset: {exc}")
        if asset.up_axis.upper() != "Z":
            log.warning(
                "edit %s: asset is %s-up, not Z-up; placement assumes Z-up (use placement.rotation_deg to correct).",
                fx.id,
                asset.up_axis,
            )
        return asset_path, asset

    def _resolve_controls(self, fx: EditSpec, asset: Any) -> list[_ResolvedControl]:
        """Resolve YAML control names/states (normalized lowercase) against the asset's
        canonical variantSets — the asset's own controls validate the authoring."""
        ctrl_by_lower = {name.lower(): c for name, c in asset.controls.items()}
        resolved: list[_ResolvedControl] = []
        for spec in fx.controls:
            control = ctrl_by_lower.get(spec.name)
            if control is None:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: control {spec.name!r} is not a variantSet on "
                    f"{asset.root_prim_path} (asset variantSets: {sorted(asset.variant_sets) or 'none'})"
                )
            state_by_lower = {s.lower(): s for s in control.states}
            if spec.default is not None and spec.default not in state_by_lower:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: control {spec.name!r} default {spec.default!r} "
                    f"is not an asset state (asset states: {control.states})"
                )
            if len(state_by_lower) != len(control.states):
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: control {spec.name!r} has states that collide "
                    f"case-insensitively ({control.states}); cannot resolve states unambiguously"
                )
            if spec.controllable and not any(control.parts_by_state.values()):
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: control {spec.name!r} is controllable but swaps no "
                    f"Gaussian prims between its states — driving it would be a silent no-op. Check "
                    f"the asset's variantSet (states: {control.states})."
                )
            default_state = state_by_lower[spec.default] if spec.default else control.default_state
            resolved.append(_ResolvedControl(spec, control, default_state))
        return resolved

    def _collect_parts(self, fx: EditSpec, asset: Any, resolved: list[_ResolvedControl]) -> list[_PartPlan]:
        """Collect every part to bake with its role and target visibility.

        Animated controls group their parts by element (the prims driven by visibility windows). An
        element whose payload is identical in every state bakes once, with a state-agnostic aspect,
        instead of once per state; the schedule is then the only variation. The collapsed element
        keys are recorded in ``self._shared_elements`` for the cluster and animation-table emission.
        """
        parts: list[_PartPlan] = []
        for p in asset.static_parts:
            parts.append(_PartPlan("static", None, None, None, p, False))
        root_prefix = asset.root_prim_path

        for spec, control, default_state in resolved:
            # A control is ACTIVE (all states baked, drivable at render time) when it
            # is marked controllable; otherwise only its default state bakes
            # (static dressing).
            is_active = spec.controllable
            anim = self._animations.get((fx.id, spec.name))
            wins_all = anim.windows_by_state if anim else {}
            membs_all = anim.members_by_state if anim else {}
            if wins_all and any(s.lower() == "shared" for s in control.states):
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: control {spec.name!r} has a state literally named "
                    "'shared', which collides with the shared-element aspect namespace"
                )
            keys_by_state = {
                s: sorted(set(wins_all.get(s.lower()) or {}) | set(membs_all.get(s.lower()) or []))
                for s in control.states
            }
            elem_parts: dict[str, dict[str, list]] = {}  # element -> state -> [parts]
            for state in control.states:
                if not is_active and state != default_state:
                    continue  # inactive control: bake only what will ever show
                role = "variant" if is_active else "variant_default"
                for p in control.parts_by_state.get(state, []):
                    k = _match_element(keys_by_state[state], p.prim_path, root_prefix) if is_active else None
                    if k is not None:
                        elem_parts.setdefault(k, {}).setdefault(state, []).append(p)
                    else:
                        parts.append(_PartPlan(role, spec.name, state, None, p, state != default_state))

            shared: list[str] = []
            d_l = default_state.lower()
            wins_d = wins_all.get(d_l) or {}
            membs_d = set(membs_all.get(d_l) or [])

            def _visible_default(k: str) -> bool:
                """Baked-in visibility = what shows before any schedule applies: the
                element's appearance in the DEFAULT state at cycle position 0."""
                return any(a <= 0.0 < b for a, b in wins_d[k]) if k in wins_d else k in membs_d

            for k in sorted(elem_parts):
                by_state = elem_parts[k]
                if _elements_identical(by_state, control.states):
                    shared.append(k)
                    for p in next(iter(by_state.values())):
                        parts.append(_PartPlan("element", spec.name, None, k, p, not _visible_default(k)))
                else:
                    # State-unique payloads: keep one copy per state (aspect stays
                    # state-scoped so the right copy shows). The default state's copy
                    # follows the SAME baked-in visibility rule as shared elements: an
                    # element windowed hidden at cycle position 0 must not bake visible.
                    for state, plist in by_state.items():
                        hidden = state != default_state or not _visible_default(k)
                        for p in plist:
                            parts.append(_PartPlan("variant", spec.name, state, k, p, hidden))
            self._shared_elements[(fx.id, spec.name)] = shared
            if shared:
                log.info(
                    "  [asset %s] control %s: %d element cluster(s) identical across states, baked once: %s",
                    fx.id,
                    spec.name,
                    len(shared),
                    shared,
                )
        if not parts:
            raise AssetEditsBakeError(f"edit {fx.id!r}: asset has no Gaussian parts")
        return parts

    def _compute_assembly_transform(
        self, lc: _LayerContext, fx: EditSpec, track_idx: int, parts: list[_PartPlan]
    ) -> _AssemblyTransform:
        """Shared assembly transform: rotate -> uniform scale -> align translate (+ trim)."""
        placement = fx.placement
        rx, ry, rz = placement.euler_xyz_deg()
        rot = euler_xyz_to_matrix(rx, ry, rz)
        q_rot = matrix_to_quat_wxyz(rot)
        rotated = [p.part.gaussians.positions @ rot.T for p in parts]
        # Fit/align against what actually SHOWS by default: bboxing hidden state copies
        # too (e.g. an extended-ladder state) would skew the shared scale and floor
        # alignment of every state. All parts still share the ONE transform derived
        # here, and ``rotated`` stays order-paired with ``parts``.
        default_visible = [r for r, p in zip(rotated, parts) if not p.hidden]
        all_rot = np.concatenate(default_visible or rotated, axis=0)
        rmin, rmax = all_rot.min(axis=0), all_rot.max(axis=0)
        extent = rmax - rmin

        # Cuboid-local axis roles (same math as the [placement] diagnostic).
        dims_local = np.asarray(lc.ct.cuboids_dims[track_idx].detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        rmat = lc.pose_world[track_idx][:3, :3]
        up_ax = int(np.argmax(np.abs(rmat[2, :])))
        horiz = [a for a in range(3) if a != up_ax]
        length_ax = max(horiz, key=lambda a: dims_local[a])
        if up_ax != 2:
            log.warning(
                "edit %s: track %d cuboid-local up axis is %s, not z; yaw_deg rotates about "
                "asset z and may not match — use placement.rotation_deg to orient explicitly.",
                fx.id,
                track_idx,
                "xyz"[up_ax],
            )

        if placement.scale_mode == "fit_length":
            if extent[length_ax] <= 1e-6:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: degenerate assembly extent along the cuboid length axis; "
                    "set an explicit numeric placement.scale"
                )
            if dims_local[length_ax] <= 1e-3:
                # A degenerate cuboid would fit_length the asset down to (near) zero:
                # an invisible bake with no error. Fail loud instead.
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: track {track_idx} has degenerate cuboid dims "
                    f"({dims_local.tolist()}) along the length axis; fit_length would scale the "
                    "asset to ~zero. Set an explicit numeric placement.scale (or bind.dims for "
                    "world binds)."
                )
            # dims_local is in NuRec units and extent in asset metres, so this ratio
            # already carries the world->NuRec unit conversion.
            assembly_scale = float(dims_local[length_ax] / extent[length_ax])
        else:
            # A fixed scale is authored in real-world terms (metres per asset metre);
            # the layer's cuboid-local frame is in NuRec units, so fold the frame scale
            # in — without it a metre-authored asset bakes wrong by target_scale.
            assembly_scale = float(placement.scale) * self._nurec_units_per_metre()

        shift = np.zeros(3, dtype=np.float64)
        if placement.align == "bbox":
            shift = -0.5 * (rmin + rmax) * assembly_scale  # bbox center -> cuboid center
            # ... except along the up axis: rest the bbox bottom on the cuboid floor.
            shift[up_ax] = -(rmin[up_ax] * assembly_scale) - dims_local[up_ax] / 2.0
        # placement.offset is authored in cuboid-local METRES; the tensors are NuRec units.
        shift = shift + np.asarray(placement.offset, dtype=np.float64) * self._nurec_units_per_metre()

        log.info(
            "  [asset %s] %d part(s), asset extent=(%.2f, %.2f, %.2f) m -> scale=%.3f (%s), "
            "align=%s shift=(%.2f, %.2f, %.2f), euler=(%.1f, %.1f, %.1f) deg, cuboid dims=(%.2f, %.2f, %.2f)",
            fx.id,
            len(parts),
            extent[0],
            extent[1],
            extent[2],
            assembly_scale,
            placement.scale_mode,
            placement.align,
            shift[0],
            shift[1],
            shift[2],
            rx,
            ry,
            rz,
            dims_local[0],
            dims_local[1],
            dims_local[2],
        )
        return _AssemblyTransform(
            placement=placement,
            euler_deg=(rx, ry, rz),
            rot=rot,
            q_rot=q_rot,
            rotated=rotated,
            extent=extent,
            scale=assembly_scale,
            shift=shift,
            scale_log_delta=float(np.log(max(assembly_scale, 1e-8))),
            up_ax=up_ax,
        )

    def _append_parts(
        self, lc: _LayerContext, track_idx: int, parts: list[_PartPlan], xf: _AssemblyTransform
    ) -> tuple[list[dict], dict[tuple, list[int]], dict[tuple, list[float]]]:
        """Append every planned part's rows; returns the per-part bookkeeping.

        Returns ``(part_records, state_rows, state_opacities)``:
        ``state_rows``/``state_opacities`` are keyed ``(control, state_lower, element|None)``
        for per-state clusters or ``(control, "shared", element)`` for cross-state shared
        element clusters.
        """
        part_records: list[dict] = []
        state_rows: dict[tuple, list[int]] = {}
        state_opacities: dict[tuple, list[float]] = {}  # parallel: real per-row alpha
        for plan, pos_rot in zip(parts, xf.rotated, strict=True):
            g = plan.part.gaussians
            positions_local = pos_rot * xf.scale + xf.shift[None, :]
            rotations = quat_mul_wxyz(np.broadcast_to(xf.q_rot, g.rotations.shape), g.rotations)
            rows = lc.appender.append_simready(
                track_idx,
                positions_local=positions_local,
                rotations=rotations,
                scales_log=g.scales_log + xf.scale_log_delta,
                densities_logit=g.densities_logit,
                sh_dc=g.sh_dc,
                sh_rest=g.sh_rest,
                hidden=plan.hidden,
            )
            part_records.append(
                {
                    "prim_path": plan.part.prim_path,
                    "role": plan.role,
                    "control": plan.control,
                    "state": plan.state,
                    "element": plan.element,
                    "first_index": rows[0] if rows else -1,
                    "count": len(rows),
                }
            )
            if plan.role in ("variant", "element"):
                key = (
                    (plan.control, "shared", plan.element)
                    if plan.role == "element"
                    else (plan.control, plan.state.lower() if plan.state else None, plan.element)
                )
                state_rows.setdefault(key, []).extend(rows)
                # The asset's post-activation per-gaussian alpha: the runtime restores
                # these on show (a flat fill would turn translucent surface opaque).
                sigm = 1.0 / (1.0 + np.exp(-g.densities_logit.reshape(-1)))
                state_opacities.setdefault(key, []).extend(float(x) for x in sigm)
            log.debug(
                "    part %-34s role=%-15s %s%6d gaussians -> rows [%d..%d]",
                plan.part.prim_path,
                plan.role,
                f"{plan.control}={plan.state if plan.state is not None else 'elem:' + str(plan.element)} "
                if plan.control
                else "",
                len(rows),
                rows[0] if rows else -1,
                rows[-1] if rows else -1,
            )
        return part_records, state_rows, state_opacities

    def _emit_clusters(
        self,
        lc: _LayerContext,
        fx: EditSpec,
        track_idx: int,
        resolved: list[_ResolvedControl],
        state_rows: dict[tuple, list[int]],
        state_opacities: dict[tuple, list[float]],
        centroid_world: np.ndarray,
    ) -> list[BakedCluster]:
        """One cluster record per (active control, state): aspect ``"<control>:<state>"``, no off
        cluster — the sender activates exactly one state aspect per control at a time,
        so the runtime shows exactly one state cluster per frame."""
        clusters: list[BakedCluster] = []
        for spec, control, default_state in resolved:
            if not spec.controllable:
                continue
            state_aspects: dict[str, str] = {}
            element_aspects: dict[str, list[str]] = {}

            def _cluster(row_key: tuple, cluster_id: str, aspect: str, shape: dict) -> BakedCluster:
                elem_key = shape.get("element")
                if elem_key is None:
                    state_aspects[str(shape["state"]).lower()] = aspect
                else:
                    element_aspects.setdefault(str(elem_key), []).append(aspect)
                rows = state_rows.get(row_key, [])
                shape.update(
                    {
                        "type": "variant",
                        "control": spec.name,
                        "variant_set": control.variant_set,
                        "default_state": default_state,
                        "gaussians": len(rows),
                    }
                )
                return BakedCluster(
                    cluster_id=cluster_id,
                    edit_id=fx.id,
                    layer_id=lc.layer_id,
                    nurec_track_id=track_idx,
                    aspect=aspect,
                    on_indices=rows,
                    on_opacities=state_opacities.get(row_key, []),
                    centroid_world=[float(x) for x in centroid_world],
                    shape=shape,
                )

            # Cross-state shared element clusters: baked once, addressed by the
            # state-agnostic aspect "<ctl>:shared:<key>" from every state's aspect set.
            for elem in self._shared_elements.get((fx.id, spec.name), []):
                clusters.append(
                    _cluster(
                        (spec.name, "shared", elem),
                        f"{fx.id}_{spec.name}_shared_{_element_token(elem)}",
                        f"{spec.name}:shared:{elem}",
                        {"state": None, "element": elem, "shared": True},
                    )
                )
            for state in control.states:
                state_l = state.lower()
                # The plain state cluster (the state's static remainder) always exists;
                # animated states add one cluster per STATE-UNIQUE element, aspect
                # "<ctl>:<state>:<elem>" (shared elements are handled above).
                elem_keys = sorted(
                    k[2] for k in state_rows if k[0] == spec.name and k[1] == state_l and k[2] is not None
                )
                for elem in [None] + elem_keys:
                    clusters.append(
                        _cluster(
                            (spec.name, state_l, elem),
                            f"{fx.id}_{spec.name}_{state_l}" + (f"_{_element_token(elem)}" if elem else ""),
                            f"{spec.name}:{state_l}" + (f":{elem}" if elem else ""),
                            {"state": state, "element": elem},
                        )
                    )
            self._control_vocab[(fx.id, spec.name)] = (
                state_aspects,
                {k: sorted(v) for k, v in element_aspects.items()},
            )
        return clusters

    def _emit_control_animations(
        self, lc: _LayerContext, fx: EditSpec, track_idx: int, resolved: list[_ResolvedControl]
    ) -> list[ControlAnimation]:
        """One animation table per ACTIVE control: everything an external animator
        needs to evaluate this control's instantaneous aspect set at any frame."""
        animations: list[ControlAnimation] = []
        for spec, control, default_state in resolved:
            if not spec.controllable:
                continue
            anim = self._animations.get((fx.id, spec.name))
            vocab_states, vocab_elements = self._control_vocab.get((fx.id, spec.name), ({}, {}))
            animations.append(
                ControlAnimation(
                    edit_id=fx.id,
                    layer_id=lc.layer_id,
                    nurec_track_id=track_idx,
                    track_id_str=str(lc.ct.tracks_id[track_idx]),
                    control=spec.name,
                    states=[s.lower() for s in control.states],
                    default_state=default_state.lower(),
                    cycle_us=int(round(anim.cycle_s * 1_000_000.0)) if anim and anim.cycle_s else 0,
                    windows_by_state=anim.windows_by_state if anim else {},
                    members_by_state=anim.members_by_state if anim else {},
                    shared_elements=list(self._shared_elements.get((fx.id, spec.name), [])),
                    state_aspects=vocab_states,
                    element_aspects=vocab_elements,
                    gains_by_state=anim.gains_by_state if anim else {},
                    cold_rgb_by_state=anim.cold_rgb_by_state if anim else {},
                    semantics=list(spec.semantics),
                )
            )
        return animations

    def _hide_original_rows(self, lc: _LayerContext, fx: EditSpec, track_idx: int) -> int:
        """Hide every ORIGINAL (pre-bake) Gaussian riding this track (replace / exclude).

        Recoverable only by rebaking from the original artifact; the manifest records the
        criterion (cuboid id == track among the first original_layer_size rows) rather
        than hundreds of thousands of indices. MUST address the snapshot's rows, never
        the node's current ids — appended rows are tagged to the same track.
        """
        orig_idx = lc.snapshot.rows_of_track(track_idx)
        if orig_idx.size:
            idx_t = torch.tensor(orig_idx, device=lc.spec.device, dtype=torch.long)
            lc.node.densities.data[idx_t] = HIDDEN_DENSITY_LOGIT
        log.info(
            "  [%s %s] hid %d captured gaussian(s) on track %d (%r) in layer %s",
            fx.op,
            fx.id,
            int(orig_idx.size),
            track_idx,
            str(lc.ct.tracks_id[track_idx]),
            lc.layer_id,
        )
        return int(orig_idx.size)

    def _bake_captured_clone(self, lc: _LayerContext, fx: EditSpec, track_idx: int) -> None:
        """Duplicate a captured track's trained rows onto the clone's synthetic track.

        Rows are copied from the pre-bake snapshot's addressing with the snapshot's
        density logits, so a source that another edit of this bake replaces or excludes
        still clones with its trained appearance. Positions stay in the source's
        cuboid-local frame: the clone track carries the source's calibrated poses (offset
        or parked), so the copies render where that track is. A per-track-frame appearance
        table is re-indexed onto the clone's keyframes (``_frame_columns``).
        """
        clone = fx.clone
        assert clone is not None and clone.source_nurec_track_id is not None
        src_idx = int(clone.source_nurec_track_id)
        src_tid = str(lc.ct.tracks_id[src_idx])
        src_rows = lc.snapshot.rows_of_track(src_idx)
        if src_rows.size == 0:
            raise AssetEditsBakeError(
                f"edit {fx.id!r}: captured track {src_idx} ({src_tid!r}) has no Gaussians in layer "
                f"{lc.layer_id!r}; nothing to clone"
            )
        rows = lc.appender.append_copied_rows(
            src_rows,
            track_idx,
            densities_logit=lc.snapshot.densities_logit[src_rows],
            frame_columns=self._frame_columns(lc.layer_id, fx, src_idx),
        )
        synth = self._synthetic_track_for(fx.id)
        log.info(
            "  [clone %s] copied %d captured gaussian(s) of track %d (%r) onto track %d (%r, %d keyframe(s))",
            fx.id,
            len(rows),
            src_idx,
            src_tid,
            track_idx,
            synth.track_id_str,
            synth.n_poses,
        )
        self.asset_records.append(
            {
                "edit_id": fx.id,
                "op": fx.op,
                "layer_id": lc.layer_id,
                "nurec_track_id": track_idx,
                "bind": self._synthetic_bind_record(synth),
                "clone": {"source": {"nurec_track_id": src_idx, "track_id": src_tid}, "motion": clone.motion.to_dict()},
                "asset": None,
                "root_prim": None,
                "up_axis": None,
                "placement": None,
                "parts": [
                    {
                        "prim_path": f"captured:{src_tid}",
                        "role": "captured",
                        "control": None,
                        "state": None,
                        "element": None,
                        "first_index": rows[0] if rows else -1,
                        "count": len(rows),
                    }
                ],
                "controls": {},
                "variant_sets_available": {},
                "replaced_original_gaussians": 0,
                "original_layer_size": lc.appender.initial_n,
            }
        )

    def _exclude_record(self, lc: _LayerContext, fx: EditSpec, track_idx: int, hidden: int) -> dict:
        """The manifest records of an exclude: its ``assets[]`` entry, plus the
        ``excluded_tracks`` entry scenario-side consumers read."""
        tid = str(lc.ct.tracks_id[track_idx])
        self.excluded_tracks.append(
            {
                "edit_id": fx.id,
                "layer_id": lc.layer_id,
                "nurec_track_id": track_idx,
                "track_id_str": tid,
                "gaussians_hidden": hidden,
            }
        )
        return {
            "edit_id": fx.id,
            "op": fx.op,
            "layer_id": lc.layer_id,
            "nurec_track_id": track_idx,
            "bind": {"type": "nurec_track", "nurec_track_id": track_idx, "track_id": tid},
            "asset": None,
            "root_prim": None,
            "up_axis": None,
            "placement": None,
            "parts": [],
            "controls": {},
            "variant_sets_available": {},
            "replaced_original_gaussians": hidden,
            "original_layer_size": lc.appender.initial_n,
        }

    @staticmethod
    def _synthetic_bind_record(synth: SyntheticTrack) -> dict[str, Any]:
        """The ``bind`` record of an edit riding a synthetic track (world bind or clone)."""
        return {
            "type": synth.kind,
            "nurec_track_id": synth.nurec_track_id,
            "track_id": synth.track_id_str,
            "dims": synth.dims_world,
            "label_class": synth.label_class,
            "keyframes": synth.n_poses,
        }

    def _asset_record(
        self,
        lc: _LayerContext,
        fx: EditSpec,
        spec: EditSpec,
        track_idx: int,
        asset_path: Path,
        asset: Any,
        xf: _AssemblyTransform,
        resolved: list[_ResolvedControl],
        part_records: list[dict],
        replaced: int,
    ) -> dict:
        """The informational manifest record for one asset-baking edit (``spec`` is its appearance)."""
        rx, ry, rz = xf.euler_deg
        bind_record: dict[str, Any]
        if fx.binds_synthetic_track:
            bind_record = self._synthetic_bind_record(self._synthetic_track_for(fx.id))
            if fx.is_world and fx.bind is not None:
                bind_record.update(
                    {
                        "position": [float(x) for x in fx.bind.position or []],
                        "yaw_deg": float(fx.bind.yaw_deg),
                        "anchor": fx.bind.anchor,
                    }
                )
        else:
            bind_record = {
                "type": "nurec_track",
                "nurec_track_id": track_idx,
                "track_id": str(lc.ct.tracks_id[track_idx]),
            }
        record: dict[str, Any] = {
            "edit_id": fx.id,
            "op": fx.op,
            "layer_id": lc.layer_id,
            "nurec_track_id": track_idx,
            "bind": bind_record,
            "asset": str(asset_path),
            "root_prim": asset.root_prim_path,
            "up_axis": asset.up_axis,
            "placement": {
                "align": xf.placement.align,
                "scale": xf.scale,
                "scale_mode": xf.placement.scale_mode,
                "euler_deg": [rx, ry, rz],
                "shift_local": [float(x) for x in xf.shift],
            },
            "parts": part_records,
            "controls": {
                spec.name: {
                    "variant_set": control.variant_set,
                    "states": list(control.states),
                    "default_state": default_state,
                    "controllable": spec.controllable,
                }
                for spec, control, default_state in resolved
            },
            "variant_sets_available": asset.variant_sets,
            "replaced_original_gaussians": replaced,
            "original_layer_size": lc.appender.initial_n,
        }
        if fx.is_clone and fx.clone is not None:
            record["clone"] = {"source": {"edit_id": fx.clone.source_edit_id}, "motion": fx.clone.motion.to_dict()}
        return record
