# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The asset-edits bake: :class:`AssetEditsBaker` turns edits into baked clusters.

The core of the compile pipeline (the CLI wrapper and USDZ repack live in
``asset_edits_cli.py``, the authoring schema in ``asset_edits_schema.py``, the
tensor-append utilities in ``layer_append.py``, and the render-time runtime in
``nurec/asset_edits/runtime.py``). Given a parsed ``AssetEditsDoc`` and a materialized
``GaussiansComposite``, the baker appends each asset's Gaussian clusters to the bound
layer (each controllable variantSet state as its own hidden cluster) and accumulates
the geometry manifest recording how to address them. An asset is a composed SimReady
USD, or — for a static prop — a 3DGS ``.ply`` (``ply_gaussians.py``).

The end-to-end pipeline is documented once, in ``nurec/utils/edit/README.md``.

Geometry frame
--------------
The whole asset is placed into the bound track's **cuboid-local** frame with one
shared transform (rotate -> uniform scale -> align translate), so the composed
parts keep their relative layout and ride the moving track.
"""

from __future__ import annotations

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
    """One NEW cuboid track synthesized for a world-bound edit (an introduced prop).

    Everything the CLI needs to persist the track into the baked artifact: the
    checkpoint splice keys off ``layer_id``/``nurec_track_id``; ``datasource_summary.json``
    takes the NuRec-frame tquat rows; ``sequence_tracks.json`` the world-frame ones.
    """

    edit_id: str
    layer_id: str
    track_id_str: str
    nurec_track_id: int  # layer-local index — MUST equal the appended rows' cuboid id
    global_index: int  # index in the composite's full (global) track list
    timestamps_us: tuple[int, int]
    pose_tquat_nurec: list[list[float]]  # 2 x [tx, ty, tz, qx, qy, qz, qw], NuRec frame
    pose_tquat_world: list[list[float]]  # same, world frame
    dims_nurec: list[float]  # cuboid dims in the layer's (NuRec) frame
    dims_world: list[float]
    label_class: str


def _tquat_rows(ct: CuboidTracks) -> list[list[float]]:
    """A track container's packed poses as plain ``[tx, ty, tz, qx, qy, qz, qw]`` rows."""
    tq = se3_matrix_to_tquat(ct.tracks_poses.matrix())
    return [[float(x) for x in row] for row in tq.detach().cpu().numpy().reshape(-1, 7)]


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


class AssetEditsBaker:
    """Bakes a parsed ``AssetEditsDoc`` into a ``GaussiansComposite``'s layer tensors.

    One instance per compile run; ``bake`` walks the edits layer by layer and
    accumulates the manifest streams (baked clusters, asset records,
    control-animation tables) on the instance; ``manifest()`` assembles them.

    Preconditions: each scheduled control's variant animation has already been read
    from its asset and passed in via ``animations``, and the edit->layer
    grouping/collision checks happened at the command level.

    The asset loaders are injectable so unit tests can bake synthetic assets
    without USD or PLY files.
    """

    def __init__(
        self,
        composite: GaussiansComposite,
        world_to_nurec: Any,
        yaml_dir: Path,
        *,
        asset_loader: Optional[Callable[..., Any]] = None,
        clip_window_us: Optional[tuple[int, int]] = None,
        animations: Optional[Mapping[tuple[str, str], VariantAnimation]] = None,
    ) -> None:
        """Bind the baker to a materialized composite.

        ``clip_window_us`` (the artifact's rig playback window) is required only when
        the document contains world binds: their synthetic tracks span that window.

        ``animations`` is the variant animation the COMPILER extracted from each asset,
        keyed ``(edit_id, control_name)`` — a constructor argument rather than fields
        written back onto the authored ``ControlSpec``, so "has the animation been
        attached yet?" is not a question anyone can ask of a parsed document. Controls
        with no entry are purely static swaps (e.g. panelState).
        """
        self.composite = composite
        self.world_to_nurec = world_to_nurec
        self.yaml_dir = yaml_dir
        self.clip_window_us = clip_window_us
        self._asset_loader = asset_loader
        self.baked_clusters: list[BakedCluster] = []
        self.asset_records: list[dict] = []  # informational manifest records, one per edit
        self.control_animations: list[ControlAnimation] = []  # runtime-consumed live-drive control tables
        # World-bind bookkeeping: tracks synthesized by _materialize_world_binds (the
        # CLI persists these into the artifact), the layer-local index each edit bakes
        # against, and imported-asset reuse between dims derivation and the bake.
        self.synthetic_tracks: list[SyntheticTrack] = []
        self._world_track_idx: dict[str, int] = {}
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
        # Validate nurec_track binds against the ORIGINAL track lists, BEFORE world-bind
        # synthesis appends new tracks: an out-of-range id that happens to collide with
        # a synthetic track's index must fail loud, not silently claim the prop's track.
        for layer_id, edits in edits_by_layer.items():
            node = self.composite.gaussians_nodes[layer_id]
            n_original = len(cast(CuboidTracks, node.cuboid_tracks).tracks_id)
            for fx in edits:
                if not fx.bind.is_world and not 0 <= fx.bind.nurec_track_id < n_original:
                    raise AssetEditsBakeError(
                        f"edit {fx.id!r}: invalid nurec_track_id {fx.bind.nurec_track_id} "
                        f"(layer {layer_id!r} has {n_original} cuboid tracks: valid ids "
                        f"0..{n_original - 1}; world-bind tracks synthesized by this bake do not count)."
                    )
        self._materialize_world_binds(edits_by_layer)
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
            controls=self.control_animations,
        )

    def bake_layer(self, layer_id: str, edits: list[EditSpec]) -> int:
        """Bake every edit targeting ``layer_id``; returns the Gaussians appended."""
        lc = self._make_layer_context(layer_id)
        n_clusters_before = len(self.baked_clusters)

        for fx in edits:
            track_idx = self._check_binding(lc, fx)
            centroid_world = lc.pose_world[track_idx][:3, 3].astype(np.float64)
            log.debug(
                "Edit %s -> %s track %d centroid=(%.1f, %.1f, %.1f)",
                fx.id,
                layer_id,
                track_idx,
                centroid_world[0],
                centroid_world[1],
                centroid_world[2],
            )
            self._log_placement_frame(lc, fx, track_idx)
            self._bake_asset(lc, fx, track_idx, centroid_world)

        # Apply every edit's staged rows in ONE cat per layer tensor (row indices are
        # already assigned arithmetically, so the result is identical to eager cats).
        lc.appender.flush()
        lc.node._validate_fields()
        log.info(
            "%s layer: baked %d overlay Gaussians across %d cluster(s) (layer now has %d Gaussians)",
            layer_id,
            lc.appender.appended,
            len(self.baked_clusters) - n_clusters_before,
            int(lc.node.get_num_gaussians()),
        )
        return lc.appender.appended

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
        # detection and replace_original must address ORIGINAL rows only.
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
    # World binds: synthetic static tracks for introduced props
    # ------------------------------------------------------------------
    def _materialize_world_binds(self, edits_by_layer: dict[str, list[EditSpec]]) -> None:
        """Synthesize one NEW constant-pose cuboid track per world-bound edit.

        Runs BEFORE any layer bake: each world edit's track is appended to the
        composite's GLOBAL track list, then every affected layer's ``LayerTrackIds``
        is re-derived so the new tracks become real layer tracks — with exactly the
        layer-local indices the appended Gaussian rows will be tagged with.

        Reload agreement holds by construction (``LayerTrackIds`` masks the global
        list in insertion order, and synthetic tracks append at the END, so their
        layer-local indices are ``old_n, old_n+1, ...`` both now and after the baked
        artifact reloads from ``datasource_summary.json``) — and is asserted anyway:
        a synthetic track captured by zero or by several layers would desynchronize
        row/track indexing, so that fails the bake loudly.
        """
        world_edits = [(layer_id, fx) for layer_id, edits in edits_by_layer.items() for fx in edits if fx.bind.is_world]
        if not world_edits:
            return
        if self.clip_window_us is None:
            raise AssetEditsBakeError(
                "document contains world binds but the baker was built without clip_window_us "
                "(the artifact's rig playback window)"
            )
        t0 = int(self.clip_window_us[0]) - _WORLD_TRACK_PAD_US
        t1 = int(self.clip_window_us[1]) + _WORLD_TRACK_PAD_US
        if t1 <= t0:
            raise AssetEditsBakeError(f"clip_window_us is degenerate: {self.clip_window_us}")
        device = self.composite.cuboid_tracks.device

        # Pre-append layer sizes: synthetic tracks land at indices old_n, old_n+1, ...
        old_layer_n = {
            layer_id: len(cast(RigidGaussianModel, self.composite.gaussians_nodes[layer_id]).cuboid_tracks.tracks_id)
            for layer_id in dict.fromkeys(layer_id for layer_id, _ in world_edits)
        }

        pending: list[tuple[str, EditSpec, CuboidTracks, CuboidTracks]] = []
        for layer_id, fx in world_edits:
            bind = fx.bind
            tid = bind.track_id or f"prop:{fx.id}"  # parse sets this; belt-and-braces for direct construction
            if tid in self.composite.cuboid_tracks.tracks_id:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: world bind track_id {tid!r} collides with an existing track in the artifact"
                )
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
                tracks_flags=[TrackFlags.DYNAMIC],
                cuboids_dims=[dims_world.astype(np.float32)],
                device=device,
            )
            ct_nurec = CuboidTracks.Ops.transform_with_frame_conversion(ct_world, self.world_to_nurec, None)
            self.composite.cuboid_tracks = CuboidTracks.Ops.concatenate([self.composite.cuboid_tracks, ct_nurec])
            pending.append((layer_id, fx, ct_world, ct_nurec))
            log.info(
                "  [world bind %s] synthetic track %r: label_class=%s dims=(%.2f, %.2f, %.2f) m "
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

        # Re-derive every affected layer's track partition from the grown global list
        # (the same rail GaussiansComposite.insert_asset rides). Unaffected layers keep
        # their existing LayerTrackIds: appends never shift existing global indices.
        for layer_id in old_layer_n:
            node = cast(RigidGaussianModel, self.composite.gaussians_nodes[layer_id])
            new_ids = LayerTrackIds(config=self.composite.obj_track_ids[layer_id].config)
            new_ids.initialize_from_tracks(self.composite.cuboid_tracks)
            self.composite.obj_track_ids[layer_id] = new_ids
            node.cuboid_tracks = new_ids.get_layer_tracks(self.composite.cuboid_tracks)

        # Membership probe across ALL layers: each synthetic track must be captured by
        # exactly its target layer.
        membership = {
            layer_id: set(self._probe_layer_track_ids(ids_obj.config))
            for layer_id, ids_obj in self.composite.obj_track_ids.items()
        }

        rank: dict[str, int] = {}
        for layer_id, fx, ct_world, ct_nurec in pending:
            tid = fx.bind.track_id or f"prop:{fx.id}"
            capturing = sorted(lid for lid, ids in membership.items() if tid in ids)
            if capturing != [layer_id]:
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: synthetic track {tid!r} (label_class={fx.bind.label_class!r}) must be "
                    f"captured by exactly its target layer {layer_id!r}, but is captured by {capturing or 'no layer'}. "
                    "Pick a label_class the target layer's track filter accepts and no other layer does."
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
            self._world_track_idx[fx.id] = idx
            self.synthetic_tracks.append(
                SyntheticTrack(
                    edit_id=fx.id,
                    layer_id=layer_id,
                    track_id_str=tid,
                    nurec_track_id=idx,
                    global_index=self.composite.cuboid_tracks.tracks_id.index(tid),
                    timestamps_us=(t0, t1),
                    pose_tquat_nurec=_tquat_rows(ct_nurec),
                    pose_tquat_world=_tquat_rows(ct_world),
                    dims_nurec=[float(x) for x in ct_nurec.cuboids_dims[0].detach().cpu().numpy()],
                    dims_world=[float(x) for x in ct_world.cuboids_dims[0].detach().cpu().numpy()],
                    label_class=fx.bind.label_class,
                )
            )

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
        """World-frame cuboid dims for a world bind: explicit, else the placed asset's bbox.

        Derivation mirrors ``_compute_assembly_transform``: the same DEFAULT-VISIBLE part
        set is rotated by the same placement euler (bboxing hidden state copies too would
        oversize the cuboid and let the visible assembly under-fill it), so with the
        default fixed scale the assembly fills its synthetic cuboid exactly and
        ``align: bbox`` rests its base at the authored position. The imported asset is
        cached for the bake to reuse.
        """
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
        if fx.bind.is_world:
            track_idx = self._world_track_idx[fx.id]
            tid = lc.ct.tracks_id[track_idx]
            if tid != (fx.bind.track_id or f"prop:{fx.id}"):
                raise AssetEditsBakeError(
                    f"edit {fx.id!r}: layer {lc.layer_id!r} track {track_idx} is {tid!r}, not the edit's "
                    "synthetic track — layer tracks changed between materialization and bake"
                )
            return track_idx
        track_idx = fx.bind.nurec_track_id
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

        The whole assembly (static parts + every control state's clusters) gets ONE
        shared transform — rotate, uniform scale, align translate — preserving the
        parts' relative layout (never per-part bbox recentring). Static parts bake
        VISIBLE and simply ride the track. Each controllable control (a native USD
        variantSet) bakes one cluster per state — default state visible, the rest
        hidden — and one BakedCluster per state with aspect ``"<control>:<state>"``,
        making the runtime an N-way variant selector. Non-controllable controls bake
        only their default state, visible, with no cluster.
        """
        prepared = self._prepared.pop(fx.id, None)
        if prepared is not None:
            asset_path, asset, resolved, parts = prepared
        else:
            asset_path, asset = self._load_asset(fx)
            resolved = self._resolve_controls(fx, asset)
            parts = self._collect_parts(fx, asset, resolved)
        xf = self._compute_assembly_transform(lc, fx, track_idx, parts)
        part_records, state_rows, state_opacities = self._append_parts(lc, track_idx, parts, xf)
        self.baked_clusters.extend(
            self._emit_clusters(lc, fx, track_idx, resolved, state_rows, state_opacities, centroid_world)
        )
        self.control_animations.extend(self._emit_control_animations(lc, fx, track_idx, resolved))
        replaced = self._replace_original(lc, fx, track_idx)
        self.asset_records.append(
            self._asset_record(lc, fx, track_idx, asset_path, asset, xf, resolved, part_records, replaced)
        )

    def _load_asset(self, fx: EditSpec) -> tuple[Path, Any]:
        """Resolve and import the edit's asset: a composed SimReady USD, or a static ``.ply``."""
        asset_path = Path(fx.asset)  # type: ignore[arg-type]
        if not asset_path.is_absolute():
            asset_path = (self.yaml_dir / asset_path).resolve()
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
        """Collect every part to bake, with its role and target visibility.

        Animated controls group their parts by ELEMENT (the prims driven by visibility
        windows/membership); an element whose payload is identical in every state it
        appears in bakes ONCE ("element" role, state-agnostic aspect) instead of one
        copy per state — geometry is baked once, the schedule is the only variation.
        Records the collapsed element keys in ``self._shared_elements`` (read back when
        clusters and animation tables emit).
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

    def _replace_original(self, lc: _LayerContext, fx: EditSpec, track_idx: int) -> int:
        """Hide every ORIGINAL (pre-bake) Gaussian riding this track, when requested.

        Recoverable by rebaking from the original artifact; the manifest records the
        criterion (cuboid id == track among the first original_layer_size rows) rather
        than hundreds of thousands of indices. MUST address the snapshot's rows, never
        the node's current ids — appended rows are tagged to the same track.
        """
        if not fx.replace_original:
            return 0
        orig_idx = np.nonzero(lc.snapshot.cuboid_ids == track_idx)[0]
        if orig_idx.size:
            idx_t = torch.tensor(orig_idx, device=lc.spec.device, dtype=torch.long)
            lc.node.densities.data[idx_t] = HIDDEN_DENSITY_LOGIT
        log.info(
            "  [asset %s] replace_original: hid %d captured gaussian(s) on track %d",
            fx.id,
            int(orig_idx.size),
            track_idx,
        )
        return int(orig_idx.size)

    def _asset_record(
        self,
        lc: _LayerContext,
        fx: EditSpec,
        track_idx: int,
        asset_path: Path,
        asset: Any,
        xf: _AssemblyTransform,
        resolved: list[_ResolvedControl],
        part_records: list[dict],
        replaced: int,
    ) -> dict:
        """The informational manifest record for one edit."""
        rx, ry, rz = xf.euler_deg
        bind_record: dict[str, Any] = {"type": fx.bind.type, "nurec_track_id": track_idx}
        if fx.bind.is_world:
            synth = next(s for s in self.synthetic_tracks if s.edit_id == fx.id)
            bind_record.update(
                {
                    "position": [float(x) for x in fx.bind.position or []],
                    "yaw_deg": float(fx.bind.yaw_deg),
                    "anchor": fx.bind.anchor,
                    "track_id": synth.track_id_str,
                    "dims": synth.dims_world,
                    "label_class": synth.label_class,
                }
            )
        return {
            "edit_id": fx.id,
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
