# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoring schema and loader for ``asset_edits.yaml``.

This module owns the hand-authored format only: the dataclasses, a YAML loader and strict
validation. It imports no torch or heavy NuRec code, so it parses and tests cheaply;
``nurec compile-asset-edits`` (``asset_edits.py``) consumes it and bakes the geometry.

Every edit names its operation with a required ``op``:

* ``add``: bake an asset onto an existing cuboid track (an attachment riding the captured
  actor) or, with a ``world`` bind, as a new actor at a fixed world pose.
* ``replace``: bake an asset onto an existing track and hide the captured actor. A render
  request without ``aspect_state`` shows the replacement in its default control states.
* ``clone``: a new actor with the appearance of a captured track (its trained Gaussians,
  duplicated) or of another edit's asset, on a new track that inherits the source's motion
  (optionally offset and time-shifted) or parks at a world pose.
* ``exclude``: hide a captured actor. The track stays in the scene metadata, so every other
  track keeps its index.

An asset is a composed SimReady USD whose variantSets are its controls: each controllable
state bakes as its own hidden Gaussian cluster and the manifest records how to address it.
Aspects are opaque strings to NuRec. A static prop may instead be a 3DGS ``.ply``: one splat
cloud without variantSets, so it takes no ``controls`` and no ``prim_path``. Edits default to
the ``dynamic_rigids`` layer; ``layer_id`` targets another Gaussian layer.

Format
------
::

    edits:
      # REPLACE: the asset rides the bound track, the captured actor on it is hidden.
      # `controllable: true` bakes every state's clusters and writes the control's animation
      # table into the manifest, so an animator drives it per frame (AspectState on the
      # render request). `semantics` binds proto enum value names to the control. Controls
      # without `controllable` bake only their default state.
      - id: truck_1
        op: replace
        bind: { type: nurec_track, nurec_track_id: 3 }
        asset: usd/configurable_asset/scene.usd
        placement: { align: bbox, scale: fit_length, yaw_deg: 0.0, offset: [0.0, 0.0, 0.0] }
        controls:
          panelState:                         # must be a variantSet on the asset
            default: off                      # state shown when undriven
            controllable: true
            semantics: [PANEL_VISIBILITY]
          appearanceMode:
            default: off
            controllable: true
            semantics: [APPEARANCE_MODE]

      # ADD on a track: an attachment. The captured actor stays and the asset rides its track
      # at its real size (an add is never fitted to the cuboid unless `scale` says so).
      # `align: bbox` centres it and rests it on the cuboid floor; `offset` lifts it onto the
      # roof (cuboid-local metres, here a 1.5 m tall car).
      - id: roof_rack_1
        op: add
        bind: { type: nurec_track, nurec_track_id: 3 }
        asset: usd/props/roof_rack/Body.usd
        placement: { align: bbox, offset: [0.0, 0.0, 1.5] }

      # ADD at a world pose: a new static actor on a synthesized constant-pose track spanning
      # the clip. `position` is world metres; `anchor: base` (default) rests the asset's base
      # there, `center` puts the cuboid center there. Cuboid dims default to the placed
      # asset's bbox; `scale` defaults to fixed 1.0.
      - id: barrier_1
        op: add
        bind: { type: world, position: [150.0, -3.5, 0.0], yaw_deg: 90.0 }
        asset: usd/props/barrier_waterfilled_01/Body.usd

      # A static prop may be a 3DGS `.ply` (Y-up file, rotated to Z-up like `edit_assets`).
      # Bank PLYs are unit-normalised: on a world bind give the real `dims` and fit to them.
      - id: cone_1
        op: add
        bind: { type: world, position: [152.0, -3.5, 0.0], dims: [0.4, 0.4, 0.7] }
        asset: ply/traffic_cone/9000014.ply
        placement: { scale: fit_length }

      # CLONE of a captured actor: track 12's trained Gaussians are duplicated onto a new
      # track that inherits track 12's motion, offset in the source's cuboid-local frame
      # (metres). `track_id` is the new actor's string id and must be unique.
      - id: lead_clone
        op: clone
        source: { nurec_track_id: 12 }
        motion: { inherit: true, offset: [0.0, -2.4, 0.0], yaw_deg: 0.0, time_shift_us: 0 }
        track_id: "9000221"

      # CLONE of another edit: that edit's asset and controls are baked again under this id
      # on a new track parked at a world pose. The clone's layer is the source's.
      - id: parked_truck
        op: clone
        source: { edit_id: truck_1 }
        motion: { world: { position: [140.0, 4.0, 0.0], yaw_deg: 180.0 } }
        track_id: "9000164"

      # EXCLUDE a captured actor: its Gaussians are hidden; the track stays.
      - id: drop_462
        op: exclude
        bind: { type: nurec_track, nurec_track_id: 35 }

Placement offsets are in the bound track's cuboid-local frame; the compile log's
``[placement]`` line reports that frame's axis roles and extents.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from nurec.utils.renamed_keys import normalize_renamed_key, normalize_renamed_value


# Layer an edit bakes into when it does not set its own ``layer_id``. Targeting
# another layer requires an explicit per-edit ``layer_id``.
DEFAULT_LAYER_ID = "dynamic_rigids"
# Current bind type and track-index key spellings.
TRACK_BIND_TYPE = "nurec_track"
TRACK_ID_KEY = "nurec_track_id"

# The edit operations, in the order the documentation introduces them.
SUPPORTED_OPS = ("add", "replace", "clone", "exclude")
# Currently supported binding types.
_SUPPORTED_BIND_TYPES = {TRACK_BIND_TYPE, "world"}
# Spellings written by asset-edit documents that predate the nre -> nurec rename.
_LEGACY_TRACK_BIND_TYPE = "nre_track"
_LEGACY_TRACK_ID_KEY = "nre_track_id"
# What a world bind's ``position`` names on the placed asset's cuboid.
_WORLD_ANCHORS = {"base", "center"}
# Label class a world bind's synthetic track carries unless overridden; must route
# the track to the edit's target layer (the bake asserts this).
DEFAULT_WORLD_LABEL_CLASS = "protruding_object"
# Suffix marking an edit's asset as a static 3DGS splat file rather than a SimReady USD.
PLY_SUFFIX = ".ply"
# Keys every edit may carry, then the per-op additions.
_COMMON_EDIT_KEYS = {"id", "op", "layer_id"}
_ASSET_EDIT_KEYS = _COMMON_EDIT_KEYS | {"bind", "asset", "placement", "controls", "prim_path"}
_EXCLUDE_EDIT_KEYS = _COMMON_EDIT_KEYS | {"bind"}
_CLONE_EDIT_KEYS = _COMMON_EDIT_KEYS | {"source", "motion", "track_id"}


class AssetEditsError(ValueError):
    """Raised for malformed ``asset_edits.yaml`` documents."""


def _norm_token(value: Any) -> str:
    """Normalise an aspect / key token to lowercase string.

    YAML 1.1 (PyYAML's safe_load) parses the bare words ``off``/``on``/``yes``/``no``
    as booleans, so an unquoted ``aspect: on`` or a bare ``off`` in an aspect list
    would arrive here as a Python ``bool``. Map those back to their intended strings
    so authors can write the natural ``off`` / ``on`` without quoting.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value).strip().lower()


def _as_vec3(value: Any, ctx: str) -> list[float]:
    """Validate and coerce a YAML ``[x, y, z]`` into a FINITE float triple; fail loud otherwise.

    Components go through :func:`_as_float`, so booleans, ``.nan`` and ``.inf`` are
    rejected instead of silently baking an invalid placement.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AssetEditsError(f"{ctx}: expected a 3-element [x, y, z], got {value!r}")
    return [_as_float(v, f"{ctx}[{i}]") for i, v in enumerate(value)]


def _as_float(value: Any, ctx: str) -> float:
    """Coerce a YAML scalar to float; fail loud (as AssetEditsError) on anything else.

    Rejects YAML booleans explicitly: ``float(True)`` would silently become 1.0.
    """
    if isinstance(value, bool):
        raise AssetEditsError(f"{ctx}: expected a number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssetEditsError(f"{ctx}: expected a number, got {value!r}") from exc
    if not math.isfinite(result):
        # YAML parses bare .nan/.inf; a non-finite placement is never intended.
        raise AssetEditsError(f"{ctx}: expected a finite number, got {value!r}")
    return result


def _as_int(value: Any, ctx: str) -> int:
    """Coerce a YAML scalar to int; reject non-integral numbers instead of truncating.

    ``int(7.9)`` silently addressing track 7 is exactly the mistake this guards.
    """
    f = _as_float(value, ctx)
    if not f.is_integer():
        raise AssetEditsError(f"{ctx}: expected an integer, got {value!r}")
    return int(f)


def _as_bool(value: Any, ctx: str) -> bool:
    """Require an actual YAML boolean; reject truthy strings like a quoted "false".

    YAML already parses bare true/false/on/off to bool; anything else (e.g. the
    quoted string "false", which is truthy) is an authoring mistake — fail loud.
    """
    if not isinstance(value, bool):
        raise AssetEditsError(f"{ctx}: expected a boolean (true/false), got {value!r}")
    return value


def _as_track_id(value: Any, ctx: str) -> str:
    """A non-empty string track id (the id a synthetic track carries downstream)."""
    if isinstance(value, bool) or value is None:
        raise AssetEditsError(f"{ctx}: expected a non-empty string track id, got {value!r}")
    text = str(value).strip()
    if not text:
        raise AssetEditsError(f"{ctx}: must be a non-empty string")
    _check_track_id_chars(text, ctx)
    return text


def _check_track_id_chars(track_id: str, ctx: str) -> None:
    """Refuse a track id NuRec would rewrite at load.

    Loading strips an ``@<source>`` suffix from every track id, so ``221@clone`` would come
    back as ``221``: a different id from the one the manifest records, and possibly the id of
    an existing track.
    """
    if "@" in track_id:
        raise AssetEditsError(
            f"{ctx}: track id {track_id!r} contains '@'; NuRec strips '@<source>' suffixes from track ids "
            f"at load, so it would come back as {track_id.split('@')[0]!r}"
        )


def _reject_unknown(d: Mapping[str, Any], known: set[str], ctx: str, what: str) -> None:
    """Fail loud on any key outside ``known`` (typos must never silently no-op)."""
    unknown = sorted(set(d) - known)
    if unknown:
        raise AssetEditsError(f"{ctx}: unknown {what} field(s) {unknown} (supported: {sorted(known)})")


def is_ply_asset(asset: Any) -> bool:
    """Whether an edit's ``asset`` names a static 3DGS ``.ply`` rather than a SimReady USD."""
    return str(asset).lower().endswith(PLY_SUFFIX)


@dataclass
class Binding:
    """Where an add/replace/exclude edit acts.

    ``nurec_track``: an existing layer-local cuboid track (the actor the asset replaces,
    rides, or that is excluded). ``world``: no source actor — the bake synthesizes a
    NEW constant-pose cuboid track at the given world pose and the asset rides that.
    """

    type: str
    # nurec_track binds only: layer-local cuboid track index.
    nurec_track_id: int = -1
    # world binds only ------------------------------------------------------
    position: Optional[list[float]] = None  # [x, y, z] world metres (required)
    yaw_deg: float = 0.0  # heading about world +Z
    anchor: str = "base"  # what `position` names: cuboid base center | cuboid center
    dims: Optional[list[float]] = None  # explicit cuboid [dx, dy, dz]; default: placed asset bbox
    label_class: str = DEFAULT_WORLD_LABEL_CLASS  # routes the synthetic track's layer partition
    track_id: Optional[str] = None  # string track id; defaults to "prop:<edit_id>" (set on parse)

    @property
    def is_world(self) -> bool:
        return self.type == "world"

    @staticmethod
    def from_dict(d: Mapping[str, Any], ctx: str) -> "Binding":
        """Parse and validate a ``bind:`` mapping (type defaults to ``nurec_track``).

        The pre-rename spellings ``nre_track`` / ``nre_track_id`` are accepted; the parsed
        binding always carries the current spelling.
        """
        if not isinstance(d, Mapping):
            raise AssetEditsError(f"{ctx}: bind must be a mapping, got {d!r}")
        original_keys = set(d)
        try:
            d = normalize_renamed_key(d, TRACK_ID_KEY, _LEGACY_TRACK_ID_KEY)
        except ValueError as e:
            raise AssetEditsError(f"{ctx}: {e}") from None
        btype = normalize_renamed_value(
            str(d.get("type", TRACK_BIND_TYPE)).strip().lower(), TRACK_BIND_TYPE, _LEGACY_TRACK_BIND_TYPE
        )
        if btype not in _SUPPORTED_BIND_TYPES:
            raise AssetEditsError(
                f"{ctx}: unsupported bind type {btype!r} (expected one of {sorted(_SUPPORTED_BIND_TYPES)})"
            )
        known = {"type"} | (
            {TRACK_ID_KEY}
            if btype == TRACK_BIND_TYPE
            else {"position", "yaw_deg", "anchor", "dims", "label_class", "track_id"}
        )
        # Report the spelling the author wrote, not the normalized one.
        unknown = sorted(
            k for k in original_keys if k not in known and not (k == _LEGACY_TRACK_ID_KEY and TRACK_ID_KEY in known)
        )
        if unknown:
            raise AssetEditsError(
                f"{ctx}: unknown field(s) {unknown} for bind type {btype!r} (supported: {sorted(known)})"
            )
        if btype == TRACK_BIND_TYPE:
            if TRACK_ID_KEY not in d:
                raise AssetEditsError(f"{ctx}: bind requires '{TRACK_ID_KEY}'")
            return Binding(type=btype, nurec_track_id=_as_int(d[TRACK_ID_KEY], f"{ctx}.nurec_track_id"))

        if "position" not in d:
            raise AssetEditsError(f"{ctx}: world bind requires 'position' ([x, y, z] world metres)")
        anchor = str(d.get("anchor", "base")).strip().lower()
        if anchor not in _WORLD_ANCHORS:
            raise AssetEditsError(f"{ctx}.anchor: expected one of {sorted(_WORLD_ANCHORS)}, got {anchor!r}")
        dims = d.get("dims")
        if dims is not None:
            dims = _as_vec3(dims, f"{ctx}.dims")
            if any(v <= 0.0 for v in dims):
                raise AssetEditsError(f"{ctx}.dims: components must be > 0, got {dims}")
        track_id = d.get("track_id")
        if track_id is not None:
            track_id = _as_track_id(track_id, f"{ctx}.track_id")
        return Binding(
            type=btype,
            position=_as_vec3(d["position"], f"{ctx}.position"),
            yaw_deg=_as_float(d.get("yaw_deg", 0.0), f"{ctx}.yaw_deg"),
            anchor=anchor,
            dims=dims,
            label_class=str(d.get("label_class", DEFAULT_WORLD_LABEL_CLASS)).strip(),
            track_id=track_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """The authored mapping form (re-parses to an equal Binding)."""
        if not self.is_world:
            return {"type": TRACK_BIND_TYPE, TRACK_ID_KEY: self.nurec_track_id}
        out: dict[str, Any] = {
            "type": "world",
            "position": list(self.position or []),
            "yaw_deg": self.yaw_deg,
            "anchor": self.anchor,
            "label_class": self.label_class,
        }
        if self.dims is not None:
            out["dims"] = list(self.dims)
        if self.track_id is not None:
            out["track_id"] = self.track_id
        return out


@dataclass
class AssetPlacement:
    """How a composed asset is placed into its bound track's cuboid-local frame.

    One SHARED transform is applied to the whole assembly (never per-part), preserving
    the parts' relative layout. ``align: bbox`` centers the assembly bbox on the cuboid
    in x/y and rests the bbox bottom on the cuboid floor (wheels on the ground);
    ``align: origin`` maps the asset's root-frame origin to the cuboid center.
    ``scale: fit_length`` uniformly scales the assembly so its bbox length matches the
    cuboid length (the default for a replace); a number is a fixed uniform scale instead
    (an add defaults to 1.0: real size).
    ``offset`` is a final cuboid-local trim applied after align/scale/rotation.
    """

    align: str = "bbox"  # bbox | origin
    scale_mode: str = "fit_length"  # fit_length | fixed
    scale: float = 1.0  # uniform scale; used when scale_mode == fixed
    yaw_deg: float = 0.0  # rotation about the cuboid up-axis (asset +x forward -> cuboid +length)
    rotation_deg: Optional[list[float]] = None  # full [rx, ry, rz] euler (deg) if yaw isn't enough
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # cuboid-local metres

    def euler_xyz_deg(self) -> tuple[float, float, float]:
        """Effective [rx, ry, rz] euler (deg); ``yaw_deg`` is added to the Z (up) component."""
        rx, ry, rz = self.rotation_deg or [0.0, 0.0, 0.0]
        return float(rx), float(ry), float(rz) + float(self.yaw_deg)

    @staticmethod
    def from_dict(d: Mapping[str, Any], ctx: str) -> "AssetPlacement":
        """Parse and validate a ``placement:`` mapping; unknown fields fail loud."""
        if not isinstance(d, Mapping):
            raise AssetEditsError(f"{ctx}: placement must be a mapping (align/scale/yaw_deg/offset), got {d!r}")
        _reject_unknown(d, {"align", "scale", "yaw_deg", "rotation_deg", "offset"}, ctx, "placement")
        align = str(d.get("align", "bbox")).strip().lower()
        if align not in {"bbox", "origin"}:
            raise AssetEditsError(f"{ctx}.align: expected 'bbox' or 'origin', got {align!r}")
        raw_scale = d.get("scale", "fit_length")
        if isinstance(raw_scale, str):
            if raw_scale.strip().lower() != "fit_length":
                raise AssetEditsError(f"{ctx}.scale: expected a number or 'fit_length', got {raw_scale!r}")
            scale_mode, scale = "fit_length", 1.0
        else:
            scale_mode, scale = "fixed", _as_float(raw_scale, f"{ctx}.scale")
            if scale <= 0.0:
                raise AssetEditsError(f"{ctx}.scale must be > 0 (got {scale})")
        rotation = d.get("rotation_deg")
        return AssetPlacement(
            align=align,
            scale_mode=scale_mode,
            scale=scale,
            yaw_deg=_as_float(d.get("yaw_deg", 0.0), f"{ctx}.yaw_deg"),
            rotation_deg=_as_vec3(rotation, f"{ctx}.rotation_deg") if rotation is not None else None,
            offset=_as_vec3(d.get("offset", [0.0, 0.0, 0.0]), f"{ctx}.offset"),
        )

    def to_dict(self) -> dict[str, Any]:
        """The authored mapping form (re-parses to an equal AssetPlacement)."""
        out: dict[str, Any] = {
            "align": self.align,
            "scale": "fit_length" if self.scale_mode == "fit_length" else self.scale,
            "yaw_deg": self.yaw_deg,
            "offset": list(self.offset),
        }
        if self.rotation_deg is not None:
            out["rotation_deg"] = list(self.rotation_deg)
        return out


@dataclass
class ControlSpec:
    """One native control of the asset: a variantSet driven at render time.

    ``name`` must be a variantSet on the asset's root prim (the compiler validates this
    against the asset and fails on an unknown set or state). ``default`` is the state
    baked VISIBLE — what renders when the control is not driven; ``None`` means the
    asset's authored selection.

    ``controllable: true`` bakes ALL states' clusters and emits the control's animation
    table into the manifest, so an external animator can drive it over gRPC.
    ``semantics`` optionally binds the control to sender-defined enum value names
    (e.g. ``PANEL_VISIBILITY``) so animators map incoming semantic state to this
    control without name heuristics. These names are illustrative, not built-in enums.
    """

    # AUTHORED FIELDS ONLY: a parsed document means the same thing at every point in the
    # pipeline. Derived data travels separately — variant animation read from the asset
    # and passed into AssetEditsBaker, bake results accumulated on the baker — so nothing
    # here is back-filled as the bake progresses.
    name: str
    default: Optional[str] = None
    # Bake all states for external driving over gRPC; see class docstring.
    controllable: bool = False
    # Proto enum value names this control answers to (uppercased on load).
    semantics: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(name: Any, d: Mapping[str, Any], ctx: str) -> "ControlSpec":
        """Parse and validate one ``controls:`` entry; unknown fields fail loud."""
        if not isinstance(d, Mapping):
            raise AssetEditsError(f"{ctx}: control entry must be a mapping (default/controllable/semantics), got {d!r}")
        _reject_unknown(d, {"default", "controllable", "semantics"}, ctx, "control")
        semantics = d.get("semantics") or []
        if not isinstance(semantics, (list, tuple)) or not all(isinstance(s, str) for s in semantics):
            raise AssetEditsError(f"{ctx}.semantics: expected a list of enum value names, got {semantics!r}")
        return ControlSpec(
            name=_norm_token(name),
            default=_norm_token(d["default"]) if d.get("default") is not None else None,
            controllable=_as_bool(d.get("controllable", False), f"{ctx}.controllable"),
            semantics=[s.strip().upper() for s in semantics],
        )

    def to_dict(self) -> dict[str, Any]:
        """The authored mapping form (the control's ``controls:`` value)."""
        out: dict[str, Any] = {"controllable": self.controllable}
        if self.default is not None:
            out["default"] = self.default
        if self.semantics:
            out["semantics"] = list(self.semantics)
        return out


@dataclass
class CloneMotion:
    """How a cloned actor moves.

    ``inherit``: the clone rides a copy of the source track's (calibrated) trajectory,
    every keyframe transformed by ``offset``/``yaw_deg`` in the SOURCE track's
    cuboid-local frame (metres / degrees about the cuboid up axis) and shifted by
    ``time_shift_us``. ``world``: the clone is parked at a fixed world pose for the
    whole clip, exactly like a world bind (``anchor`` says what ``position`` names).
    """

    kind: str  # inherit | world
    # inherit -------------------------------------------------------------------
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw_deg: float = 0.0
    time_shift_us: int = 0
    # world ---------------------------------------------------------------------
    position: Optional[list[float]] = None
    anchor: str = "base"

    @property
    def is_world(self) -> bool:
        """True when the clone parks at a world pose instead of inheriting motion."""
        return self.kind == "world"

    @staticmethod
    def from_dict(d: Mapping[str, Any], ctx: str) -> "CloneMotion":
        """Parse ``motion: {inherit: true, ...}`` or ``motion: {world: {...}}``."""
        if not isinstance(d, Mapping):
            raise AssetEditsError(
                f"{ctx}: motion must be a mapping ({{inherit: true, ...}} or {{world: {{...}}}}), got {d!r}"
            )
        has_inherit = "inherit" in d
        has_world = "world" in d
        if has_inherit == has_world:
            raise AssetEditsError(f"{ctx}: motion needs exactly one of 'inherit: true' or 'world: {{position, ...}}'")
        if has_inherit:
            _reject_unknown(d, {"inherit", "offset", "yaw_deg", "time_shift_us"}, ctx, "motion")
            if _as_bool(d["inherit"], f"{ctx}.inherit") is not True:
                raise AssetEditsError(f"{ctx}.inherit: must be true (omit the key to park the clone with 'world')")
            return CloneMotion(
                kind="inherit",
                offset=_as_vec3(d.get("offset", [0.0, 0.0, 0.0]), f"{ctx}.offset"),
                yaw_deg=_as_float(d.get("yaw_deg", 0.0), f"{ctx}.yaw_deg"),
                time_shift_us=_as_int(d.get("time_shift_us", 0), f"{ctx}.time_shift_us"),
            )
        _reject_unknown(d, {"world"}, ctx, "motion")
        w = d["world"]
        if not isinstance(w, Mapping):
            raise AssetEditsError(
                f"{ctx}.world: expected a mapping with 'position' (and optional yaw_deg/anchor), got {w!r}"
            )
        _reject_unknown(w, {"position", "yaw_deg", "anchor"}, f"{ctx}.world", "motion.world")
        if "position" not in w:
            raise AssetEditsError(f"{ctx}.world: requires 'position' ([x, y, z] world metres)")
        anchor = str(w.get("anchor", "base")).strip().lower()
        if anchor not in _WORLD_ANCHORS:
            raise AssetEditsError(f"{ctx}.world.anchor: expected one of {sorted(_WORLD_ANCHORS)}, got {anchor!r}")
        return CloneMotion(
            kind="world",
            position=_as_vec3(w["position"], f"{ctx}.world.position"),
            yaw_deg=_as_float(w.get("yaw_deg", 0.0), f"{ctx}.world.yaw_deg"),
            anchor=anchor,
        )

    def to_dict(self) -> dict[str, Any]:
        """The authored mapping form (re-parses to an equal CloneMotion)."""
        if self.is_world:
            return {"world": {"position": list(self.position or []), "yaw_deg": self.yaw_deg, "anchor": self.anchor}}
        return {
            "inherit": True,
            "offset": list(self.offset),
            "yaw_deg": self.yaw_deg,
            "time_shift_us": self.time_shift_us,
        }


@dataclass
class CloneSpec:
    """What a clone copies and how it moves.

    Exactly one of ``source_nurec_track_id`` (a captured track of the clone's layer: its
    trained Gaussians are duplicated) and ``source_edit_id`` (an add/replace edit of the
    same document: its asset, placement and controls are baked again under the clone's
    id) is set. ``track_id`` is the new track's string id.
    """

    motion: CloneMotion
    track_id: str
    source_nurec_track_id: Optional[int] = None
    source_edit_id: Optional[str] = None

    @property
    def is_captured(self) -> bool:
        """Whether the clone copies a captured track (as opposed to another edit's asset)."""
        return self.source_edit_id is None

    @staticmethod
    def from_dict(source: Any, motion: Any, track_id: Any, ctx: str) -> "CloneSpec":
        """Parse the clone's ``source`` / ``motion`` / ``track_id`` keys."""
        if not isinstance(source, Mapping):
            raise AssetEditsError(
                f"{ctx}.source: expected a mapping with 'nurec_track_id' or 'edit_id', got {source!r}"
            )
        _reject_unknown(source, {"nurec_track_id", "edit_id"}, f"{ctx}.source", "clone source")
        if ("nurec_track_id" in source) == ("edit_id" in source):
            raise AssetEditsError(
                f"{ctx}.source: needs exactly one of 'nurec_track_id' (a captured track) or 'edit_id'"
            )
        source_track: Optional[int] = None
        source_edit: Optional[str] = None
        if "nurec_track_id" in source:
            source_track = _as_int(source["nurec_track_id"], f"{ctx}.source.nurec_track_id")
        else:
            source_edit = str(source["edit_id"]).strip()
            if not source_edit:
                raise AssetEditsError(f"{ctx}.source.edit_id: must be a non-empty edit id")
        return CloneSpec(
            motion=CloneMotion.from_dict(motion, f"{ctx}.motion"),
            track_id=_as_track_id(track_id, f"{ctx}.track_id"),
            source_nurec_track_id=source_track,
            source_edit_id=source_edit,
        )

    def source_dict(self) -> dict[str, Any]:
        """The authored ``source:`` mapping."""
        if self.is_captured:
            return {"nurec_track_id": self.source_nurec_track_id}
        return {"edit_id": self.source_edit_id}


@dataclass
class EditSpec:
    """One asset edit (see the module docstring for the four operations).

    ``bind`` is set for add/replace/exclude, ``asset`` for add/replace, ``clone`` for
    clone; the parser enforces the per-op key sets, so the fields an op does not use
    are ``None``.
    """

    id: str
    op: str
    # Gaussian layer to bake this edit into (resolved at parse: an absent YAML
    # ``layer_id`` becomes ``DEFAULT_LAYER_ID``, so consumers never re-resolve it; a
    # clone of another edit takes that edit's layer).
    layer_id: str = DEFAULT_LAYER_ID
    bind: Optional[Binding] = None
    asset: Optional[str] = None  # composed stage (e.g. scene.usd) or a 3DGS .ply; relative to the assets root
    # Always present for add/replace: from_dict builds one from `placement: {}` when the key is absent.
    placement: AssetPlacement = field(default_factory=AssetPlacement)
    controls: list[ControlSpec] = field(default_factory=list)
    prim_path: Optional[str] = None  # optional explicit root prim inside the asset
    clone: Optional[CloneSpec] = None

    # ------------------------------------------------------------------ predicates
    @property
    def is_clone(self) -> bool:
        """True for an ``op: clone`` edit."""
        return self.op == "clone"

    @property
    def is_exclude(self) -> bool:
        """True for an ``op: exclude`` edit."""
        return self.op == "exclude"

    @property
    def has_asset(self) -> bool:
        """Whether the edit bakes an asset of its own (add / replace)."""
        return self.op in ("add", "replace")

    @property
    def hides_original(self) -> bool:
        """Whether the edit hides the captured actor on its bound track (replace / exclude)."""
        return self.op in ("replace", "exclude")

    @property
    def is_world(self) -> bool:
        """An add at a world pose (a new prop on a synthetic constant-pose track)."""
        return self.bind is not None and self.bind.is_world

    @property
    def binds_synthetic_track(self) -> bool:
        """Whether the edit's track is created by the bake (world binds and clones)."""
        return self.is_world or self.is_clone

    @property
    def synthetic_track_id(self) -> Optional[str]:
        """The string id of the track this edit synthesizes, if any."""
        if self.is_clone:
            assert self.clone is not None
            return self.clone.track_id
        if self.is_world:
            assert self.bind is not None
            return self.bind.track_id
        return None

    @property
    def nurec_track_id(self) -> int:
        """The bound layer-local track index of a nurec_track edit (add-on-track / replace / exclude)."""
        if self.bind is None or self.bind.is_world:
            raise ValueError(f"edit {self.id!r} ({self.op}) does not bind an existing track")
        return self.bind.nurec_track_id

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def from_dict(d: Mapping[str, Any], ctx: str) -> "EditSpec":
        """Parse and validate one ``edits:`` entry; unknown fields fail loud."""
        if not isinstance(d, Mapping):
            raise AssetEditsError(f"{ctx}: edit entry must be a mapping, got {d!r}")
        if "id" not in d:
            raise AssetEditsError(f"{ctx}: edit requires an 'id'")
        fid = str(d["id"])
        if not fid.strip():
            raise AssetEditsError(f"{ctx}: edit 'id' must be a non-empty string")
        if "op" not in d:
            raise AssetEditsError(f"{ctx} ({fid}): edit requires an 'op' (one of {list(SUPPORTED_OPS)})")
        op = str(d["op"]).strip().lower()
        if op not in SUPPORTED_OPS:
            raise AssetEditsError(f"{ctx} ({fid}): unsupported op {d['op']!r} (expected one of {list(SUPPORTED_OPS)})")

        layer_id = DEFAULT_LAYER_ID
        if d.get("layer_id") is not None:
            layer_id = str(d["layer_id"]).strip()
            if not layer_id:
                raise AssetEditsError(f"{ctx} ({fid}).layer_id: must be a non-empty layer id")

        if op == "clone":
            _reject_unknown(d, _CLONE_EDIT_KEYS, f"{ctx} ({fid})", "clone edit")
            for key in ("source", "motion", "track_id"):
                if key not in d:
                    raise AssetEditsError(f"{ctx} ({fid}): a clone edit requires '{key}'")
            clone = CloneSpec.from_dict(d["source"], d["motion"], d["track_id"], f"{ctx} ({fid})")
            return EditSpec(id=fid, op=op, layer_id=layer_id, clone=clone)

        if "bind" not in d:
            raise AssetEditsError(f"{ctx} ({fid}): edit requires a 'bind'")
        bind = Binding.from_dict(d["bind"], f"{ctx} ({fid}).bind")

        if op == "exclude":
            _reject_unknown(d, _EXCLUDE_EDIT_KEYS, f"{ctx} ({fid})", "exclude edit")
            if bind.is_world:
                raise AssetEditsError(f"{ctx} ({fid}): an exclude edit binds an existing track (bind.type nurec_track)")
            return EditSpec(id=fid, op=op, layer_id=layer_id, bind=bind)

        _reject_unknown(d, _ASSET_EDIT_KEYS, f"{ctx} ({fid})", "edit")
        if not d.get("asset"):
            raise AssetEditsError(
                f"{ctx} ({fid}): edit requires an 'asset' (a composed SimReady USD, or a .ply for a static prop)"
            )
        if op == "replace" and bind.is_world:
            raise AssetEditsError(
                f"{ctx} ({fid}): a replace edit binds an existing track (there is no captured actor at a world pose;"
                " use op: add to introduce a new one)"
            )

        raw_controls = d.get("controls") or {}
        if not isinstance(raw_controls, Mapping):
            raise AssetEditsError(f"{ctx} ({fid}).controls: expected a mapping of variantSet -> control spec")
        controls = [
            ControlSpec.from_dict(name, spec or {}, f"{ctx} ({fid}).controls[{name}]")
            for name, spec in raw_controls.items()
        ]
        seen_names = [c.name for c in controls]
        dupes = sorted({n for n in seen_names if seen_names.count(n) > 1})
        if dupes:
            raise AssetEditsError(f"{ctx} ({fid}).controls: duplicate control name(s) after normalization: {dupes}")

        if is_ply_asset(d["asset"]):
            # One splat cloud, no variantSets: nothing for a control to switch, no prim to pick.
            if controls:
                raise AssetEditsError(
                    f"{ctx} ({fid}): a .ply asset is a static splat cloud with no variantSets, so it takes no "
                    f"controls (got {[c.name for c in controls]}); controllable edits need a SimReady USD asset"
                )
            if d.get("prim_path") is not None:
                raise AssetEditsError(f"{ctx} ({fid}): prim_path does not apply to a .ply asset (it has no prims)")

        raw_placement = d.get("placement") or {}
        placement = AssetPlacement.from_dict(raw_placement, f"{ctx} ({fid}).placement")

        if bind.is_world:
            if bind.track_id is None:
                bind.track_id = f"prop:{fid}"
                _check_track_id_chars(bind.track_id, f"{ctx} ({fid}).bind.track_id (defaults to 'prop:<id>')")
            if placement.scale_mode == "fit_length" and "scale" in raw_placement and bind.dims is None:
                raise AssetEditsError(
                    f"{ctx} ({fid}).placement.scale: 'fit_length' on a world bind requires explicit bind.dims"
                    " (there is no source actor to fit)"
                )
        if op == "add" and "scale" not in raw_placement:
            # Only a replace stands in for the actor, so only a replace is fitted to its cuboid by
            # default. A new prop (world bind) or an attachment riding an actor keeps its authored
            # real-world size unless the author scales explicitly.
            placement.scale_mode, placement.scale = "fixed", 1.0

        return EditSpec(
            id=fid,
            op=op,
            layer_id=layer_id,
            bind=bind,
            asset=str(d["asset"]),
            placement=placement,
            controls=controls,
            prim_path=str(d["prim_path"]) if d.get("prim_path") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """The authored mapping form of this edit (re-parses to an equal EditSpec).

        Every resolved default is written out explicitly (layer, placement, the world
        bind's track id), so a re-parse cannot re-derive anything differently.
        """
        out: dict[str, Any] = {"id": self.id, "op": self.op, "layer_id": self.layer_id}
        if self.is_clone:
            assert self.clone is not None
            out["source"] = self.clone.source_dict()
            out["motion"] = self.clone.motion.to_dict()
            out["track_id"] = self.clone.track_id
            return out
        assert self.bind is not None
        out["bind"] = self.bind.to_dict()
        if self.is_exclude:
            return out
        out["asset"] = self.asset
        out["placement"] = self.placement.to_dict()
        out["controls"] = {c.name: c.to_dict() for c in self.controls}
        if self.prim_path is not None:
            out["prim_path"] = self.prim_path
        return out


def binding_key(fx: EditSpec) -> tuple[str, Any]:
    """The track an edit claims, as a hashable key.

    The schedule registry and the runtime are keyed per ``(layer, nurec_track_id)``: a
    track's active aspect set applies to EVERY cluster on the pair, so two edits on one
    existing track would clobber each other. Synthetic tracks (world binds, clones) are
    keyed by their string id instead — each gets its own track, so an exclude of the
    source track and a clone of it may coexist.
    """
    if fx.binds_synthetic_track:
        return ("synthetic", fx.synthetic_track_id)
    return (fx.layer_id, fx.nurec_track_id)


@dataclass
class AssetEditsDoc:
    """A parsed, validated ``asset_edits.yaml`` (every edit's ``layer_id`` is resolved)."""

    edits: list[EditSpec]

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AssetEditsDoc":
        """Parse and validate a whole document (top-level ``edits:`` list)."""
        if not isinstance(data, Mapping):
            raise AssetEditsError("top-level document must be a mapping")
        unknown = sorted(set(data) - {"edits"})
        if unknown:
            raise AssetEditsError(f"unknown top-level field(s) {unknown} (supported: ['edits'])")
        raw_edits = data.get("edits", [])
        if not isinstance(raw_edits, (list, tuple)):
            raise AssetEditsError(f"'edits' must be a list of edit mappings, got {raw_edits!r}")
        edits = [EditSpec.from_dict(e, f"edits[{i}]") for i, e in enumerate(raw_edits)]
        if not edits:
            raise AssetEditsError("document has no edits")
        # Edit ids key the baker's bookkeeping (asset cache, schedules, manifests):
        # a duplicate would silently bake the first edit's asset in both places.
        seen_ids = [e.id for e in edits]
        dupes = sorted({i for i in seen_ids if seen_ids.count(i) > 1})
        if dupes:
            raise AssetEditsError(f"duplicate edit id(s) {dupes}: every edit needs a unique 'id'")
        by_id = {e.id: e for e in edits}

        # Clone-of-edit: the source must be an asset edit of this document, and the
        # clone bakes into the source's layer (an explicit different layer is a typo,
        # not a feature: the copied geometry only makes sense on the source's layer).
        # Only an omitted layer_id is inherited: the parsed value cannot tell an omitted
        # one from an explicit DEFAULT_LAYER_ID, so the authored mapping decides.
        for raw, fx in zip(raw_edits, edits):
            if not fx.is_clone:
                continue
            assert fx.clone is not None
            if fx.clone.is_captured:
                continue
            src = by_id.get(fx.clone.source_edit_id or "")
            if src is None:
                raise AssetEditsError(
                    f"edit {fx.id!r}: clone source edit {fx.clone.source_edit_id!r} is not in this document"
                )
            if not src.has_asset:
                article = "an" if src.op[0] in "aeiou" else "a"
                raise AssetEditsError(
                    f"edit {fx.id!r}: clone source {src.id!r} is {article} {src.op} edit; only add/replace edits "
                    "(which bake an asset) can be cloned by edit id — clone a captured track with "
                    "source.nurec_track_id instead"
                )
            if raw.get("layer_id") is not None and fx.layer_id != src.layer_id:
                raise AssetEditsError(
                    f"edit {fx.id!r}: layer_id {fx.layer_id!r} differs from its clone source {src.id!r} "
                    f"(layer {src.layer_id!r}); a clone bakes into its source's layer"
                )
            fx.layer_id = src.layer_id

        # Synthetic track ids (world binds, clones) name NEW tracks: two edits claiming
        # one id would resolve to the same track and desynchronize row/track indexing.
        synthetic = [fx.synthetic_track_id for fx in edits if fx.binds_synthetic_track]
        dupes = sorted({t for t in synthetic if t is not None and synthetic.count(t) > 1})
        if dupes:
            raise AssetEditsError(
                f"synthetic track id(s) {dupes} are claimed by more than one edit (world binds / clones)"
            )
        return AssetEditsDoc(edits=edits)

    def to_dict(self) -> dict[str, Any]:
        """The authored document form (re-parses to an equal document)."""
        return {"edits": [fx.to_dict() for fx in self.edits]}

    def by_id(self, edit_id: str) -> EditSpec:
        """The edit with ``edit_id`` (the document guarantees uniqueness)."""
        for fx in self.edits:
            if fx.id == edit_id:
                return fx
        raise KeyError(edit_id)

    def appearance_of(self, fx: EditSpec) -> EditSpec:
        """The edit whose asset/placement/controls/prim_path define ``fx``'s appearance.

        ``fx`` itself for add/replace; the SOURCE edit for a clone of an edit. Captured
        clones and excludes have no asset appearance (``None``-free by contract: callers
        check ``is_clone``/``clone.is_captured`` first).
        """
        if fx.is_clone and fx.clone is not None and not fx.clone.is_captured:
            return self.by_id(fx.clone.source_edit_id or "")
        return fx

    @property
    def needs_clip_window(self) -> bool:
        """Whether any edit synthesizes a constant-pose track spanning the clip window."""
        return any(
            fx.is_world or (fx.is_clone and fx.clone is not None and fx.clone.motion.is_world) for fx in self.edits
        )


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated mapping key.

    YAML resolves a repeated key to its last value, which no later validation can see: a
    repeated ``bind:`` binds the asset to a different track, a repeated control entry drops
    the first one's ``controllable`` and ``semantics``. Both are silent.
    """


def _no_duplicate_keys(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    """Construct a mapping, raising on any key that appears more than once."""
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            mark = key_node.start_mark
            raise AssetEditsError(
                f"duplicate key {key!r} at line {mark.line + 1}, column {mark.column + 1}: "
                "YAML would silently keep only the last one"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)  # type: ignore[arg-type]


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def load_asset_edits(path: Path) -> AssetEditsDoc:
    """Parse and validate an ``asset_edits.yaml`` document."""
    with open(path, "r") as f:
        try:
            data = yaml.load(f, Loader=_StrictLoader)  # noqa: S506 - _StrictLoader derives from SafeLoader
        except yaml.YAMLError as exc:
            # PyYAML's scanner/parser errors are not AssetEditsError, and the CLI only
            # translates AssetEditsError — without this a bad indent prints a traceback.
            raise AssetEditsError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        raise AssetEditsError(f"{path} is empty")
    return AssetEditsDoc.from_dict(data)


__all__ = [
    "DEFAULT_LAYER_ID",
    "DEFAULT_WORLD_LABEL_CLASS",
    "SUPPORTED_OPS",
    "AssetEditsError",
    "Binding",
    "AssetPlacement",
    "ControlSpec",
    "CloneMotion",
    "CloneSpec",
    "EditSpec",
    "AssetEditsDoc",
    "binding_key",
    "is_ply_asset",
    "load_asset_edits",
]
