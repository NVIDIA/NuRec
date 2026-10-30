# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nurec compile-asset-edits``: the command-line entry for the asset-edits bake.

Loads and validates the ``asset_edits.yaml`` document, materializes the trained
artifact, runs :class:`~nurec.utils.edit.asset_edits.AssetEditsBaker`, then splices the
baked layer tensors into the original checkpoint and repacks the USDZ with the
geometry manifest embedded. Everything artifact/zip-shaped lives here; the bake
itself is pure in-memory work in ``asset_edits.py``.

The end-to-end pipeline is documented once, in ``nurec/utils/edit/README.md``.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile

from collections.abc import Collection
from pathlib import Path
from typing import Optional, cast

import click
import torch

from nurec.artifact import Artifact
from nurec.asset_edits.runtime import EDIT_MANIFEST_FILENAME, EditManifest, layer_track_digest
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.render.render import RenderableModel
from nurec.utils.edit.asset_edits import AssetEditsBakeError, AssetEditsBaker, SyntheticTrack
from nurec.utils.edit.asset_edits_schema import (
    AssetEditsDoc,
    AssetEditsError,
    EditSpec,
    load_asset_edits,
)
from nurec.utils.edit.layer_append import LAYER_PARAM_NAMES
from nurec.utils.edit.variant_animation import (
    VariantAnimation,
    VariantAnimationError,
    read_variant_animation,
)
from nurec.utils.types import FrameConversion, RigTrajectories


log = logging.getLogger(__name__)


def _resolve_world_to_nurec(renderable: RenderableModel, artifact: Artifact) -> FrameConversion:
    """The artifact's world <-> NuRec frame conversion, from the model or the rig trajectories.

    Every world-frame offset in every edit is expressed through this, so which source
    supplied it is worth a log line rather than a silent fallback.

    ``_world_to_nurec`` is private; RenderableModel exposes no accessor for it, and adding
    one is a change to a file this feature does not otherwise touch. Reading it here is
    safe (it is a declared attribute with a ``None`` default, so a rename fails loudly at
    the fallback rather than silently), but it is the reason this helper logs its source.
    """
    world_to_nurec = getattr(renderable, "_world_to_nurec", None)
    if world_to_nurec is not None:
        log.info("world_to_nurec frame conversion: taken from the loaded model.")
        return world_to_nurec
    try:
        resolved = RigTrajectories.from_dict(artifact.rig_trajectories).world_to_nurec
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(
            f"Could not resolve the world_to_nurec frame conversion: the model carries none and "
            f"the artifact's rig trajectories are unusable ({type(exc).__name__}: {exc}). "
            "World-frame offsets cannot be placed without it."
        ) from exc
    log.info("world_to_nurec frame conversion: derived from the artifact's rig trajectories.")
    return resolved


def _read_variant_animations(doc: AssetEditsDoc, yaml_dir: Path) -> dict[tuple[str, str], VariantAnimation]:
    """Read each scheduled control's USD-native variant-state animation from its asset.

    For every edit's control whose name matches a root variantSet on the asset, extract
    per-state visibility animation from INSIDE the variant contents
    (variant_animation.read_variant_animation). This is what makes the asset's own
    animation — the same samples usdview plays — the schedule's source of truth.

    Returns a ``(edit_id, control_name) -> VariantAnimation`` map for the baker. Purely
    static variantSets (e.g. panelState) are simply absent from it; nothing is written
    back onto the authored document.
    """
    animations: dict[tuple[str, str], VariantAnimation] = {}
    for fx in doc.edits:
        if not fx.controls:
            continue
        asset_path = Path(fx.asset)  # type: ignore[arg-type]
        if not asset_path.is_absolute():
            asset_path = (yaml_dir / asset_path).resolve()
        if not asset_path.exists():
            raise click.ClickException(f"asset not found: {asset_path} (edit {fx.id!r}, asset={fx.asset!r})")
        for spec in fx.controls:
            try:
                anim = read_variant_animation(asset_path, spec.name, root_prim_path=fx.prim_path)
            except VariantAnimationError as exc:
                raise click.ClickException(
                    f"edit {fx.id!r} control {spec.name!r}: bad variant animation: {exc}"
                ) from exc
            if anim is None or not anim.is_animated:
                continue
            animations[(fx.id, spec.name)] = anim
            log.info(
                "edit %s: control %s has USD-native animated states %s (cycle %.3fs)",
                fx.id,
                spec.name,
                sorted(s for s, w in anim.windows_by_state.items() if w),
                anim.cycle_s,
            )
    return animations


def _splice_synthetic_tracks(
    state_dict: dict,
    layer_id: str,
    synths: list[SyntheticTrack],
    old_n_tracks: int,
    old_n_poses: int,
    *,
    track_albedo_identity: Optional[torch.Tensor] = None,
    track_scale_identity: Optional[torch.Tensor] = None,
) -> list[str]:
    """Extend every track-count-sized checkpoint entry of ``layer_id`` for K new tracks.

    World binds append their synthetic tracks at the END of the layer's track list, so
    every per-track / per-keyframe-pose tensor extends positionally: identity rows for
    the calibration deltas (2 keyframes per synthetic track), the tracks' timestamp
    ranges for the time embedding, and a bumped ``n_tracks`` in the layer's extra
    state (asserted by ``RigidGaussianModel.set_extra_state`` at load). The composite's
    persisted ``obj_track_ids`` list for the layer is extended to match.

    Pure state-dict surgery (unit-testable without a model); returns the mutated keys.
    A final scan raises on any remaining tensor under the layer prefix whose leading
    dimension still matches the OLD track/pose count — an unhandled track-sized tensor
    would desynchronize silently at load, so fail the bake instead.
    """
    prefix = f"model.gaussians_nodes.{layer_id}."
    k = len(synths)
    handled: list[str] = []

    def _cat(key: str, rows: torch.Tensor, expect_n: int) -> None:
        cur = state_dict[key]
        if cur.shape[0] != expect_n:
            raise click.ClickException(
                f"checkpoint tensor {key} has leading dim {cur.shape[0]}, expected {expect_n}; "
                "cannot splice synthetic tracks"
            )
        state_dict[key] = torch.cat([cur, rows.to(dtype=cur.dtype)], dim=0).contiguous()
        handled.append(key)

    key_q = prefix + "tracks_calib.tracks_delta_q"
    key_t = prefix + "tracks_calib.tracks_delta_t"
    if key_q in state_dict:
        _cat(key_q, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2 * k, 1), old_n_poses)
    else:
        log.info("skipping %s: not present in original state_dict", key_q)
    if key_t in state_dict:
        _cat(key_t, torch.zeros(2 * k, 3), old_n_poses)

    key_ranges = prefix + "time_embed.timestamps_us_ranges"
    if key_ranges in state_dict:
        ranges = torch.tensor([[int(s.timestamps_us[0]), int(s.timestamps_us[1])] for s in synths])
        _cat(key_ranges, ranges, old_n_tracks)

    key_u = prefix + "time_embed.u"
    if key_u in state_dict:
        # IndividualStepTimeInputEmbedding.u is (n_tracks, n_dims, n_steps); new tracks
        # get the same fresh linspace block __init__ would build.
        _, n_dims, n_steps = state_dict[key_u].shape
        fresh = (
            torch.linspace(0.0, 1.0, n_steps * n_dims + 2)[1:-1]
            .reshape(1, n_steps, n_dims)
            .transpose(1, 2)
            .repeat(k, 1, 1)
        )
        _cat(key_u, fresh, old_n_tracks)

    for key_suffix, identity in (("track_albedos", track_albedo_identity), ("track_scales", track_scale_identity)):
        key = prefix + key_suffix
        if key in state_dict:
            if identity is None:
                raise click.ClickException(
                    f"checkpoint has {key} but no identity row was provided; cannot splice synthetic tracks"
                )
            row = identity.reshape((1,) + tuple(state_dict[key].shape[1:]))
            _cat(key, row.repeat(k, *([1] * (row.ndim - 1))), old_n_tracks)

    key_extra = prefix + "_extra_state"
    extra = state_dict.get(key_extra)
    if isinstance(extra, dict) and "n_tracks" in extra:
        if int(extra["n_tracks"]) != old_n_tracks:
            raise click.ClickException(f"{key_extra} records n_tracks={extra['n_tracks']}, expected {old_n_tracks}")
        extra["n_tracks"] = old_n_tracks + k
        handled.append(key_extra)

    # The composite persists each layer's track-id list alongside the node's
    # n_tracks; append the synthetic ids so both stay in step. (Today's loader
    # re-derives the partition from config and ignores this list, but persisted
    # state that contradicts n_tracks is a trap for any future restore path.)
    key_model_extra = "model._extra_state"
    model_extra = state_dict.get(key_model_extra)
    if isinstance(model_extra, dict):
        layer_track_ids = model_extra.get("obj_track_ids", {}).get(layer_id)
        if layer_track_ids is not None:
            if len(layer_track_ids) != old_n_tracks:
                raise click.ClickException(
                    f"{key_model_extra} obj_track_ids[{layer_id!r}] has {len(layer_track_ids)} entries, "
                    f"expected {old_n_tracks}; cannot splice synthetic tracks"
                )
            layer_track_ids.extend(s.track_id_str for s in synths)
            handled.append(key_model_extra)

    # Safety net: any tensor still sized by the OLD track/pose count is a
    # track-sized entry this splice does not know about.
    per_gaussian = {prefix + name for name in LAYER_PARAM_NAMES}
    leftovers = [
        key
        for key, val in state_dict.items()
        if key.startswith(prefix)
        and key not in handled
        and key not in per_gaussian
        and isinstance(val, torch.Tensor)
        and val.ndim >= 1
        and val.shape[0] in (old_n_tracks, old_n_poses)
    ]
    if leftovers:
        raise click.ClickException(
            f"unhandled track-sized checkpoint tensor(s) for layer {layer_id!r}: {leftovers}. "
            "Splicing synthetic tracks would desynchronize them; teach _splice_synthetic_tracks about these keys."
        )
    log.info("spliced %d synthetic track(s) into %d %s checkpoint entr(ies)", k, len(handled), layer_id)
    return handled


def _append_tracks_to_serialized_cuboid_tracks(
    ct_dict: dict, synths: list[SyntheticTrack], frame: str, *, expect_global_index: bool = False
) -> None:
    """Append synthetic tracks to one serialized ``CuboidTracks`` mapping (in place).

    ``frame`` selects the pose/dims frame ("nurec" for ``datasource_summary.json``,
    "world" for ``sequence_tracks.json``). With ``expect_global_index`` the append
    position must equal each track's recorded global index — the reload-agreement
    invariant for the member the model actually loads from.
    """
    td = ct_dict["tracks_data"]
    cd = ct_dict["cuboidtracks_data"]
    for s in synths:
        if expect_global_index and s.global_index != len(td["tracks_id"]):
            raise click.ClickException(
                f"synthetic track {s.track_id_str!r} has global index {s.global_index} but would append at "
                f"position {len(td['tracks_id'])}; the artifact's track list does not match the loaded model"
            )
        if s.track_id_str in td["tracks_id"]:
            raise click.ClickException(f"synthetic track id {s.track_id_str!r} already present in artifact tracks")
        td["tracks_id"].append(s.track_id_str)
        td["tracks_poses"].append(s.pose_tquat_nurec if frame == "nurec" else s.pose_tquat_world)
        td["tracks_timestamps_us"].append([int(t) for t in s.timestamps_us])
        td["tracks_label_class"].append(s.label_class)
        td["tracks_flags"].append("DYNAMIC")
        cd["cuboids_dims"].append(list(s.dims_nurec if frame == "nurec" else s.dims_world))


def _rewrite_track_members(zin: zipfile.ZipFile, synths: list[SyntheticTrack]) -> dict[str, bytes]:
    """New bytes for the artifact members that persist cuboid tracks.

    ``datasource_summary.json`` (NuRec frame) is what the model loads tracks from at
    inference — both ``sequence_tracks_all`` and ``sequence_tracks_dynamic`` get the
    synthetic tracks. ``sequence_tracks.json`` (world frame) feeds bbox overlays /
    viewer / eval tooling and is rewritten when present.
    """
    ordered = sorted(synths, key=lambda s: s.global_index)
    replacements: dict[str, bytes] = {}

    summary = json.loads(zin.read("datasource_summary.json"))
    all_tracks = summary.get("sequence_tracks_all") or {}
    if len(all_tracks) != 1:
        raise click.ClickException(
            f"datasource_summary.json has {len(all_tracks)} sequence(s) in sequence_tracks_all; expected exactly 1"
        )
    _append_tracks_to_serialized_cuboid_tracks(
        next(iter(all_tracks.values())), ordered, "nurec", expect_global_index=True
    )
    dynamic_tracks = summary.get("sequence_tracks_dynamic") or {}
    if len(dynamic_tracks) == 1:
        _append_tracks_to_serialized_cuboid_tracks(next(iter(dynamic_tracks.values())), ordered, "nurec")
    replacements["datasource_summary.json"] = json.dumps(summary, indent=4).encode("utf-8")

    if "sequence_tracks.json" in zin.namelist():
        chunks = json.loads(zin.read("sequence_tracks.json"))
        for chunk in chunks.values():
            _append_tracks_to_serialized_cuboid_tracks(chunk, ordered, "world")
        replacements["sequence_tracks.json"] = json.dumps(chunks).encode("utf-8")

    return replacements


def _repack_artifact(
    in_path: Path,
    out_path: Path,
    checkpoint: dict,
    manifest: EditManifest,
    synthetic_tracks: list[SyntheticTrack],
) -> None:
    """Copy the source USDZ to ``out_path``, replacing the checkpoint and edit manifest.

    Every member is copied by STREAM, never through memory: at this point the process
    already holds the materialized model on GPU and the loaded checkpoint dict on CPU, and
    a ``BytesIO`` round trip would add two more full copies of the largest member.

    Members are written through their SOURCE ``ZipInfo``, which is what carries
    ``compress_type`` across — a USDZ requires uncompressed members, so the output
    archive's own ``compression=ZIP_STORED`` is inert and the source metadata is what
    actually preserves the format. The write goes to a sibling temp file and is renamed
    only on success, so a mid-repack failure leaves no truncated artifact behind.
    (``nurec/compression/usdz.py`` solves the same shape with private helpers; reusing them
    would mean making them public, which is not this change's business.)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    def _write(tmp_path: Path) -> None:
        with (
            zipfile.ZipFile(in_path, "r") as zin,
            zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zout,
        ):
            # Synthetic tracks must also live in the artifact's track members, or the baked
            # rows' cuboid ids would index past the reloaded track lists.
            replacements: dict[str, bytes] = _rewrite_track_members(zin, synthetic_tracks) if synthetic_tracks else {}
            for item in zin.infolist():
                info = item
                if item.filename == "checkpoint.ckpt":
                    # Serialize to a sibling temp file and stream it in, so the serialized
                    # checkpoint exists once, on disk, rather than twice in memory.
                    with tempfile.NamedTemporaryFile(
                        prefix=".checkpoint.", suffix=".tmp", dir=tmp_path.parent
                    ) as ckpt_tmp:
                        torch.save(checkpoint, ckpt_tmp)
                        ckpt_tmp.flush()
                        ckpt_tmp.seek(0)
                        # force_zip64: the SOURCE ZipInfo carries the ORIGINAL (smaller)
                        # file_size, and CPython decides the zip64 header from that —
                        # a bake that grows the checkpoint past 2 GiB would otherwise
                        # RuntimeError mid-write.
                        with zout.open(info, "w", force_zip64=True) as dst:
                            shutil.copyfileobj(ckpt_tmp, dst)  # type: ignore[misc]
                elif item.filename in replacements:
                    zout.writestr(info, replacements.pop(item.filename))
                elif item.filename == EDIT_MANIFEST_FILENAME:
                    continue  # rewritten fresh below
                else:
                    with zin.open(item) as src, zout.open(info, "w") as dst:
                        shutil.copyfileobj(src, dst)  # type: ignore[misc]  # AnyStr inference fails on ZipFile IO
            if replacements:
                raise click.ClickException(
                    f"artifact is missing expected member(s) {sorted(replacements)}; cannot persist synthetic tracks"
                )
            zout.writestr(EDIT_MANIFEST_FILENAME, json.dumps(manifest.to_dict(), indent=2).encode("utf-8"))

    try:
        _write(tmp_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(out_path)


def _validate_asset_edit_target_layers(composite: GaussiansComposite, target_layers: Collection[str]) -> None:
    """Validate every bake target before the baker changes Gaussian rows."""

    rigid_layers = [k for k, v in composite.gaussians_nodes.items() if isinstance(v, RigidGaussianModel)]
    for target_layer in target_layers:
        if target_layer not in composite.gaussians_nodes:
            raise click.ClickException(
                f"Edits target Gaussian layer {target_layer!r}, which the artifact does not have "
                f"(available: {list(composite.gaussians_nodes.keys())})."
            )
        if target_layer not in rigid_layers:
            # Fail with the YAML field named, not a raw AttributeError deep in the bake.
            raise click.ClickException(
                f"Edits target layer {target_layer!r}, which has no cuboid tracks "
                f"({type(composite.gaussians_nodes[target_layer]).__name__}); asset edits "
                f"bind to cuboid-track layers only (here: {rigid_layers})."
            )
        node = cast(RigidGaussianModel, composite.gaussians_nodes[target_layer])
        try:
            node.validate_external_gaussian_topology_edit("baked asset editing")
        except NotImplementedError as exc:
            raise click.ClickException(f"Edits target unsupported layer {target_layer!r}: {exc}") from exc


@click.command(name="compile-asset-edits")
@click.option(
    "--artifact-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    required=True,
    help="Input trained USDZ artifact (typically 'last.usdz').",
)
@click.option(
    "--asset-edits-yaml",
    "asset_edits_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    required=True,
    help=(
        "Hand-authored asset_edits.yaml describing the edits (asset per track) and their "
        "controls. This is the AUTHORING format; the render-time schedule is a different "
        "format and goes to `nurec render --asset-edits-schedule`."
    ),
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    default=None,
    help="Output baked USDZ. Defaults to '<input>.edits_baked.usdz' alongside the input.",
)
def compile_asset_edits(
    artifact_path: str,
    asset_edits_path: str,
    output: Optional[str],
) -> None:
    """Bake an asset_edits.yaml into a renderable USDZ.

    Run once per authoring iteration; re-run after editing the YAML.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    in_path = Path(artifact_path)
    out_path = Path(output) if output else in_path.with_suffix(in_path.suffix + ".edits_baked.usdz")
    if out_path.resolve() == in_path.resolve():
        raise click.ClickException(
            "--output must differ from --artifact-path: the baked artifact is rewritten "
            "from the input at the end of the bake, so writing in place would destroy the "
            "trained artifact."
        )
    try:
        already_baked = EDIT_MANIFEST_FILENAME in zipfile.ZipFile(in_path).namelist()
    except zipfile.BadZipFile:
        already_baked = False  # not a zip; the artifact loader below gives the real error
    if already_baked:
        raise click.ClickException(
            f"{in_path} is already a baked artifact (it embeds {EDIT_MANIFEST_FILENAME}). "
            "Re-baking it would append a second copy of the asset geometry while the first "
            "bake's rows stay orphaned in the checkpoint. Bake from the ORIGINAL trained "
            "artifact instead; that also undoes the previous bake."
        )

    log.info("Loading asset edits from %s", asset_edits_path)
    try:
        doc = load_asset_edits(Path(asset_edits_path))
    except AssetEditsError as exc:
        raise click.ClickException(f"Invalid asset_edits.yaml: {exc}")
    log.info(
        "Parsed %d edit(s) across layer(s) %s", len(doc.edits), list(dict.fromkeys(fx.layer_id for fx in doc.edits))
    )
    # SimReady asset paths in the YAML may be relative; resolve them against the YAML's directory.
    yaml_dir = Path(asset_edits_path).resolve().parent

    log.info("Loading artifact from %s", in_path)
    artifact = Artifact(in_path)
    animations = _read_variant_animations(doc, yaml_dir)
    log.info("Constructing RenderableModel (materializes GaussiansComposite + cuboid tracks)")
    renderable = RenderableModel.load_from_artifact(artifact)
    composite = renderable._model
    if not isinstance(composite, GaussiansComposite):
        raise click.ClickException(f"Expected a GaussiansComposite model, got {type(composite).__name__}.")

    # One edit per (layer, track): the schedule registry is keyed by that pair and the
    # runtime applies a track's active aspect set to EVERY cluster on the pair, so two
    # edits binding the same track would silently clobber each other (an edit that
    # hides all its variant parts would leave a hole where the actor was). World binds
    # each get their OWN synthetic track, keyed by its string id — two world edits may
    # not claim the same track_id.
    binding_owners: dict[tuple[str, object], list[str]] = {}
    for fx in doc.edits:
        key = ("world", fx.bind.track_id) if fx.bind.is_world else (fx.layer_id, fx.bind.nurec_track_id)
        binding_owners.setdefault(key, []).append(fx.id)
    collisions = {k: v for k, v in binding_owners.items() if len(v) > 1}
    if collisions:
        raise click.ClickException(
            "multiple edits bind the same (layer, nurec_track_id) or world track_id, which the schedule "
            f"registry cannot represent: {collisions}. Merge them into one edit or rebind."
        )

    # Group edits by their resolved target layer (the default, or an explicit
    # per-edit ``layer_id``); a single document can drive several Gaussian layers at once.
    edits_by_layer: dict[str, list[EditSpec]] = {}
    for fx in doc.edits:
        edits_by_layer.setdefault(fx.layer_id, []).append(fx)
    _validate_asset_edit_target_layers(composite, edits_by_layer.keys())

    # The world<->nurec frame conversion is shared across layers; per-layer track poses are
    # derived from it inside AssetEditsBaker.
    world_to_nurec = _resolve_world_to_nurec(renderable, artifact)

    # World binds synthesize NEW tracks spanning the artifact's rig playback window
    # (same computation the composite uses for temporal appearance).
    clip_window_us: Optional[tuple[int, int]] = None
    if any(fx.bind.is_world for fx in doc.edits):
        try:
            rt = RigTrajectories.from_dict(artifact.rig_trajectories)
            clip_window_us = (
                int(rt.rig_trajectories[0].T_rig_world_timestamps_us[0].item()),
                int(rt.rig_trajectories[-1].T_rig_world_timestamps_us[-1].item()),
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise click.ClickException(f"world binds need the artifact's rig playback window: {exc}")

    # Pre-bake per-layer track/pose counts: the checkpoint splice for synthetic tracks
    # must extend tensors relative to what the CHECKPOINT holds, not the mutated model.
    pre_bake_counts = {
        layer_id: (
            len(node.cuboid_tracks.tracks_id),
            int(node.cuboid_tracks.tracks_poses.shape[0]),
        )
        for layer_id, node in composite.gaussians_nodes.items()
        if isinstance(node, RigidGaussianModel)
    }

    baker = AssetEditsBaker(composite, world_to_nurec, yaml_dir, clip_window_us=clip_window_us, animations=animations)
    try:
        # The baker raises domain errors (it must stay usable outside a CLI); this is the
        # one place that translates them into CLI failures.
        baker.bake(edits_by_layer)
    except AssetEditsBakeError as exc:
        raise click.ClickException(str(exc)) from exc
    manifest = baker.manifest()
    # Bind the manifest to this exact bake: its row indices address these post-bake layer
    # sizes AND this exact per-row track assignment, and serve/render refuse a mismatching
    # checkpoint at install (EditManifest.validate_against_layers).
    manifest.layer_row_counts = {
        layer_id: int(composite.gaussians_nodes[layer_id].get_num_gaussians()) for layer_id in manifest.layer_ids()
    }
    manifest.layer_track_digest = {
        layer_id: layer_track_digest(cast(torch.Tensor, composite.gaussians_nodes[layer_id].gaussian_cuboid_ids))
        for layer_id in manifest.layer_ids()
    }

    # Splice updated tensors for every baked layer into the original checkpoint (minimal diff).
    log.info("Loading original checkpoint to splice in updated tensors for layer(s) %s", list(edits_by_layer))
    checkpoint = torch.load(artifact.checkpoint, weights_only=False, map_location="cpu")
    # weights_only=False is the repo-wide policy for trusted NuRec checkpoints (they carry
    # OmegaConf containers; see TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD in nurec/__init__.py).
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise click.ClickException(
            f"unsupported checkpoint format in {in_path}: expected a dict with a "
            f"'state_dict' entry, got {type(checkpoint).__name__}"
        )
    original_state_dict = checkpoint["state_dict"]
    for layer_id in edits_by_layer:
        layer_node = composite.gaussians_nodes[layer_id]
        layer_state_prefix = f"model.gaussians_nodes.{layer_id}."
        spliced = 0
        for name in LAYER_PARAM_NAMES:
            state_key = layer_state_prefix + name
            if state_key not in original_state_dict:
                log.info("skipping spliced tensor %s: not present in original state_dict", state_key)
                continue
            new_tensor = getattr(layer_node, name)
            if not isinstance(new_tensor, torch.Tensor):
                new_tensor = new_tensor.data
            original_state_dict[state_key] = new_tensor.detach().cpu().contiguous()
            spliced += 1
        log.info("spliced %d %s tensors into the checkpoint", spliced, layer_id)

    # World binds: extend every track-count-sized checkpoint entry for the synthetic
    # tracks, so the artifact loads with the SAME per-layer track lists the bake used.
    synths_by_layer: dict[str, list[SyntheticTrack]] = {}
    for synth in baker.synthetic_tracks:
        synths_by_layer.setdefault(synth.layer_id, []).append(synth)
    for layer_id, synths in synths_by_layer.items():
        layer_node = composite.gaussians_nodes[layer_id]
        old_n_tracks, old_n_poses = pre_bake_counts[layer_id]
        albedo_identity = scale_identity = None
        if getattr(layer_node, "track_albedos", None) is not None:
            albedo_identity = torch.eye(4)[:3]
        if getattr(layer_node, "track_scales", None) is not None:
            scale_identity = layer_node.scale_activation_inv(torch.ones(1, 3, device=layer_node.device)).detach().cpu()
        _splice_synthetic_tracks(
            original_state_dict,
            layer_id,
            sorted(synths, key=lambda s: s.nurec_track_id),
            old_n_tracks,
            old_n_poses,
            track_albedo_identity=albedo_identity,
            track_scale_identity=scale_identity,
        )

    checkpoint["load_in_place"] = False  # shapes changed; force reassign on load

    log.info("Writing baked artifact to %s", out_path)
    _repack_artifact(in_path, out_path, checkpoint, manifest, baker.synthetic_tracks)

    log.info("Done.")
    log.info("  baked USDZ: %s", out_path)
    log.info("  clusters  : %s", ", ".join(c.cluster_id for c in baker.baked_clusters) or "(none)")
    for ctl in manifest.controls:
        log.info(
            "  control '%s' (states %s, default '%s') aspect vocabulary (SimReady name -> wire aspect):",
            ctl.control,
            "|".join(ctl.states),
            ctl.default_state,
        )
        for state, aspect in sorted(ctl.state_aspects.items()):
            log.info("    state   %-44s -> %s", state, aspect)
        for elem, aspects in sorted(ctl.element_aspects.items()):
            log.info("    element %-44s -> %s", elem, ", ".join(aspects))
    log.info(
        "Serve live over gRPC (nurec serve-grpc), or replay a schedule JSON with: "
        "nurec render --artifact-path %s --asset-edits-schedule <schedule.json>",
        out_path,
    )


__all__ = ["compile_asset_edits"]
