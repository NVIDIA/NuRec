<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Asset edits: controllable SimReady assets on NuRec playback

**Module paths:** `nurec/utils/edit/`, `nurec/asset_edits/runtime.py`,
`nurec/grpc/serve.py`, `nurec/utils/io/export/render.py`

NuRec playback is normally static: whatever the capture recorded is what renders. Asset
edits make parts of the scene controllable. An edit replaces a captured actor with a
SimReady asset whose switchable states are baked into the scene — or introduces a NEW
asset at a fixed world pose (a `world` bind; see below) — and at render time an
external sender picks which state is visible. The runtime matches aspects by string
equality and applies the sender's instantaneous `gain`/`cold_rgb` values to cluster
opacities and colors. It evaluates no animation curves and attaches no semantic meaning
to the aspect names or the optional metadata carried by the manifest.

Four terms, used throughout:

- An **edit** is one entry in `asset_edits.yaml`: "use this SimReady asset in place of
  whatever the capture has on this cuboid track", or "introduce this asset as a new
  actor at this world pose".
- A **cluster** is a group of gaussians baked from one switchable state of the asset.
- An **aspect** is the opaque string that names a cluster, e.g. `"panelstate:on"` or
  `"appearancemode:sequence"`. NuRec never interprets these — they are just keys.
- The **manifest** is a JSON file embedded in the baked USDZ that maps each aspect to
  its cluster's gaussian row indices.

## Aspect naming: from SimReady to wire

### SimReady USD basics

A SimReady asset is a tree of **prims** — named nodes that carry geometry (for
our purposes, gaussian splats). A prim can declare a **variantSet**: a named
switch with a fixed list of mutually exclusive **variants**; selecting a variant
decides which prims beneath it exist and how they look. An asset can author a
variantSet per configurable behavior, with per-prim visibility over time inside
each variant.

### Example: a prop with alternate panels

Consider a configurable prop with a **body** and a `panel` variantSet with two
variants, `off` and `on`. Both variants contain a **frame** and a **primary panel**.
The `on` variant also contains a **secondary panel**. In `off`, the primary panel is
always visible; in `on`, visibility samples alternate the two panels.

The bake takes the prop apart into groups of gaussians (**clusters**) and gives
each switchable group a name (its **aspect**). At render time the sender sends a
list of names each frame: whatever is named is visible, every other cluster is
hidden. Only animated prims get a name of their own; whatever else a variant
contains is grouped into one **catch-all cluster** per variant.

| Piece           | What the bake does with it                                                | Can you switch it?            |
| --------------- | ------------------------------------------------------------------------- | ----------------------------- |
| body            | baked in once, always visible — it isn't under the variantSet             | no (it's just scene geometry) |
| primary panel   | its own cluster: the animation shows/hides it                             | yes, individually             |
| secondary panel | its own cluster: the animation shows/hides it                             | yes, individually             |
| frame           | not animated, so no name of its own: grouped into the variant's catch-all | only by switching the variant |

Aspects are composed from names the asset author already knows (from the
SimReady USD structure):

- the **control** is the variantSet name: `panel`;
- each variant gets one **state aspect**: the name of its catch-all cluster,
  holding everything in that variant that is not an animated element. Ours are
  `panel:off` and `panel:on`; the frame lives in each;
- each animated prim (an **element**) gets its own aspect, in one of two forms:
  - `panel:shared:Panels/Primary` — the primary panel is **identical in every
    variant**, so it baked once and can be shown in any state;
  - `panel:on:Panels/Secondary` — the secondary panel **exists only in the `on`
    variant**, so its name carries that state.

Why a catch-all per state rather than one per control (just `panel`)? Geometry
that truly doesn't depend on the state belongs outside the variantSet, where it
bakes once as static geometry with no aspect at all — the body already shows
this. Whatever the author placed inside a variant is, by that placement,
state-dependent, so its home is that state's cluster. The per-state aspect also
doubles as the state **marker**: every frame names exactly one state per
control, which keeps request logs self-describing even when a catch-all is
empty.

A frame's aspect list assembles the look explicitly — naming is literal, nothing is
implied:

- panel off: `["panel:off", "panel:shared:Panels/Primary"]`
- panel on, primary visible: `["panel:on", "panel:shared:Panels/Primary"]`
- panel on, secondary visible: `["panel:on", "panel:on:Panels/Secondary"]`

The middle frame uses the same primary-panel cluster in a different state, so only
the catch-all switched. The lists are exhaustive — sending `panel:off` alone would
show the frame without either panel.

Because each panel is its own aspect, a sender can show any combination without
rebaking: replay the authored sequence, hold one panel visible, or show both at
once. Geometry in a catch-all cluster changes only when the state switches.

### Boundary cases

The element/catch-all split adapts to the asset, and both extremes are valid.

If every prim inside a variant is an animated element — remove the frame from
the example — nothing is left over, and the catch-alls bake with zero gaussians.
They still exist and are still named every frame: every variant the USD declares
gets its aspect whether or not any geometry landed in it. In the frame lists
above, "panel off" and "panel on, primary visible" differ only by the state aspect.
Every piece of such an asset's variant geometry is individually controllable.

If a variant contains no animated prims at all — say a snowplow whose
variantSet just holds the blade raised in one variant and lowered in the other —
everything bakes into its catch-all and there are no element aspects. Switching
state is the only control such an asset offers.

### Reading the vocabulary

None of the naming has to be derived by hand: every control table in the manifest
carries the finished dictionary (`state_aspects`, `element_aspects`), the bake log
prints the same table, and both the install gate and the offline render CLI reject
an unknown aspect by listing the full valid vocabulary. Read it straight off a
baked artifact:

```
unzip -p baked.usdz asset_edits_manifest.json | jq '.controls[] | {control, state_aspects, element_aspects}'
```

## The idea: geometry in, addressing out

The bake does two things. It physically appends the asset's gaussians to the scene
(every switchable state at once — non-default states hidden, the default state and
static dressing visible), and it writes down where they landed:

```
aspect "appearancemode:sequence" -> (layer "dynamic_rigids", track 7) -> rows [284101..284660]
aspect "panelstate:on"           -> (same layer, same track)          -> rows [...]
```

At render time, showing a state means writing opacity into exactly those rows. Nothing
else in the scene changes. Because every appended row is tagged with the track's cuboid
id, the clusters move with the track through the existing pose pipeline — no new motion
machinery.

## The design rule: the renderer is dumb

NuRec never learns what the asset is or what its states mean. The sender decides
animation phases, state transitions, and fades. It tells the renderer, per frame,
the full set of aspects that should be visible right now.
The renderer shows those clusters and hides the rest of that track's clusters.

On the wire (`DynamicObject.aspect_state`):

- The aspect list is the complete visible set for that actor **for this request's
  frame**. The renderer is stateless: it keeps nothing between requests, so the
  sender sends the full set every frame.
- An empty list means "every baked cluster of this actor HIDDEN this frame"
  (zero opacity — the clusters simply do not rasterize; nothing renders black).
  That includes the off state's geometry, so selecting `off` requires the off-state
  aspects, not an empty list. Nor does it restore the captured actor:
  `replace_original: true` — the bake option that hides the actor an edit
  replaces (see The pipeline) — is baked into the checkpoint, out of the aspect
  set's reach. A missing `aspect_state` field
  means "no statement — render this actor's scheduled/baked state this frame". The two
  cases are distinguishable because `aspect_state` is a message field, which always has
  explicit presence in proto3 (the `optional` keyword documents the intent; it is not
  what provides the semantics).
- `AspectGain` optionally fades a cluster (`gain` scales the baked opacities; `cold_rgb`
  is its reference color at gain 0), so the sender can do continuous fades. Both are
  validated at ingestion: `gain` must be finite in [0, 1], `cold_rgb` three channels in
  [0, 1] — a request carrying anything else is rejected.
- `aspect_state` rides `DynamicObject`, so a driven actor is also a _listed_ actor:
  every listed actor must carry `pose_pair`. A pose-less entry is rejected loudly —
  the editing-actors path re-poses every listed actor, so accepting one would silently
  teleport the actor to the origin.

Why this split: the sender already owns scenario time and the animation programs. If
the renderer also animated, there would be two clocks to keep in sync. This way an
animation bug is always a sender bug, and camera and lidar at the same timestamp always
agree, because the visible state is a pure function of the request and the frame time.
Statelessness is also what makes server-side pooling safe: one scene may be served by
several pooled or replicated backends, and since no request depends on an earlier one,
it does not matter which backend renders which frame. This includes multi-process
serving (`serve-grpc --multiproc-workers N`, which round-robins requests across worker
processes sharing one read-only copy of the model weights): each worker installs the
manifest itself at scene load, and the per-frame writes land only in that render's
collected buffers (see Render-time switching below), never in the shared weights — so
any worker can serve any frame, including concurrently with the others.

### Target-layer compatibility

Both baked and in-place asset editing change the target layer's Gaussian topology.
The offline `compile-asset-edits` path and the live insertion and track-PLY replacement
paths therefore ask the target model to validate that operation before changing any
rows.

Interpolated-table deformable layers reject these operations. Their deformation table
stores a row-aligned temporal trajectory for every Gaussian, so adding or replacing
Gaussian rows without making the corresponding table edit would break that alignment.
Choose a supported rigid layer, or reconstruct with the legacy deformable model, when
the scene must support baked asset edits, live insertion, or PLY replacement.

## The pipeline

1. **Find the track.** `nurec render --list-tracks <layer_id>` writes the layer's cuboid
   tracks (id, class, position, distance) to `layer_tracks.json` in the output directory,
   logs them, and then **exits without rendering** — it is a query, not a render mode.
   `--label-tracks <layer_id>` does render, with an id label drawn on every track of that
   layer, on every product named by `--output-type`. Pick the track, write the edit.
2. **Bake** (`nurec compile-asset-edits`). The composed asset is placed onto the track
   with one shared transform: uniformly scaled so its length matches the cuboid
   (`fit_length`) and rested on the cuboid floor. `replace_original: true` hides the
   captured actor's gaussians (a rebake from the original artifact undoes this). Each
   `controllable` variantSet state bakes as its own cluster — the default state
   visible, every other state hidden; elements identical in every state are baked
   once. The manifest is embedded in the output USDZ.
   Its row indices are only valid for that exact bake, which is why it travels inside
   the artifact. A baked USDZ is refused as bake input (re-baking would orphan the
   first bake's rows); iterate by re-baking from the original trained artifact.
3. **Serve.** `nurec serve-grpc` detects the manifest at scene load and installs the edit
   runtime. Live `AspectState` messages switch aspects per frame; they ride
   `DynamicObject`, the scene-editing channel, so the server must run with
   `--enable-editing-actors` (the "senders may mutate the scene" switch). If a
   schedule JSON sits next to the USDZ (`<stem>.schedule.json`), it is loaded too —
   see Replay below.
   Install fails LOUD: a baked artifact whose runtime cannot install (corrupt manifest,
   layer-size mismatch against the checkpoint, a model without override support) refuses to
   serve. (One narrow degrade: a manifest track that passes validation but cannot be
   resolved against the composite's track LIST is dropped with a warning rather than
   failing the scene — its clusters stay hidden and unaddressable.) Refusal matters
   because — with `replace_original` — the captured actor is hidden in the bake, so
   serving without the edits would silently render a hole where it was. For the same reason the
   `edit_assets` RPC (asset replacement/insertion) is refused on edit-baked scenes: it
   reorders layer rows and would invalidate the manifest's baked indices. Clients can
   detect the wire via the `asset_edits.aspect_state = stateless-v1` key in
   `get_server_config` (advertised only when the server runs with
   `--enable-editing-actors`, since the field rides `DynamicObject`).
4. **Render-time switching** (`AspectOverrideRuntime`). Once per frame it writes the
   active clusters' baked per-gaussian opacities into the collected buffers and zeroes
   the inactive ones. When a gRPC request row-filters a layer (editing-actors), the
   baked row indices are remapped through that request's actual layout, so filtering
   never corrupts unrelated gaussians.

## Replay without a sender

Aspect timing can also come from data instead of a live sender: a schedule JSON next to
the USDZ lists timed aspect switches ("at t=5606420401000 µs, show `panelstate:on`").
The server auto-loads it at scene start, and
`nurec render --asset-edits-schedule <schedule.json>` replays it offline. `nurec render`
REFUSES an edit-baked artifact without a schedule (the controllable clusters would
render undriven) — except in the `--list-tracks`/`--label-tracks` discovery modes,
which exist precisely to find the ids a schedule needs. This is just a
file — swapping it re-times the animation with no rebake and no extra process. Live
messages take precedence over it.

Two flags, two different formats, so they carry different names:
`nurec compile-asset-edits --asset-edits-yaml` takes the authoring YAML; `nurec render
--asset-edits-schedule` takes this render-time JSON.

`nurec render --asset-manifest <manifest.json>` overrides where the GEOMETRY manifest is
read from; by default it comes from inside the baked USDZ, which is where it belongs.
It only takes effect together with `--asset-edits-schedule`, and the manifest is checked
against the loaded checkpoint before anything renders (row counts, a per-layer track-id
digest, and each cluster's rows actually belonging to the track it claims) — pairing a
manifest with the wrong artifact is refused rather than rendered.

## Assets are the USD-native surface

Switchable behavior is authored in the SimReady asset itself, as standard USD:
variantSets for the states, `visibility.timeSamples` for visibility sequences, plus
two custom attributes (`nurec:gain` curves, `nurec:coldRgb`). The same animation plays in
vanilla usdview. The bake extracts these into `ControlAnimation` tables inside the
manifest so the sender knows how to animate each control without opening the asset. NuRec
itself never evaluates them.

Authored STATIC visibility is respected too: a splat that composes `invisible` with no
visibility timeSamples anywhere on its chain is the current state's hidden dressing
(e.g. an alternate panel hidden in the selected variant) and is skipped by the
importer. A controls-free bake therefore shows exactly the authored default state,
not every variant's payload at once. (The importer reads the raw `visibility`
attribute up the ancestor chain rather than `ComputeVisibility()`, which silently
degrades when the ParticleField schema is not registered.)

An edit can target another supported Gaussian layer with an explicit `layer_id`.

### Static props may be a `.ply`

A prop with nothing to switch does not need USD. An edit whose `asset` ends in `.ply`
is read as one 3D Gaussian Splatting cloud — the same file format, reader and
conventions as an AssetBank track or the `edit_assets` RPC (Y-up in the file, rotated
to Z-up on import; pre-activation attributes; spherical harmonics copied as stored) —
and bakes exactly like a controls-free SimReady asset: one static part, no clusters,
no manifest controls, a normal `assets[]` provenance record. It takes no `controls`
and no `prim_path` (the schema rejects both). Animated or controllable assets remain
SimReady USD only.

A `.ply` carries no units. On a track bind the default `fit_length` fits its bbox to
the cuboid, so the unit-normalised PLYs the asset bank ships need nothing extra; on a
world bind pass the prop's real-world `dims` and `scale: fit_length` (or a numeric
`scale`), otherwise a normalised file bakes about a metre long. Higher-order SH is
zero-padded or truncated to the scene's SH width at bake time, like any asset.

## World-pose binds: introducing new props

A `world` bind has no source actor. The bake synthesizes a brand-new cuboid track — a
constant pose over the whole clip window — and the asset rides it exactly like a
replacement rides a captured track:

```yaml
edits:
  - id: barrier_1
    bind: { type: world, position: [150.0, -3.5, 0.0], yaw_deg: 90.0 }
    asset: props/barrier_waterfilled_01/Body.usd
    # no controls: a bare static prop; add controls later to animate it
  - id: cone_1
    bind: { type: world, position: [152.0, -3.5, 0.0], dims: [0.4, 0.4, 0.7] }
    asset: props/traffic_cone/9000014.ply # a static prop may be a 3DGS .ply
    placement: { scale: fit_length } # bank PLYs are unit-normalised: fit to dims
```

- `position` is world metres (the same frame `--list-tracks` prints); `anchor: base`
  (default) rests the placed asset's base there, `center` centers the cuboid there.
- The cuboid dims default to the placed asset's bbox, so `align: bbox` puts the asset
  exactly at the authored point; `dims` overrides. `scale` defaults to fixed `1.0`
  (props keep their authored real-world size — `fit_length` has nothing to fit against
  and is rejected unless `dims` is explicit).
- The synthetic track gets the string id `prop:<edit_id>` (override with `track_id`), a
  `label_class` (default `protruding_object`) that must route it to the edit's target
  layer and no other — the bake asserts this — and 2 keyframes spanning the artifact's
  rig playback window.
- `replace_original` is rejected: there is nothing to replace.

The bake persists the track fully: it is appended to `datasource_summary.json` (NuRec
frame) and `sequence_tracks.json` (world frame) inside the output USDZ, and every
track-count-sized checkpoint tensor (pose-calibration deltas, per-track time-embedding
state, the layer's `n_tracks` extra state) is extended positionally. The artifact
therefore stays self-describing — serve, offline render, `--list-tracks`, bbox overlays
and downstream consumers all see the prop as an ordinary track with zero special-casing.
The index-agreement invariant (the baked rows' cuboid id equals the track's layer-local
index after reload) holds because layer partitions preserve global insertion order and
synthetic tracks always append at the end; the bake asserts it anyway and fails loud.

Consequences worth knowing:

- `get_dynamic_objects` lists the prop, so an editing-actors sender that re-sends every
  listed actor keeps it visible with no client changes — and conversely, a request that
  omits it row-filters the prop away for that frame, like any other actor.
- Making the prop controllable later is just re-baking with a `controls:` block —
  everything downstream already addresses it by `(layer_id, nurec_track_id)` and its
  string track id, exactly like a replacement edit.
- Props carry no semantic signal (class 0 in semantic outputs) and zeroed lidar
  intensity, same as every baked cluster.

## Relation to in-place editing (`edit_assets`)

The gRPC server has a second, older editing mechanism: the `edit_assets` RPC inserts new
actors or replaces existing ones **in place**, by mutating the loaded model's parameters
(the asset is an AssetBank track or a server-local `.ply`), with
`restore_model_parameters` as the whole-scene undo. The two look adjacent — both put new
assets on tracks — but they solve different problems with opposite mechanics:

|                    | Baked asset edits (this doc)                     | `edit_assets`                                   |
| ------------------ | ------------------------------------------------ | ----------------------------------------------- |
| Asset              | SimReady USD (or a static `.ply`), baked offline | one static splat cloud (AssetBank id or `.ply`) |
| Applied            | at bake time; the artifact ships the result      | live, per RPC, on the loaded model              |
| Server state       | none — per-request `aspect_state`, nothing kept  | edits persist until `restore_model_parameters`  |
| Animation          | per-frame aspect switching                       | none                                            |
| Model weights      | untouched (per-frame collected-buffer overrides) | mutated in place                                |
| Multi-process pool | works (statelessness — see The design rule)      | refused (in-place weight mutation cannot share) |

They are mutually exclusive **per scene**: `edit_assets` insert/replace reorders and
regrows layer rows, which would silently invalidate the baked manifest's row indices, so
it is refused on an edit-baked artifact (bake the desired assets instead, or serve a
plain artifact for `edit_assets`). Every scene without a baked manifest is untouched by
this design — `edit_assets` behaves there exactly as before. Composing the two on one
scene — live inserts on top of a baked artifact — would require the runtime's row
addressing to be remapped live as edits mutate the layers; that is a known follow-up,
not part of this design.

## Limitations

- **Lidar intensity is approximate.** Baked clusters occlude correctly (they change
  opacity, which lidar also sees), but their lidar intensity channel is zeroed.
- **No USD-driven appearance channel.** NuRec's "dynamics via timesample" is pose
  animation for the external Kit runtime; at render time NuRec reads the checkpoint and
  JSON tracks, not USD. Aspect timing comes only from the wire or the schedule file.
