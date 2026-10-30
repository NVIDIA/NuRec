<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PPISP USD Export

Deep dive into the PPISP integration paths exposed by
`export-gaussian-usd-asset`. Companion document to the entry in
[`../README.md`](../README.md). The base CLI surface (USD format,
schema, percentage Gaussians, half-precision, etc.) is documented
there; this doc covers only the PPISP-related parameters and
the authored USD surface they produce.

## Contents

- [Overview](#overview)
- [Story 1 -- Color matching for USD integration](#story-1-color-matching-for-usd-integration)
- [Story 2 -- SH-optimized export (PPISP folded into SH)](#story-2-sh-optimized-export-ppisp-folded-into-sh)
- [Story 3: Controller export](#story-3-controller-export)
- [PPISP CLI reference](#ppisp-cli-reference)
- [Authored USD surface](#authored-usd-surface)
- [Traps and gotchas](#traps-and-gotchas)
- [Troubleshooting](#troubleshooting)
- [Best practices](#best-practices)

## Overview

PPISP (Physically-Plausible Image Signal Processing) is the
learned post-processing module from
[`nv-tlabs/ppisp`](https://github.com/nv-tlabs/ppisp): per-frame
exposure, per-camera vignetting, per-frame color correction, and
per-camera CRF, plus a controller CNN that predicts exposure and
color from the rendered radiance image at inference time.

A Gaussian USD asset can carry these effects two different ways
depending on the target viewer's capabilities:

- **`spg-runtime`** _(default)_: PPISP is authored as Omniverse
  Sensor Processing Graph (SPG) shader prims on every
  RenderProduct. The viewer applies vignetting, ZCA color, exposure
  and CRF at render time. Highest fidelity. Requires an
  SPG-capable viewer (Omniverse Kit / RTX). **Sensor-specific.**
- **`sh-optimized`**: PPISP is folded into the Gaussian spherical
  harmonic coefficients before export. Any USD viewer can render
  the corrected output without runtime PPISP support. Lower
  fidelity (PPISP becomes a per-asset baked approximation tied to
  one reference camera/frame). **Single-sensor static look.**

A third path -- the per-camera PPISP **controller** -- is used by
default in `spg-runtime` export when the loaded PPISP module was
trained with controllers; see [Story 3](#story-3-controller-export).

The seven CLI flags that drive PPISP integration are:

```
--ppisp-integration-mode {spg-runtime,sh-optimized}
--scene-radiance-scale FLOAT
--ppisp-responsivity FLOAT
--ppisp-reference-camera-id INT
--ppisp-reference-frame-id INT
--enable-ppisp-controller-export / --disable-ppisp-controller-export
--sh-optimization-num-iterations INT
```

All other flags (`--config-name`, `--output-dir`, schema, format,
percentage, half-precision, ...) are described in
[`../README.md`](../README.md) and are unchanged by this work.

The exporter's `--help` is the canonical CLI surface.

```bash
bazel run //:run -- export-gaussian-usd-asset --help
```

## Story 1 - Color matching for USD integration

A Gaussian asset is trained against an arbitrary, scene-internal
radiometric scale. The USD scene it gets composited into has its
_own_ radiometric scale (whatever physical units the target
renderer expects -- nits, normalized lumens, an arbitrary
linear-space convention, etc.). Dropping the trained asset in
unmodified leaves the Gaussian radiance authored at the wrong
scale relative to the rest of the scene's lights and surfaces, so
the asset reads either too bright or too dark, and downstream
exposure / tonemap stages misbehave.

`--scene-radiance-scale` is the **radiometric matching knob**:
multiply Gaussian SH by `k` to bring the asset into the target
renderer's radiance scale. `--ppisp-responsivity` is the
**post-ISP compensator**: when the Gaussian radiance is rescaled
by `k`, the HDR buffer arriving at PPISP is also rescaled by `k`,
so PPISP's exposure / vignetting / CRF stages see a different
input and the LDR look drifts. Setting `inputs:responsivity = 1/k`
puts the value PPISP sees back to its training-time magnitude, so
the post-ISP look is preserved.

Pipeline:

```
Gaussian SH coefficients
        |  multiplied by --scene-radiance-scale = k (asset-level, baked)
        v
   per-view render -> HDR buffer at k * (training radiance)
        |
        |  multiplied by inputs:responsivity (= --ppisp-responsivity, default 1.0)
        v
   PPISP shader (spg-runtime only): exposure -> vignette -> ZCA color -> CRF
        v
   LDR
```

Two intent-driven recipes:

- **Match the target renderer's units, keep the original look**:
  set `--scene-radiance-scale k` to whatever rescaling the target
  scene needs, and set `--ppisp-responsivity 1/k`. PPISP sees the
  same input it saw at training time, the LDR look is preserved.
- **Match the target renderer's units, also retune the look**:
  set `--scene-radiance-scale k` for the radiometric matching,
  then set `--ppisp-responsivity` to whatever post-ISP brightness
  you want (independent of `k`). The two flags are independent --
  the `1/k` is just the "no look change" choice.

Both flags can also be used in isolation:

- Only `--scene-radiance-scale` (responsivity left at `1.0`):
  rescale Gaussian radiance and _let_ PPISP react to it. The look
  changes; useful when the original look was wrong and the
  rescale should drive the new exposure.
- Only `--ppisp-responsivity` (scale left at `1.0`): retune the
  post-ISP look without touching the asset's radiance. Useful for
  per-shot grading at consume time -- the attribute is
  overridable in USD, so consumers can edit it without
  re-exporting.

### `--scene-radiance-scale FLOAT` (default `1.0`)

Multiplies the SH coefficients of every Gaussian _before_ they
are written to the USD asset. Permanent (baked into the asset);
retuning requires re-exporting.

Strictly positive finite floats only; `0.0`, negatives, `inf`,
`nan` and non-numeric values are rejected by
`validate_export_integration_args`. Layered after SH-match in
`sh-optimized` mode so the bake happens on the unscaled SH and
the scale multiplies the bake (see
[Traps and gotchas](#sh-match-runs-_before_-radiance-scaling)).

### `--ppisp-responsivity FLOAT` (default `1.0`)

Authors the same float on every PPISP shader's
`inputs:responsivity` USD attribute (image PPISP shader plus, in
controller mode, the `PPISPControllerPool` shader). The SPG shader
multiplies the HDR buffer by this value _before_ any PPISP stage;
the controller pool applies the same scale to its HDR input before
feature extraction so its exposure / color predictions are computed
on the same signal the image shader processes. The attribute is
overridable in USD: a downstream consumer can edit it per shot
without re-exporting the asset.

Same fail-loud validation rules as `--scene-radiance-scale`.

> **Trap**: `--ppisp-responsivity` is consumed only by the SPG
> shader, which only runs in `spg-runtime` mode. In
> `sh-optimized` mode there is no SPG shader to read the
> attribute, so the value is currently neither baked nor
> rejected -- it is silently a no-op. To preserve the post-ISP
> look in `sh-optimized` mode, the responsivity compensation
> would have to be folded into the SH bake itself; that is not
> implemented today. Treat `--ppisp-responsivity` as
> SPG-runtime-specific. See
> [Traps and gotchas](#--ppisp-responsivity-is-silently-ignored-in-sh-optimized).

## Story 2 - SH-optimized export (PPISP folded into SH)

When the target viewer cannot run PPISP at runtime (no SPG /
no RTX, or the asset must work in any USD-aware tool), pick
`--ppisp-integration-mode sh-optimized`. The exporter then bakes
PPISP into the Gaussian SH coefficients so the rendered image
already looks PPISP-corrected without a shader.

The bake is a two-phase optimization:

1. **DC warm-start** (`sh_match/dc_warmstart.py`). Each Gaussian's
   diffuse RGB is passed through PPISP at the resolved
   `(camera_id, frame_id)`, then written
   back into the SH DC band of `features_albedo`. This gives the
   refinement loop a much better starting point than the trained
   coefficients (which live in pre-PPISP space). `features_specular`
   is intentionally not warm-started: rotating the higher-order
   bands by the per-Gaussian PPISP Jacobian introduces
   ill-conditioned outliers at PPISP saturation that hurt the
   subsequent fit.
2. **Multi-view Adam refinement loop**
   (`optimize_sh_to_match_ppisp` in `sh_match/optimization.py`).
   Each iteration draws a single training view uniformly at random
   (with replacement) from the training dataloader, renders the
   _current_ model and the _reference_ model under that view,
   computes the L1 loss between PPISP(reference) and current, and
   takes one Adam step on the SH coefficients (and on the per-
   Gaussian density, which absorbs spatial frequencies SH cannot).

The reference model is a deepcopy of the live system's model taken
once at loop entry, so its SH does not drift while the live model
is updated. Per-iteration re-rendering is intentional -- caching
all reference views for large datasets blows memory; we trade
compute for bound memory, mirroring the Lavoisier choice.

The reference render disables vignetting, matching the warm-start
which also bakes PPISP without vignetting. This keeps the warm-
start and the refinement target color-space-consistent and avoids
baking a per-pixel vignetting falloff into per-Gaussian SH (which
SH cannot represent properly anyway).

### Reference camera/frame in `sh-optimized`

`--ppisp-reference-camera-id` and `--ppisp-reference-frame-id`
pick _which_ PPISP configuration gets folded:

- Both unset: defaults to camera `0`, frame `0`.
- One set, the other unset: see
  `resolve_ppisp_reference_ids` -- the resolver fills the missing
  axis with `0` and emits a debug log so the run is reproducible.
- Both set: the bake targets exactly that `(camera_id, frame_id)`
  pair. The pair must be in range; out-of-range values are
  rejected with `<name> must be in [0, N-1], got <value>`.

> Once baked, the asset has a single PPISP look. To compare or
> swap looks, run the export twice with different reference IDs
> and ship the two assets separately.

### `--sh-optimization-num-iterations INT` (default `None` ⇒ 3000)

Replaces `SHMatchingOptimizationConfig.num_iterations`. Strictly
positive int, `0` / negatives / floats / bools rejected. Only
meaningful in `sh-optimized` mode -- setting it in any other mode
is rejected loudly so a CLI typo does not silently become a
no-op.

The default 3000 brings parity with the
`threedgrut/export/usd/post_processing_sh_bake.py` baseline
(~7 epochs \* a few hundred training views) on representative
scenes. The previous default of 700 produced visible color/detail
regressions on valiant-scale datasets. Reduce if export time
matters more than bake quality; raise if the bake still looks
under-converged. The Adam learning rates and the optimization
device are _not_ CLI-tunable -- they are algorithm-tuned and only
exposed via the keyword-only `sh_optimization_config` parameter
to the Python entry point (e.g. tests forcing `device="cpu"`).

## Story 3: Controller export

The PPISP controller is a small CNN that predicts per-frame PPISP
exposure / color latents from the per-view feature buffer. During
training it runs every iteration; it is the path that yields the
most realistic novel-view PPISP behavior. In `spg-runtime` mode,
USD export uses the controller by default when the loaded PPISP
module was trained with controllers. The exporter generates
per-camera CUDA sidecars with weights embedded in device constants,
so target viewers must be able to compile and resolve those
generated sidecars. No separate controller-weight sidecar or
runtime `inputs:weights` binding is authored.

### Toggle

```bash
bazel run //:run -- export-gaussian-usd-asset \
    --config-name configs/your_config.yaml \
    -- model.ckpt=/path/to/checkpoint.ckpt
```

When the checkpoint contains controller-trained PPISP, the command
above authors:

- a per-camera `PPISPControllerPool` shader prim wired to the
  pre-PPISP HDR RenderVar (it scales that HDR by
  `inputs:responsivity` before feature extraction, matching the
  image PPISP shader),
- a per-camera `PPISPController` MLP shader prim wired to the
  intermediate controller-feature RenderVar,
- a per-camera _automatic-parameter_ PPISP shader (variant of the
  static SPG shader, reads exposure / color from the controller
  output instead of from USD attributes).

Controller exports package generated per-camera
`ppisp_controller_<camera>.cu` / `.cu.lua` sidecars plus the
`ppisp_usd_spg_auto.*` sidecar set. The static `ppisp_usd_spg.*`
sidecar set is not packaged in controller mode; see
[Authored USD surface](#authored-usd-surface).

Use `--disable-ppisp-controller-export` to force the static SPG
fallback path for a controller-trained checkpoint. Use
`--enable-ppisp-controller-export` to fail loudly if the loaded
checkpoint does not contain trained controllers.

### Incompatibility with `sh-optimized`

`--enable-ppisp-controller-export` and
`--ppisp-integration-mode sh-optimized` are mutually exclusive
and the validator rejects the combination with:

```
enable_ppisp_controller_export is incompatible with
ppisp_integration_mode='sh-optimized': SH-match folds PPISP into
Gaussian SH coefficients, leaving no per-frame work for the runtime
controller. Use spg-runtime mode for controller export.
```

Locked in by `test_controller_export_with_sh_optimized_rejected`.

### Production fallback table

When the controller is not present, or when
`--disable-ppisp-controller-export` forces it off, the static
`spg-runtime` path picks one of three authoring strategies based on
the reference IDs:

| `--ppisp-reference-camera-id` | `--ppisp-reference-frame-id` | Authored                                                                                                                |
| ----------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| unset                         | unset                        | Time-sampled per-camera (one USD time sample per training frame)                                                        |
| set                           | unset                        | Static, single-camera, neutral exposure (`0.0`) and identity color latents                                              |
| set                           | set                          | Static, single-frame: exposure / color latents pulled from `ppisp.exposure_params[frame_id]` / `color_params[frame_id]` |
| unset                         | set                          | Rejected upstream by `resolve_ppisp_reference_ids` -- frame-only fixing has no camera axis to anchor vignetting/CRF to  |

The fourth row's rejection message is:

```
ppisp_reference_frame_id was set without ppisp_reference_camera_id
in spg-runtime export mode. Frame-only fixing is ambiguous because
vignetting and CRF live on the camera axis.
```

## PPISP CLI reference

Each entry below documents one of the seven new flags. Mode-of-
effect is one of `spg-runtime` (consumed by SPG shaders),
`sh-optimized` (consumed by the SH-match loop), or `both`.

### `--ppisp-integration-mode {spg-runtime,sh-optimized}`

- **Type**: choice • **Default**: `spg-runtime`
- **Mode-of-effect**: meta-flag (selects the path)
- **Effect**: picks the integration story
  ([1](#story-1-color-matching-for-usd-integration) for color
  matching applies in either, [2](#story-2-sh-optimized-export-ppisp-folded-into-sh)
  selects the bake, default selects runtime SPG).
- **Rejected values**: anything else. Click rejects unknown
  choices before our code runs.

### `--scene-radiance-scale FLOAT`

- **Type**: float, strictly positive, finite
- **Default**: `1.0` (no-op)
- **Mode-of-effect**: both
- **Effect**: multiplies every Gaussian's SH coefficients before
  export. Permanent. Layered after SH-match.
- **Rejection**: `0.0`, negatives, `inf`, `nan`, non-numeric
  → `ValueError: scene_radiance_scale must be ...` /
  `TypeError: scene_radiance_scale must be a real number, ...`.
  Locked in by
  `TestValidateExportIntegrationArgs.test_non_positive_radiance_scale_rejected`
  and
  `TestValidateExportIntegrationArgs.test_non_numeric_scale_rejected`
  in `export_integration_test.py`.

### `--ppisp-responsivity FLOAT`

- **Type**: float, strictly positive, finite
- **Default**: `1.0` (no-op)
- **Mode-of-effect**: `spg-runtime` only (silent no-op in
  `sh-optimized`; see
  [Traps and gotchas](#traps-and-gotchas))
- **Effect**: authored verbatim on every PPISP shader's
  `inputs:responsivity` USD attribute. Multiplies the HDR buffer
  before PPISP. To preserve the post-ISP look while applying a
  radiance scale `k`, set this to `1/k`. Overridable downstream
  without re-export, so consumer-side per-shot grading is
  possible without rebuilding the asset.
- **Rejection**: same as `--scene-radiance-scale`.

### `--ppisp-reference-camera-id INT`

- **Type**: int • **Default**: unset
- **Mode-of-effect**: both, with different semantics:
  - `spg-runtime`: pins the static export path to that camera.
    Setting the camera without a frame yields neutral-exposure /
    identity-color authoring (vignetting + CRF still come from
    the camera).
  - `sh-optimized`: selects the camera whose vignetting / CRF /
    exposure / color get folded into SH. Default `0` if unset.
- **Rejection**: out of range `[0, num_cameras-1]` →
  `ppisp_reference_camera_id must be in [0, N-1], got V`.

### `--ppisp-reference-frame-id INT`

- **Type**: int • **Default**: unset
- **Mode-of-effect**: both, with the same dual semantics as
  `--ppisp-reference-camera-id` (see
  [Production fallback table](#production-fallback-table)).
- **Rejection**: out of range, or set without a camera ID in
  `spg-runtime` mode (see fallback table fourth row).

### `--enable-ppisp-controller-export` / `--disable-ppisp-controller-export`

- **Type**: optional boolean pair • **Default**: auto
- **Mode-of-effect**: `spg-runtime` only
- **Effect**: when omitted, exports the controller path whenever
  the loaded PPISP module was trained with controllers. `--enable`
  forces the controller path and fails if trained controllers are
  absent. `--disable` forces the static SPG path.
- **Rejection**: combined with
  `--ppisp-integration-mode sh-optimized` (see
  [Story 3](#story-3-controller-export)).

### `--sh-optimization-num-iterations INT`

- **Type**: int, strictly positive • **Default**: unset
  (resolver falls back to `SHMatchingOptimizationConfig.num_iterations`,
  i.e. `3000`)
- **Mode-of-effect**: `sh-optimized` only
- **Effect**: replaces `SHMatchingOptimizationConfig.num_iterations`
  via `dataclasses.replace`. Does not mutate any caller-supplied
  config.
- **Rejection**: `0`, negatives, floats, bools, set in non
  `sh-optimized` mode. Locked in by
  `TestResolveShOptimizationConfig.test_num_iterations_override_*`.

## Authored USD surface

This section enumerates every prim and attribute the PPISP export
authors. Items split into **standard USD** (works on any USD
runtime) and **Omniverse / RTX-specific** (only meaningful on
SPG-capable viewers). On non-OV runtimes the latter are
inert custom data; consumers can ignore them safely.

### Standard USD

- `UsdGeomPoints` or `UsdVol ParticleField3DGaussianSplat`
  primitive carrying the Gaussian payload. Schema selected by
  `--usd-schema`. Documented in `README.md`.
- `UsdLux DomeLight` for the trained sky environment, when one
  exists. Standard USD light schema.
- The original rig `Camera` prims, untouched.

### Omniverse / RTX-specific

The following are _not_ part of Pixar/standard USD; they are
specific conventions consumed by Omniverse's RTX renderer and
SPG runtime.

- **SPG sidecar files** packaged into the USDZ archive next to
  the USD layers:
  - Static path:
    `ppisp_usd_spg.cu`, `ppisp_usd_spg.cu.lua`,
    `ppisp_usd_spg.usda`.
  - Controller path:
    the static set is replaced by
    generated per-camera `ppisp_controller_<camera>.cu` and
    `ppisp_controller_<camera>.cu.lua` files with controller weights
    embedded in CUDA device constants, plus
    `ppisp_usd_spg_auto.cu`, `ppisp_usd_spg_auto.cu.lua`, and
    `ppisp_usd_spg_auto.usda`.
    Selection is done by `select_spg_files_for_export`. The static
    and controller file sets share no filenames (locked in by
    `TestSelectSpgFilesForExport.test_no_overlap_between_static_and_controller_sidecars`).
    No standalone controller-weight asset is packaged, and no
    `inputs:weights` binding is authored on the controller shader
    prims; weights are embedded into the generated CUDA source.
- **PPISP Shader prims** authored on every RenderProduct:
  - Reads from the `HdrColor` RenderVar
    (`omni:rtx:aov` connection).
  - Writes to the `PPISPColor` RenderVar.
  - The viewer's display RenderVar `LdrColor` is rewired to
    `PPISPColor`.
  - Inputs (`inputs:responsivity`, per-camera vignetting / CRF,
    per-frame exposure / color latents) are authored as USD
    _connections_ to the `<cam>_ppisp.ppisp:*` attributes rather
    than literal values, so the shader pulls parameters from
    whichever camera the RenderProduct points at. (Direct
    shader-writer callers without a camera path fall back to
    literal `inputs:*` values.)
- **Gaussian shader inputs**
  (`inputs:apply_srgb_linear=False`, `inputs:apply_inverse_tonemap=False`).
  Both default to `False` because PPISP runs the linearization /
  tonemap pipeline itself; double-applying would clip highlights
  and crush blacks.
- **`<cam>_ppisp` camera**: a hidden sibling of each rig camera
  that inherits its intrinsics, pins the `exposure:*` namespace to
  neutral values, and carries every PPISP parameter as a
  `ppisp:*` attribute (the source of truth the shader inputs
  connect to). The RenderProduct's `camera` relationship is
  rewired to it; the user-facing rig camera is untouched.
  Exposure conventions:

  - `exposure = 0.0`
  - `exposure:fStop = 1.0`
  - `exposure:iso = 100.0`
  - `exposure:responsivity = 1.0`
  - `exposure:time = 1.0`
  - `visibility = invisible`, `hidden = True`

  `ppisp:*` attributes authored on it: `responsivity`,
  `vignetting*`, `crf*`, `exposureOffset`, `colorLatent*`,
  and (controller path) `controllerWeights` (a
  documentation mirror; the runtime reads weights from the
  embedded CUDA sidecar). See
  [Traps and gotchas](#why-the-camera-and-the-exposure-overrides)
  for why the camera exists.

  > **Tiling**: the shaders support tiled RenderProduct atlases
  > (per-tile vignetting UV and per-tile controller params) through
  > the `inputs:tileCountX/Y` _constant shader inputs_ (not camera
  > attributes -- tiling is a RenderProduct property). The exporter
  > authors the untiled `1x1` default; a consumer assembling a
  > multi-camera tiled product overrides these shader inputs to
  > match the RenderProduct's tile grid (`viewTilesPerDimension`).

- **`PPISPControllerPool` / `PPISPController` shader prims**
  (controller path only): per-camera CNN/pool and MLP stages,
  wired to the auto-parameter PPISP shader's exposure / color
  inputs through the `ControllerParams` AOV. Both also carry
  `inputs:tileCountX/Y` constant inputs, sizing their per-tile
  feature / params buffers and launch grids for tiled atlases.
  `PPISPControllerPool` additionally carries `inputs:responsivity`
  (mirrors `--ppisp-responsivity`): the CNN scales the HDR input by
  it before feature extraction, so the controller predicts exposure
  / color from the same responsivity-scaled radiance the image
  PPISP shader processes (no-op at the default `1.0`).

## Traps and gotchas

Concrete failure modes we hit during development. Each one names
the mechanism and points to the locking test or code line.

### Why the camera and the `exposure:*` overrides

Without intervention, Omniverse Kit applies its own
exposure / shutter / ISO model to the camera bound to a
RenderProduct _before_ the SPG PPISP shader runs. PPISP itself
applies an exposure stage (per-camera, per-frame). The two
exposure stages compose multiplicatively, so the asset looks
roughly two stops too dark in default Kit settings, with the
exact error depending on the rig camera's authored
`exposure:*` values.

The fix authored by `_create_ppisp_camera` in
[`usd_writer.py`](usd_writer.py) authors a hidden `<cam>_ppisp`
sibling of the rig camera (inheriting its intrinsics via a
path-based `inherits` arc), neutralizes the entire `exposure:*`
namespace on it, then rewires the RenderProduct's `camera`
relationship to it. The rig camera prim itself is untouched, so
the user-facing camera tree and any other consumer of the same
camera are unaffected. This `<cam>_ppisp` camera also carries the
PPISP parameters as `ppisp:*` attributes that the shader inputs
connect to.

This `exposure:*` neutralization is parity with the Lavoisier
"camera ISP disable" mechanism for the `exposure:*` axis.
**Verification of broader parity (other ISP-axis attributes)** is
tracked as Commit 13 in the planning doc. If a target viewer
applies additional ISP stages outside `exposure:*`, the
neutralization may need extending.

### `--ppisp-responsivity` is silently ignored in `sh-optimized`

The flag is wired to the PPISP shader's `inputs:responsivity`
attribute; in `sh-optimized` mode no PPISP shader is authored, so
nothing reads the flag. Validation does _not_ currently reject a
non-default value in `sh-optimized` mode.

Concretely, this means the "set `responsivity = 1/k` to preserve
the post-ISP look after a radiance rescale" recipe from
[Story 1](#story-1-color-matching-for-usd-integration) does not
apply to `sh-optimized` exports: the SH bake captures PPISP at a
fixed input scale, and rescaling the SH afterwards via
`--scene-radiance-scale` will shift the look. To preserve the
look in `sh-optimized` mode the responsivity compensation would
have to be folded into the SH bake itself, which is not
implemented today.

Workaround for now: in `sh-optimized` exports, treat
`--scene-radiance-scale` as a "match-and-accept-the-look-shift"
knob and leave `--ppisp-responsivity` at `1.0`.

### `--ppisp-reference-frame-id` requires a camera id in `spg-runtime`

PPISP's per-frame parameters (exposure, color) are scoped _under_
a camera; vignetting and CRF are pure camera attributes. Fixing
just the frame would leave the camera axis time-sampled or
defaulted, which would make the resulting static-frame look
non-deterministic across viewers. The validator rejects this
combination loudly with the message quoted in the
[fallback table](#production-fallback-table).

### Controller export and `--ppisp-integration-mode sh-optimized` are mutually exclusive

Folding PPISP into SH leaves nothing for the controller to do at
runtime, so exporting both would ship dead weights. The validator
rejects explicit `--enable-ppisp-controller-export` with
`--ppisp-integration-mode sh-optimized`; see
[Story 3](#story-3-controller-export) for the exact message.

### SH-match runs _before_ radiance scaling

Order matters: scaling is multiplicative on SH, so applying it
before SH-match would cause the optimization to undo it (the
reference is rendered from a deepcopy taken _before_ the scale,
so the loss would push the live SH back to the unscaled values).
The orchestrator runs SH-match first, then `apply_radiance_scale_to_sh`.

### Only `num_iterations` is CLI-tunable on the SH-match config

The Adam learning rates (albedo `2.5e-3`, specular = albedo / 20,
density `5e-2`) and the device (`cuda`) are tuned for the
algorithm; surfacing them as CLI flags would invite mis-tuning
without much upside. Python callers that need to override them
(e.g. tests forcing `device="cpu"`) build a
`SHMatchingOptimizationConfig` and pass it via the keyword-only
`sh_optimization_config` parameter to `export_gaussian_usd_asset`.
The CLI override `--sh-optimization-num-iterations` layers on top
of the explicit config via `dataclasses.replace`, never mutating
the caller's instance.

### Controller export and embedded weights

The controller's CNN/MLP weights are authored directly into
generated per-camera CUDA sources. This avoids a per-frame binary
weight-buffer upload from Lua and avoids packaging any external
weight sidecar, but downstream USD viewers still need CUDA SPG
support that can resolve and compile the generated
`ppisp_controller_<camera>.cu` files. Use
`--disable-ppisp-controller-export` when the target viewer should
receive the static SPG shader instead.

## Troubleshooting

Errors are designed to be self-explanatory; this table maps the
ones we author directly to their fix.

| Error message                                                                                                                    | Cause                                                                                                          | Fix                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `ppisp_integration_mode must be one of ('spg-runtime', 'sh-optimized'), got ...`                                                 | Typo in `--ppisp-integration-mode`.                                                                            | Use `spg-runtime` or `sh-optimized`.                                        |
| `scene_radiance_scale must be strictly positive, got ...`                                                                        | Passed `0.0`, a negative, or a non-numeric.                                                                    | Use a strictly positive float.                                              |
| `scene_radiance_scale must be finite, got ...`                                                                                   | Passed `inf` or `nan`.                                                                                         | Use a finite float.                                                         |
| `ppisp_responsivity must be ...`                                                                                                 | As above, on the responsivity flag.                                                                            | Same.                                                                       |
| `ppisp_integration_mode='sh-optimized' requires sh_optimization_config (...)`                                                    | Programmatic caller selected `sh-optimized` without supplying a config and without going through the resolver. | Reach the orchestrator via the CLI or via `resolve_sh_optimization_config`. |
| `enable_ppisp_controller_export is incompatible with ppisp_integration_mode='sh-optimized': ...`                                 | Explicit controller export was requested with `sh-optimized`.                                                  | Drop `--enable-ppisp-controller-export`, or switch mode.                    |
| `sh_optimization_num_iterations is only meaningful when ppisp_integration_mode='sh-optimized'; got ppisp_integration_mode='...'` | `--sh-optimization-num-iterations` set in the wrong mode.                                                      | Drop the flag, or switch mode.                                              |
| `sh_optimization_num_iterations must be strictly positive, got ...`                                                              | `0`, negative, or bool.                                                                                        | Use a positive int.                                                         |
| `sh_optimization_num_iterations must be an int, got float`                                                                       | Passed e.g. `100.0`.                                                                                           | Pass an int.                                                                |
| `ppisp_reference_camera_id must be in [0, N-1], got V`                                                                           | Out-of-range camera index.                                                                                     | Use a valid `[0, num_cameras - 1]` value.                                   |
| `ppisp_reference_frame_id must be in [0, N-1], got V`                                                                            | Out-of-range frame index.                                                                                      | Use a valid `[0, num_frames - 1]` value.                                    |
| `ppisp_reference_frame_id was set without ppisp_reference_camera_id in spg-runtime export mode. ...`                             | Frame-only fixing in `spg-runtime`.                                                                            | Set `--ppisp-reference-camera-id` too, or drop the frame ID.                |

PPISP controller export: if PPISP looks wrong on novel views, the
most likely cause is a generated CUDA controller sidecar not being
resolved or compiled by the target viewer. Fall back to the static
path with `--disable-ppisp-controller-export`.

## Best practices

- Production target on SPG-capable viewers (Omniverse / RTX):
  default mode (`spg-runtime`). Controller-trained checkpoints use
  the controller path by default; non-controller checkpoints use the
  static SPG path. Do not set reference IDs unless you need a
  single-frame static look on the static path.
- Production target on non-SPG viewers: `sh-optimized` mode with
  reference IDs picking the look you want baked.
- Asset radiance does not match the target renderer's
  radiometric scale: set `--scene-radiance-scale k` to bring the
  Gaussians into the target's units, then set
  `--ppisp-responsivity 1/k` if you want the post-ISP look
  unchanged. Drop the `1/k` if the rescale should also drive a
  look change.
- Per-shot grading expected: ship the asset with whatever
  `--ppisp-responsivity` value matches the radiance scale, and
  let consumers override `inputs:responsivity` in USD per shot
  (the attribute is overridable, no re-export needed).
- Controller path: validate on the target viewer first (look for
  SPG/CUDA sidecar warnings in viewer logs). There is no external
  controller-weight sidecar to deploy, but the target viewer must
  still compile the generated CUDA sidecars.
- SH-match feels under-converged: bump
  `--sh-optimization-num-iterations` to e.g. `6000`. Going much
  higher rarely helps because the LRs are tuned for the 3000-step
  reference setup.
