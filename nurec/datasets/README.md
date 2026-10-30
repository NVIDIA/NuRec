<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Model-Free Unstructured LiDAR

## Overview

NuRec can render a LiDAR frame in two ways. The **structured** path rasterizes the
sensor's `(n_rows, n_columns)` range-view grid, which requires a fitted
`RowOffsetStructuredSpinningLidarModel` calibration. The **unstructured** path renders
the measured `(N, 1)` return list directly from per-ray directions and timestamps, so it
needs no sensor model at all.

The unstructured path is config-gated and **off by default**. The camera path is
untouched by everything on this page.

## Why it exists

The structured model assumes a spinning sensor whose beams land on a regular
row/column lattice. That assumption fails for:

- **non-spinning sensors** (MEMS, solid-state) that have no row/column lattice;
- **raw point clouds / radar**, where intrinsics were never recorded;
- **any sensor whose calibration is unavailable or wrong** — a mis-fitted grid
  silently distorts every ray it places.

Because the unstructured path consumes the measured ray geometry, it is correct by
construction in all of those cases.

## The two axes

Ray _delivery_ and ray _geometry_ are configured independently.

| axis                          | values                         | meaning                                                                                                                 |
| ----------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `dataset.lidar_ray_mode`      | `structured` \| `unstructured` | shape of the ray list handed to the renderer: the `(n_rows, n_columns)` grid, or the `(N, 1)` return list               |
| `dataset.use_real_lidar_rays` | `false` \| `true`              | where directions come from: generated at grid elements from the sensor model, or the true measured point-cloud geometry |

The combinations:

| ray_mode       | use_real_lidar_rays | result                                                                                                                                                         |
| -------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `structured`   | `false`             | classic path (default)                                                                                                                                         |
| `structured`   | `true`              | classic grid, measured directions                                                                                                                              |
| `unstructured` | `false`             | unstructured renderer, modeled rays. **Needs a sensor model.** Useful as an A/B control: both renderers see identical rays, so any difference is the renderer. |
| `unstructured` | `true`              | **genuinely model-free.** The production configuration.                                                                                                        |

`dataset.lidar_ray_mode=unstructured` must be paired with
`model.renderer.lidar_unstructured=True`; `NuRecConfig` rejects the mismatch at config
time rather than at the first render.

> **All** `lidar_ray_mode`, `use_real_lidar_rays` and `lidar_unstructured_*` fields are
> `PerSensor`: pass a scalar to apply it to every active LiDAR, or a
> `{lidar_id: value}` dict to differ per sensor. A dict must list **every** active
> `lidar_id` — there is no fallback key.

## How the render works

`GSplatRenderer._render_unstructured`:

1. Builds an `UnstructuredLidarFrameContext` from the measured rays and their per-ray
   relative timestamps. The context (tiling) is pose-independent, so it is built once
   per frame **in the dataloader worker**, shipped on the batch as plain tensors
   (`RenderingData.unstructured_lidar_tiling`), and rebuilt in the renderer at zero
   compute.
2. Calls the public `gsplat.rasterization(..., lidar_frame_context=...)`.
3. Picks the sampling mode automatically: `adaptive` when `multi_visit_fraction > 0.2`,
   else `endpoints`.
4. Composites the learned ray-drop head exactly as the structured path does. The
   geometric `1 - opacity` is only a fallback for when no head is configured.

Unlike the structured rasterization, which references its projection to the mid-frame
pose, the unstructured renderer interpolates **each ray's** rolling-shutter pose between
the frame-start and frame-end sensor poses, anchored to the same window as the per-ray
timestamps.

### Renderer knobs

All `PerSensor`, all under `model.renderer.culling`:

| knob                                 | default | notes                                                                                                       |
| ------------------------------------ | ------- | ----------------------------------------------------------------------------------------------------------- |
| `box_method_lidar_unstructured`      | `ut`    | `ut` = cheap 7-sigma-point unscented box; `pbf` = exact closed-form silhouette; `hybrid` = PBF near, UT far |
| `cull_dead_pairs_lidar_unstructured` | `true`  | exact pre-cull; **required** at scale (see below)                                                           |
| `pbf_near_range_lidar_unstructured`  | `8.0`   | metres; `hybrid` only                                                                                       |
| `rs_single_cover_lidar_unstructured` | `false` | requires `box_method=hybrid`                                                                                |

`cull_dead_pairs` is not an optimization you can casually disable. The unstructured
model is omnidirectional — a Gaussian gets a candidate pair for every angular tile it
overlaps — while the rays are directional and sparse, so under densification the vast
majority of `(gaussian, tile)` pairs are dead. Without the cull the pair count reaches
~1B and overflows the rasterizer's int32 batch-state buffer, crashing training.

## Model-free ray-drop

The unstructured path renders only _returning_ rays, so on its own it has no ray-drop
negatives to learn from. They are supplied as **virtual rays** in unoccupied field of
view, selected by `dataset.lidar_unstructured_raydrop_virtual_rays`:

| mode                | requires                    | how negatives are produced                                                                  |
| ------------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| `off`               | —                           | none (ray-drop loss must be 0)                                                              |
| `model_elements`    | `use_real_lidar_rays=False` | exact complement of the returned grid elements — the true sensor drop mask                  |
| `aggregate_lattice` | `use_real_lidar_rays=True`  | **model-free.** Samples unoccupied FOV tiles and times them from a `FiringLatticeEstimator` |

Both reuse `RayFlags.DROPPED` and render real and virtual rays in a single pass.

### The firing lattice

`aggregate_lattice` is what makes ray-drop work without a sensor model.
`nurec/utils/lidar_firing_lattice.py` reconstructs, over multiple frames, an aggregate
lattice mapping `(azimuth, elevation) -> relative frame time`: **where** the sensor
fires (its support) and **when** in the sweep it fires there. It is built once in the
main process before the DataLoader forks.

`lidar_unstructured_raydrop_lattice_time_modes` controls the number of distinct visit
times kept per cell: `1` (default) suits a spinning sensor where each direction is
visited once per frame; `>1` supports self-crossing MEMS patterns.

### Comparing ray-drop between the two paths

**Only `model_elements` gives a like-for-like ray-drop A/B.** The two paths do not score
the same ray population:

- **structured** scores the full `(n_rows, n_columns)` grid, so _every_ non-returning
  beam is a ray-drop positive;
- **unstructured** scores measured returns plus _sampled_ virtual rays. Under
  `aggregate_lattice` most true positives are simply absent from the denominator.

So under `aggregate_lattice` the unstructured arm **under-recalls by construction** —
high precision, low recall, low IoU — even when the ray-drop head is equally good. The
signature is unmistakable: precision and accuracy move the _opposite_ way to IoU, and
`IoU == recall` to three decimals, because the score is entirely false-negative
dominated.

Geometry metrics (`depth_*`, `chamfer`, `intensity_*`) mask by `did_return_gt` and are
comparable under either mode.

## Field of view

`dataset.lidar_unstructured_fov_vert_deg` / `_horiz_deg` take `[lo_deg, hi_deg]` and
**override** the model FOV when set. `fov_vert_deg` is **required** for a sensor with no
model — the render errors without it. A horizontal span of ~360 degrees marks the sensor
periodic (spinning); anything narrower is treated as limited-FOV.

## Interaction with active tile masking

`model.renderer.active_tile_mask` has two strategies, and only one is compatible:

- **`camera_invalid_only`** — safe. Masks camera tiles only; LiDAR rendering and
  supervision are untouched.
- **`valid_camera_lidar_overlap`** — **rejected** by `NuRecConfig` when any sensor is
  unstructured. Its LiDAR half derives the mask via
  `build_camera_to_lidar_tile_mapping` -> `compute_lidar_element_to_tile`, which needs a
  structured spinning-LiDAR model and reshapes to `(n_rows, n_columns)`. The model-free
  path has neither, and `render_unstructured` accepts no mask, so the combination could
  only crash or silently drop the mask. `_render_unstructured` carries a render-time
  backstop for non-config producers.

## Configuration

Minimal enablement on top of an existing LiDAR preset:

```yaml
dataset:
  lidar_ray_mode: unstructured
  use_real_lidar_rays: true
  lidar_unstructured_raydrop_virtual_rays: aggregate_lattice

model:
  renderer:
    lidar_unstructured: true

loss:
  raydrop:
    lambda_: 1.0
```

## Config-time guards

`NuRecConfig._validate_lidar_ray_mode_consistency` fails fast, per sensor, on:

- `lidar_ray_mode=unstructured` without `renderer.lidar_unstructured=True` (and the
  reverse);
- `lidar_ray_mode=unstructured` on a non-GSplat renderer;
- `raydrop` loss enabled with `virtual_rays=off` and stored no-return supervision disabled (no negatives to learn from);
- `aggregate_lattice` without `use_real_lidar_rays=True`;
- `model_elements` with `use_real_lidar_rays=True`;
- lattice bin counts set partially (both or neither);
- FOV not ordered `[lo, hi]`;
- `system.test.lidar.floating_point_filter` with `neighbor_source="rangeview"` (defined on
  the range-view grid; use `neighbor_source="angle"` for unstructured lidar);
- `active_tile_mask.strategy=valid_camera_lidar_overlap`;
- `datamodule.train_batch_size != 1` (per-frame tilings are variable-size and cannot be
  collated).

## Known limitations

- **Ray-drop class balance.** Returns are subsampled to `n_train_sample_lidar_rays`
  (default 2048), but virtual rays are appended _after_ that subsample, so the full
  per-frame drop set enters every training batch — roughly a 30-80:1 drop:return ratio
  in the ray-drop loss, versus the natural grid ratio the structured path trains on. A
  cap plus a paired A/B is follow-up work.
- **Model-less rigs.** `LidarFreePoseViewGeometry.from_rig_trajectories` returns an
  empty geometry if _any_ LiDAR calibration lacks a model, so a genuinely model-less rig
  does not reach the render yet. This predates the unstructured path and is unchanged by
  it.
- **Batch size is pinned to 1** for unstructured runs (see the guard above).

## See also

- `nurec/config/dataset.py` — field-level documentation for every option above.
- `nurec/utils/lidar_firing_lattice.py` — the firing-lattice estimator.
- `nurec/models/gaussians/renderers.py` — `GSplatRenderer._render_unstructured`.
