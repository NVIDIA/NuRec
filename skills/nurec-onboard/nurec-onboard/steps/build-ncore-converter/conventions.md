<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore V4 Conventions Reference

Converter-facing operational patterns for NCore V4 + NuRec. For axis/origin conventions,
transformation notation, pose-graph semantics, and camera/LiDAR intrinsic parameter schemas,
consult the NCore spec directly:

- Coordinate frames, `T_a_b` notation, pose-graph static/dynamic edges:
  [NCore data conventions](https://nvidia.github.io/ncore/data/conventions.html)
- Component store layout, dtypes, and per-component record fields:
  [NCore data formats](https://nvidia.github.io/ncore/data/formats.html)
- Camera/LiDAR intrinsic parameter schemas (FTheta, OpenCV pinhole/fisheye,
  row-offset-spinning): [NCore sensor models](https://nvidia.github.io/ncore/data/sensor_models.html)

This file documents only what NCore does not spell out, or where NuRec imposes tighter
constraints than NCore alone.

## Contents

- [Coordinate frames](#coordinate-frames)
- [Sensor selection scope](#sensor-selection-scope)
- [Sequence ID constraints](#sequence-id-constraints)
- [Transformations](#transformations)
- [Timestamps](#timestamps)
- [LiDAR data](#lidar-data)
- [Cameras](#cameras)
- [Masks](#masks)
- [Cuboids (annotations)](#cuboids-annotations)
- [Component group profiles](#component-group-profiles)
- [Store types](#store-types)
- [Data quality requirements for NuRec](#data-quality-requirements-for-nurec)
- [NCore convention checklist](#ncore-convention-checklist)

## Sensor selection scope

`FileBasedDataConverter` owns `self._active_camera_ids`,
`self._active_lidar_ids`, and `self._active_radar_ids` for
`--camera-id` / `--lidar-id` / `--radar-id` scope filters. Do not assign those
attributes in dataset converters; filter the dataset's discovered sensor lists
with `get_active_camera_ids()`, `get_active_lidar_ids()`, and
`get_active_radar_ids()` so operator-provided scope filters stay active.

## Sequence ID constraints

The sequence ID names the output directory, the data shard, the sequence meta
JSON, and the auxiliary store prefix. Keep it to letters, digits, `_`, and `-`.

A `.` is the one character that silently breaks reconstruction. NuRec locates
auxiliary stores by truncating the **data shard** filename at its first `.` and
keeping every neighboring file whose name starts with that prefix plus `.aux.`.
A dotted sequence ID truncates mid-name, so nothing matches and the
reconstruction runs with no auxiliary data at all — conversion, validation, and
aux generation each report success. This bites only when `dataset.aux_data` is
enabled, which is the default, and NuRec Onboard always generates auxiliary
data.

`/` and whitespace do not break the prefix match; they are rejected because the
ID becomes a path component and a store filename.

Converter `get_sequence_ids()` implementations can route source names through
`ncore_example.utils.sanitize_sequence_id()`, which replaces dots with `-` and
rejects path separators and whitespace. That mapping is not injective, so check
the sanitized IDs of a sequence set for collisions before writing — a collision
overwrites an earlier conversion's output directory and auxiliary stores.
Preserve the source identifier separately when exact round-tripping matters.

## Coordinate frames

### Camera sensor frame

NCore's camera convention is OpenCV (+X right, +Y down, +Z optical, see NCore spec). Every
converter stores `T_camera_rig` in this frame; datasets whose native camera frame differs
must apply a 3x3 basis change before storing.

**Basis-change pattern.** Make the decision explicit in every converter (and record it in
`camera_frame_convention` in `critical_parameter_justifications`):

- **Native is already OpenCV/NCore.** No basis change. Document the fact inline and cite
  the dataset docs.
- **Native differs** (e.g. a dataset whose native camera frame is +X forward, +Y left,
  +Z up). Apply an explicit 3x3 basis change and document the axis mapping.

The basis change composes as `T_camera_rig_ncore = T_camera_rig_native @ basis_change`
where each column of the 3x3 rotation block of `basis_change` is the native-frame
coordinates of the corresponding NCore axis. Never implicitly default to identity; see
`ncore_example/converter.py` in the materialized converter workdir for a scaffolded
axis-by-axis comment block (or `ncore_<dataset>/converter.py` after the scaffold rename).

### LiDAR sensor frame

Directions are stored in the **spinning model frame** (next section), not the sensor's
native frame. `T_lidar_rig` maps the model frame to the rig frame.

### Spinning LiDAR model frame

The `(row, col) -> direction` formula lives in the NCore sensor-models spec (row-offset
spinning model). Converter-facing consequences:

- Model azimuth 0° is +X, 90° is +Y, +Z up -- independent of the physical sensor's native
  azimuth zero. The converter decides which physical column becomes model +X by the way it
  populates `column_azimuths_rad`.
- Native sensor conventions vary by manufacturer and even between a vendor's "lidar frame"
  vs. "sensor frame"; do not assume a dataset's raw column 0 is forward.
- **Do not** use `RowOffsetStructuredSpinningLidarModel.elements_to_sensor_rays` to derive
  the directions you store -- that API is for validation. Use direct trigonometry in
  `_write_spinning_lidar_frame` (see "LiDAR direction derivation" below).

Patterns to watch for when consulting a sensor's datasheet:

- The native sensor frame is frequently not `+X forward`. Many vendors use `+Y forward`
  (`X` sideways), and some define a "lidar frame" (e.g. toward the cable connector) that
  is rotated 180° from the "sensor frame" used to report data.
- ROS / driver stacks sometimes apply an implicit rotation between the raw sensor frame
  and the frame that reaches the dataset; verify from the driver source if the dataset
  was captured through one.
- Confirm from the datasheet which direction is azimuth 0° and whether reported column
  indices increase clockwise or counter-clockwise when viewed from above (rig +Z looking
  down).

### Canonical azimuth-correction formula

Store `T_lidar_rig` verbatim from calibration (full SE(3), no rotation decomposition) and
bake the yaw offset into `column_azimuths_rad`:

```
az_correction = atan2(T_lidar_rig[1, 0], T_lidar_rig[0, 0])
column_azimuths_rad[col] = sensor_native_azimuth[col] - az_correction
```

**Anti-pattern.** Do **not** carry the yaw in both the extrinsic AND the per-column
azimuths. Splitting the yaw (e.g. computing `T_lidar_rig` as a rotation-only hack after
separately deriving a heading offset) double-applies the correction and rotates the entire
point cloud by `az_correction` relative to the rig.

After conversion, verify that model azimuth 0° (direction `[1, 0, 0]`) maps through
`T_lidar_rig` to the expected rig-frame direction.

### T_sensor_rig rotation direction verification

Per NCore's own API (`T_sensor_rig` docstring in `ncore/impl/data/compat.py`): `T_sensor_rig` transforms points **from the sensor coordinate frame to the rig coordinate frame** (`v_rig = R @ v_sensor + t`). When a calibration file expresses sensor poses as "rig-relative" transforms — or when you derive the rotation from a basis change — it is easy to apply the matrix in the wrong direction, silently producing a mirrored or rotated point cloud that passes all format checks.

**Required spot-check before running validation**: from the dataset documentation and sensor datasheet, identify **two non-collinear** sensor axes whose rig-frame directions are unambiguous (e.g. the sensor's up axis must map to rig +Z, and the sensor's forward axis must map to rig +X). Verify each mapping directly:

```python
R = T_sensor_rig[:3, :3]
# Verify TWO non-collinear axes whose rig-frame direction is unambiguous from
# the dataset. One axis is insufficient: a 90° rotation about the verified axis
# preserves it but rotates everything orthogonal to it, producing a silently
# wrong T_sensor_rig that passes a single-axis check.
#
# Replace each (v_sensor, v_rig_expected) pair with the axis mapping you can
# verify from the dataset (e.g. for an OpenCV camera with NCore-canonical rig:
#   up:      v_sensor_up      = [0, -1, 0]   v_rig_up      = [0, 0, 1]
#   forward: v_sensor_forward = [0,  0, 1]   v_rig_forward = [1, 0, 0]
# ).
for v_sensor, v_rig_expected, axis_name in (
    (v_sensor_up,      v_rig_up,      "up"),
    (v_sensor_forward, v_rig_forward, "forward"),
):
    v_rig = R @ v_sensor
    assert np.allclose(v_rig, v_rig_expected, atol=1e-3), (
        f"T_sensor_rig {axis_name}-axis wrong: got {v_rig}, expected {v_rig_expected}"
    )
```

If an assertion fails on the first axis only, the rotation is inverted — take the transpose: `R_correct = R.T` (valid only for orthonormal rotation matrices). If both assertions fail in different ways, the matrix is composed against the wrong rig frame; trace back which intermediate basis change was applied incorrectly (see "Track the rig frame at every step" below).

### Static extrinsics over empirical solving

When a dataset ships a static extrinsic for a sensor (calibration file, sensor metadata, factory-calibration JSON, etc.), use it. Empirically deriving `T_sensor_rig` from per-image poses — e.g. by averaging `inv(T_rig_world(t)) @ T_sensor_world(t)` across timestamps — is a **smell**, not a canonical recipe. It introduces per-frame pose noise, creates frame-mixing risk (the empirical solve carries the noise of whatever stream defines the rig trajectory), and bypasses the calibration team's own verification of the static value.

Empirical solving is acceptable in only two narrow cases:

- The dataset demonstrably ships **no** static extrinsics for that sensor.
- Static extrinsics exist but are **demonstrably wrong** by independent verification — not just "disagrees with my mental model" or "disagrees with one other stream by an unexplained amount". When the disagreement is rotation-only, the agent must additionally verify that the static extrinsic's translation block is _also_ wrong before going fully empirical: compare static translations against per-image-pose-derived translations, or read the input YAML's commentary on translation accuracy. Going empirical for rotation forces going empirical for translation too; if the static translation block is independently accurate, the SVD-mean replaces it with per-frame pose noise. Fix the rotation in isolation rather than rebuilding the full `T_sensor_rig` empirically.

If the static extrinsic and per-image poses disagree, the disagreement itself is the bug to investigate before working around. Common root causes:

- **Different world-frame origins between sensor pose streams**. A dataset may ship `lidar/poses.json` referenced to the lidar mount and `camera/<id>/poses.json` referenced to ground level (or vice versa); the constant offset between those origins shows up as a constant disagreement when you compose with the static extrinsic.
- **Per-sensor quaternion conventions**. Scalar-first `(w, x, y, z)` vs scalar-last `(x, y, z, w)`; left-handed vs right-handed; passive vs active. Each silently produces a different rotation matrix from the same four numbers.
- **Static-extrinsic file expressing rotation in a non-standard sensor frame**. Some datasets express the rotation in a sensor frame that is the OpenCV camera frame rotated by some fixed amount (e.g. 180° about an axis), so applying the static rotation to OpenCV-convention sensor data yields a wrong but plausible-looking result. The fix is to compose the documented basis change, not to retreat to empirical solving. (Hint: if applying a uniform basis change works for some sensors in the calibration block but not others, suspect a per-sensor residual rotation layered on top of the uniform one — diagnose by comparing the static rotation against an independent per-sensor reference, not by abandoning the static source.)
- **Direction-of-transform ambiguity**. The static block may be `T_rig_sensor` (rig→sensor) rather than `T_sensor_rig` (sensor→rig). Inverting the wrong direction produces a translation that disagrees with per-image poses by an offset proportional to the rig-vs-sensor pose. Cross-check via the canonical NCore convention (`T_sensor_rig` docstring in `ncore/impl/data/compat.py`).

Resolving the disagreement at its source preserves calibration accuracy across all sensors and avoids the noise-stacking that empirical SVD averaging introduces.

When `data.profile` records calibration direction conventions or known
transform mismatches, treat those entries as primary evidence for interpreting
raw calibration fields. NCore's V4 pose writer stores source-to-target
transforms under the `(source_frame_id, target_frame_id)` key, so a raw field
name is never enough by itself to decide whether a matrix can be passed to
`store_static_pose` unchanged or must be inverted first.

#### When the input YAML names an authoritative source

When the input YAML's `data.description` or `data.docs` calls out a static-extrinsic source as authoritative for an extrinsic, treat that statement as the operator's **starting hypothesis**, not a directive. Input YAMLs describe dataset facts; they are not code prescriptions. The converter falsifies the hypothesis against the data rather than promoting it to a guarantee.

When the dataset ships per-image (or per-frame) pose data for the sensor, that pose data is an **independent cross-check** — it lives in whatever frame the converter actually uses for the rig trajectory, so it answers the question "where is this sensor in the rig the converter writes" directly. Verify the static source against it before trusting:

- **Rotation**: load the static rotation block (with any documented basis change composed), verify against per-image rotations within the documented tolerance.
- **Translation**: compute the sensor mount as `inv(T_rig_world(t)) @ T_sensor_world(t).t` at a representative frame and compare against the static translation block. They agree within calibration tolerance (often ≤10 cm) when the static block and the rig trajectory share a frame; they disagree when the static block lives in a different frame from the rig trajectory the converter actually uses (a common, non-pathological situation — calibration files routinely express positions relative to a calibration-internal anchor that differs from the trajectory's world anchor). When they disagree, the per-image-derived position wins because it is in the rig the converter built.

Rotation agreement does **not** imply translation agreement: the static rotation block can be in the right frame while the static translation block is not. Cross-check both independently.

When the dataset does **not** ship per-image pose data for that sensor, no independent cross-check is available; trust the static source and record that the verification was unverifiable.

The mechanical companion is the `authoritative_source.verification` sub-block on the `T_lidar_rig` / `T_camera_rig` justifications, plus the conditionally-mandatory translation block on `T_camera_rig`'s worked-example (with a `position_in_rig_expected_derivation` field that records where the prediction came from, so a reviewer can confirm independence). See [critical-parameters.md](critical-parameters.md) for the schema.

### Track the rig frame at every step

When a converter chains two basis changes — for example, an `R_anchor` that defines the rig orientation against one sensor (e.g. front-camera-anchored NCore-canonical rig) and an `R_native_to_canonical` that converts another sensor's native frame (e.g. a sensor whose native frame defines forward as +Y while the rig defines forward as +X) — explicitly assert which rig orientation each intermediate variable lives in **before** composing the next basis change. Without this, the same matrix can get applied twice (or applied to a frame that's already canonical), silently producing a 90°-rotated `T_sensor_rig` that passes all format checks.

Recommended pattern: for every line that constructs or stores a `T_*_rig`, add a one-line comment naming the rig orientation the result is in (e.g. `# rig = NCore-canonical (+X forward, +Y left, +Z up)`), and verify with the two-axis spot-check above before storing.

Example failure mode (a 90°-azimuth bug from chained basis changes):

```python
# rig = NCore-canonical (+X forward, +Y left, +Z up), anchored at front_camera
T_camera_anchored_rig = solve_camera_rig_empirically(camera_world_poses, ...)

# WRONG: this assumes T_camera_anchored_rig is in the source's native (non-canonical) rig coords,
# but the empirical solve produced an NCore-canonical-rig matrix. The basis
# change rotates an already-canonical T by another 90° about Z.
T_camera_rig_stored = R_native_to_ncore @ T_camera_anchored_rig
```

The comment on the first line records the actual rig frame; the comment on the third line reveals the wrong assumption. A two-axis spot-check on `T_camera_rig_stored` (using a known camera-forward-axis to rig-forward-axis pair) catches this every time.

### Spinning LiDAR model-frame choice (Option A vs Option B)

NCore's writer interprets stored `direction` as "unit-norm ray direction in sensor
coordinates", and `T_lidar_rig` as the sensor-to-rig transform of that **same** sensor
frame. V4 compat reconstructs sensor-frame points as `xyz_m = direction * distance`
(verified in `SequenceLoaderV4.RayBundleSensor.get_frame_point_cloud` in
[`compat.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/v4/compat.py));
downstream rig/world use then depends on `T_lidar_rig` being the sensor-to-rig
transform for that same sensor frame (for example,
`xyz_rig = T_lidar_rig.R @ xyz_m + T_lidar_rig.t`). "Sensor frame" here means the
one in which `T_lidar_rig` is sensor-to-rig, not the manufacturer's native-rotor
frame, unless those happen to coincide.

Two options express the native-to-rig yaw consistently. They are **not** equivalent
in practice -- Option A is the default; Option B has a hard precondition that is easy
to violate and produces silently wrong rig-frame geometry when violated.

| Option                        | `T_lidar_rig.R`        | `column_azimuths_rad[c]`           | Predicted `d_rig` at `(row=n_rows//2, col=0)` |
| ----------------------------- | ---------------------- | ---------------------------------- | --------------------------------------------- |
| **A** (canonical)             | full `R_native_to_rig` | `native_az[c] - yaw_native_to_rig` | ≈ `[1, 0, 0]`                                 |
| **B** (advanced; conditional) | full `R_native_to_rig` | `native_az[c]` (unchanged)         | `R_native_to_rig @ [1, 0, 0]_native`          |

**Building `T_lidar_rig.R` from axis directions.** Each column of the rotation matrix is
the rig-frame coordinates of the corresponding sensor axis. Column-stack construction
makes the direction explicit and avoids accidental transposition:

```python
# T_lidar_rig.R encodes "where each sensor axis lands in the rig frame"
# (each COLUMN is the rig-frame coordinates of one sensor axis):
T_lidar_rig_R = np.column_stack([
    sensor_X_axis_in_rig,   # column 0: where sensor +X points in rig
    sensor_Y_axis_in_rig,   # column 1: where sensor +Y points in rig
    sensor_Z_axis_in_rig,   # column 2: where sensor +Z points in rig
])
# Column-ordering check (tautological by construction; only catches stack-order
# bugs, not wrong-vector bugs — for direction verification see the
# "T_sensor_rig rotation direction verification" section above):
# T_lidar_rig_R @ np.array([0, 1, 0]) == sensor_Y_axis_in_rig
```

If the calibration file expresses extrinsics in any convention other than sensor → rig
(the direction NCore expects for `T_sensor_rig`), convert to that direction first. Two
common cases:

- **`R_rig_to_sensor` (rig → sensor)**: transpose the rotation
  (`T_lidar_rig_R = R_rig_to_sensor.T`); for the translation, take
  `t_sensor_rig = -R_rig_to_sensor.T @ t_rig_sensor`.
- **Chained transforms** (e.g. sensor → another sensor → rig, or sensor → world ← rig):
  compose the chain into a single 4×4, then invert if the composed direction is
  rig → sensor. Use full SE(3) inversion, not just `R.T`, when translations are involved.

**Option A (default).** Apply the formula from "Canonical azimuth-correction formula"
above:

```
az_correction = atan2(T_lidar_rig[1, 0], T_lidar_rig[0, 0])
column_azimuths_rad[col] = sensor_native_azimuth[col] - az_correction
```

The column-0 ray in the spinning model frame then aligns with rig +X after
`T_lidar_rig` is applied. Stored `direction` (unit-norm ray) lives in this
yaw-corrected sensor frame, matching the frame named by `T_lidar_rig`: the yaw is
represented once in the azimuth schedule / stored `direction`, and downstream
`T_lidar_rig.R @ xyz_m + T_lidar_rig.t` then lands the point in rig coordinates.
Use Option A whenever `|az_correction| > 1e-3` rad (any sensor with a non-trivial
yaw in its raw extrinsic).

**Option B (advanced; conditional).** Permitted **only** when stored `direction` is
XYZ-derived from real sensor-frame XYZ produced by the dataset's reader (or by a
re-implementation that has been independently verified against the dataset's
reference output). Then `direction` already lives in the native sensor frame,
`T_lidar_rig` is the sensor-to-rig transform of that **same** native frame, and
reconstruction is consistent regardless of the per-column azimuth schedule (the
schedule only affects `model_element[:, 1]` binning and `elements_to_sensor_rays`
validation, not point geometry). The trap is seductive: hand-rolling
`xyz = [cos(az_native)*cos_e, sin(az_native)*cos_e, sin_e] * distance` from polar
data with un-corrected azimuths is mathematically identical to model-derived from
the un-shifted schedule and rotates the entire point cloud by `az_correction`
relative to the rig. Reach for Option A unless you have read XYZ directly from a
verified reader output, or from a sensor-frame XYZ field the dataset publishes as
raw data.

Record the choice under `spinning_lidar_model_frame_choice` (`"option_a_rig_aligned"`
or `"option_b_native_aligned"`) with a `worked_example` mapping a specific
`(row, col)` to `d_rig`; `validators/build_ncore_converter_validator.py` checks it against the
stored intrinsics + `T_lidar_rig`. The hard check `check_lidar_forward_ray_in_rig`
catches the case where no column ray anywhere aligns with rig +X (a 360° spinner
always has _some_ column landing near rig +X regardless of how the cloud is
rotated, so this hard check is necessary but not sufficient). The companion warning
`warn_lidar_forward_ray_rig_axis_residual` samples the four cardinal model
azimuths `{0, π/2, π, 3π/2}` and fires when any of them fails to project to the
expected rig direction within ~14° -- this surfaces wholesale residual yaw between
`T_lidar_rig.R` and `column_azimuths_rad` at the LiDAR-only step, but legitimate
Option B converters also trip it. The authoritative cross-sensor arbiter is
`check_cuboid_lidar_overlap`; see
[validate-ncore-data common-failures.md](../validate-ncore-data/common-failures.md)
for the demotion rule.

### Spinning LiDAR model parameter pitfalls

Two parameters are silent correctness traps — wrong values produce no error at write time but
cause visible reconstruction artifacts in NuRec:

**`spinning_direction`** — a property of the stored `column_azimuths_rad` array, **not** of
the physical rotor. NCore validates ordering via `relative_angle(azim[0], azim,
spinning_direction)` and asserts the result is strictly increasing (see "`column_azimuths_rad`"
below for the exact check). The decision rule is therefore mechanical and reads only from the
data:

- If the per-column azimuth array (after dedup, projected to `[0, 2π)`) is monotonically
  **increasing**, use `"ccw"`.
- If it is monotonically **decreasing**, use `"cw"`.

The physical rotor direction and the dataset's azimuth measurement convention (which axis is
0, which sign is positive) compose to determine whether the stored array increases or
decreases. Do **not** infer `spinning_direction` from a verbal "clockwise" / "counter-clockwise"
label in the dataset documentation or sensor datasheet -- the same physical rotor produces
opposite stored monotonicities under different azimuth measurement conventions, and the
verbal label maps to whichever was reported (hardware or stored data) without telling you
which. Verify directly against the per-column azimuth values from the source data.

The bundled template's skeleton defaults to `"ccw"`; the wrong value mirrors the Y-axis of every
direction vector and produces a Z-flip in NuRec.

**Beam elevations are non-uniform for every spinning-LiDAR family** -- do not use
`linspace` for `row_elevations_rad`. Always read actual per-beam inclination angles from the
sensor calibration (often shipped alongside the recording as a per-unit JSON / XML / calibration blob
rather than as a raw-data field). Elevations must be strictly descending; reverse if the
calibration is ascending. If raw `ring_id` is not sorted by elevation (common on high-channel
sensors where ring ids reflect firing order), build a `ring_id → row_index` permutation and
apply it when populating `model_element[:, 0]`. If you derive elevations from point data, nudge
exact duplicates by ~1e-6 rad -- NCore asserts `np.diff > 0` (strict).

**Per-beam azimuth offsets.** Zero is correct for most spinning LiDARs. Some sensors ship a
non-zero per-beam offset array (queryable via the dataset's reader or via a calibration file
shipped with the recording); when present, it maps directly to `row_azimuth_offsets_rad`. Omitting a
non-zero offset table misplaces every beam's points within each column and degrades LiDAR↔camera
alignment.

**`column_azimuths_rad`** — must reflect the actual per-column azimuth at frame start:

- NCore validates ordering via `relative_angle(azim[0], azim, spinning_direction)` and asserts
  the result is strictly increasing. This means each successive column must be further along in
  the spin direction from column 0. Raw azimuth values can use any range (e.g. `(-π, π]`,
  `[0, 2π)`, negative values) — NCore projects to `[0, 2π)` internally before comparing.
- **Pairing with `spinning_direction` is mechanical** (this is how NCore decides the array is
  valid; do not derive `spinning_direction` from a verbal label):
  - If `column_azimuths_rad` (mod 2π) is monotonically **increasing** → `spinning_direction = "ccw"`.
  - If `column_azimuths_rad` (mod 2π) is monotonically **decreasing** → `spinning_direction = "cw"`.
  - For example, `linspace(0, 2π, N, endpoint=False)` works for `"ccw"` and
    `linspace(0, -2π, N, endpoint=False)` works for `"cw"`, but these are convenient special
    cases, not the only valid inputs.
  - Equivalently, when extracted from a free-spinning rotor stream:
    `np.all(np.diff(np.unwrap(column_azimuths_rad)) > 0)` → `"ccw"`;
    `np.all(np.diff(np.unwrap(column_azimuths_rad)) < 0)` → `"cw"`.
    Mixed signs mean the schedule is not a single continuous rotation -- fix the extraction
    rather than picking a direction.
- Some datasets embed per-column sensor pose or heading data (e.g. per-column IMU measurements
  or per-pixel pose range images). When available, derive azimuths from that data rather than a
  synthetic sweep; the actual starting azimuth is determined by the sensor's heading at the
  beginning of the frame and is generally not 0.
- A wrong starting azimuth rotates the entire point cloud horizontally and, combined with a wrong
  `spinning_direction`, compresses the apparent 3-D extent of the scene.

### World frames

(See NCore spec for the `world` / `world_global` frame definitions.) Converter-side
constraints:

- **Re-reference `T_rig_world` to the first ego pose.** All intermediate pose math is in
  float64; cast to float32 only immediately before `store_dynamic_pose`. Recommended pattern:

  ```python
  from ncore.impl.common.transformations import se3_inverse, PoseInterpolator

  T_world_world_global = poses_f64[0].copy()  # stored as world->world_global (float64)
  poses_f64 = se3_inverse(T_world_world_global)[None] @ poses_f64
  poses_f32 = poses_f64.astype(np.float32)
  ```

  NuRec uses the `world→world_global` edge for scene extent, geo-location, and Physical AI
  USDZ export. The variant
  `inv(poses_f64[0]) @ poses_f64` + `world→world_global = eye(4)` re-references identically
  on the dynamic trajectory but discards the absolute frame-0 pose; prefer the form above
  unless the dataset exposes a real ECEF/UTM calibration.

- **Skipping re-referencing is wrong even when the dataset's world origin looks local.** Even
  a few-km offset causes sub-centimeter float32 precision loss that surfaces as diagonal
  streaking under per-ray motion compensation. Apply unconditionally.

- **`T_world_world_global` is optional on the pose graph** (and never
  required by NCore). When the edge is absent, NuRec defaults it to identity (`eye(4)`)
  float64. When the edge **is** present its dtype must be `float64` (NuRec requires `float64`
  for the geo-location / USDZ export paths that consume it; `validate_poses.py` enforces this
  only when the edge exists). Per NCore's spec `world_global` is ECEF. Storing the edge remains
  **recommended** whenever the dataset has a meaningful global frame: the mandatory rig-trajectory
  re-referencing below already computes `poses_f64[0].copy()`, so storing it as
  `world→world_global` is nearly free and preserves the absolute frame-0 pose that NuRec's
  geo-location / USDZ export consume. Omitting the edge is safe for plain reconstruction
  (NuRec falls back to identity, so `world_global` collapses onto `world` = rig-at-t=0) but
  loses absolute geo-referencing. When the dataset exposes a real ECEF/UTM calibration, store
  that transform instead of `poses_f64[0]`. The rig-trajectory re-referencing itself is **not**
  optional -- it is float32-precision hygiene independent of whether the edge is stored.

#### World re-referencing and world-anchored data

Re-referencing the rig trajectory (non-identity `T_world_world_global` applied
to dynamic rig poses) splits `world` (rig at t=0) from `world_global` (original).
Every world-anchored quantity (cuboid centroids **and** rotations, map points,
road-graph geometry, anchor poses, ...) must end up in the same `world` as the
rig. Two valid strategies:

1. **Pre-transform on write:** apply `se3_inverse(T_world_world_global)` to the
   quantity, store with `reference_frame_id="world"`.
2. **Store in world_global:** leave data untransformed, set
   `reference_frame_id="world_global"`; the pose graph resolves the static edge.

Leaving data in `world_global` while labeling it `"world"` is the canonical
failure mode (`check_cuboid_rig_proximity` catches it).
Record the chosen strategy under `cuboid_world_anchored_frame` with evidence the
stored values match the declared frame after re-referencing.

## Transformations

Edges a converter stores (see NCore spec for the `T_a_b` / `T_source_target` notation and
pose-graph static/dynamic semantics):

| Transform              | From   | To           | Static/Dynamic | Writer               |
| ---------------------- | ------ | ------------ | -------------- | -------------------- |
| `T_rig_world`          | rig    | world        | dynamic        | `store_dynamic_pose` |
| `T_camera_rig`         | camera | rig          | static         | `store_static_pose`  |
| `T_lidar_rig`          | lidar  | rig          | static         | `store_static_pose`  |
| `T_radar_rig`          | radar  | rig          | static         | `store_static_pose`  |
| `T_world_world_global` | world  | world_global | static         | `store_static_pose`  |

### Pose dtypes

NuRec's `FrameConversion.transform_poses` uses `float32` transformation matrices
(`FrameConversion.matrix` is asserted `float32`; `get_transformation_matrices()` returns
`float32` T, S used in `T @ T_poses @ S`). Mismatches surface **late** -- typically as
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: double != float` at
`Get Lidar Point Clouds` -- and are not caught by standard validation.

| Pose                                                                     | dtype stored via `store_*_pose` | Why                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `world→world_global` (static)                                            | `float64`                       | NuRec requires `float64` here for the geo-location / USDZ export paths that consume this edge. `float32` causes numerical divergence.                                                                                                             |
| All other static poses (`T_camera_rig`, `T_lidar_rig`, `T_radar_rig`, …) | `float32`                       | NuRec's `FrameConversion.matrix` is asserted `float32`. `float64` here triggers the `double != float` matmul error during LiDAR point cloud assembly.                                                                                             |
| `T_rig_world` (dynamic)                                                  | `float32`                       | NuRec promotes stored rig poses to `float64` for trajectory interpolation, but batch assembly casts back to `float32`. Storing `float32` is the expected NCore convention and avoids the `double != float` error in downstream matmul operations. |

Note that NCore's singular accessors (`get_static_pose`, `get_dynamic_pose`) always return
`float64` regardless of stored dtype. Use the plural forms (`get_static_poses`,
`get_dynamic_poses`) to inspect the _stored_ dtype. The stored dtype is what NuRec observes at
load time.

This split dtype requirement exists because NuRec expects pose inputs to be typed consistently
at load time. Converters must apply the table above exactly.

### Pose trajectory density

NuRec interpolates `T_rig_world` at **every per-ray timestamp** during motion compensation.
Sparse waypoints produce a staircase approximation and manifest as **rowing** -- horizontal
stripe artifacts across the LiDAR sweep. Target waypoint spacing smaller than the LiDAR sweep
duration (< 50 ms for a 10 Hz spinning LiDAR).

**Rules:**

- **Verify pose-timestamp correspondence before pairing.** Some datasets provide a pose field
  and a timestamp field on the same record that refer to different moments (e.g. pose at
  mid-sweep, timestamp at sweep start). A mispairing shifts every frame by the offset.
- **Combine every available pose source.** Priority order:
  1. Per-camera-image poses (primary) -- iterate **all** cameras; N×M waypoints.
  2. Frame-level / LiDAR-level poses, if at distinct timestamps.
  3. IMU / GPS / odometry if available.
     Concatenate, `np.unique` by timestamp, sort.
- **Dedup applies to single-source feeds too.** Even with one pose stream the timestamp
  axis can be non-strict-monotonic (replay buffers, async logging, retry-on-resync). Always
  dedup by timestamp before passing the array to `PoseInterpolator` -- duplicates produce
  zero-duration intervals that yield silent garbage SLERP outputs rather than an explicit
  error. Inspect the duplicates first: when paired RT matrices are identical, `np.unique`
  on timestamps with `return_index=True` is sufficient; when they differ, the disposition
  is dataset-specific (average, keep-first by recording order, or treat as a hard error)
  and must be documented in the converter's `critical_parameter_justifications`.
- **Do not** densify from per-pixel or per-column pose grids embedded in polar LiDAR data.
  Invalid-return pixels contain zero/garbage poses; mixing them with clean poses produces
  angular rates in the thousands of deg/s. If such grids are the **only** source, filter by
  valid-return mask and cap angular rate < 180 deg/s and translation rate by expected vehicle
  speed before storing.
- **Boundary extrapolation.** If velocity data is available, extrapolate to the sequence start
  / end so the trajectory covers the first and last sweeps. When the dataset's angular
  velocity is in the vehicle/body frame, rotate to global before applying:
  `omega_global = T_rig_world[:3, :3] @ omega_vehicle`. Using body-frame angular velocity in
  global-frame extrapolation introduces up to ~0.04 deg per 50 ms.

**Anti-patterns:**

- Do NOT compute a static sensor→rig pose from a single frame's per-image pose. Interpolate
  the dense rig trajectory at each sensor-image timestamp, invert via
  `se3_inverse(T_rig_world_i) @ T_sensor_world_i`, then average rotations via SVD (chordal L2)
  and translations via arithmetic mean.
- Do NOT linearly interpolate or extrapolate 4×4 SE(3) matrices directly
  (`p0 + alpha*(p1-p0)`) -- the rotation block becomes non-orthogonal. Use NCore's
  `PoseInterpolator` (SLERP + linear) from `ncore.impl.common.transformations`.
- Do NOT derive the trajectory from only one pose source when multiple are available.

### Rig-pose pipeline (canonical data flow)

The canonical pipeline every `_write_poses (dynamic)` must implement:

```mermaid
flowchart LR
  rawPoses[raw sensor poses<br/>per-camera + per-frame + IMU/GPS] --> f64[collect in float64]
  f64 --> dedup["np.unique + sort by timestamp"]
  dedup --> storeWG["T_world_world_global = poses[0] (f64)<br/>store_static_pose('world','world_global', ...)"]
  storeWG --> reref["poses = se3_inverse(T_world_world_global)[None] @ poses<br/>(NuRec uses the edge above for geo-location / USDZ export)"]
  reref --> extrap["PoseInterpolator (SLERP + linear)<br/>extrapolate to t_start / t_end using velocity"]
  extrap --> cast["cast to float32 (last step only)"]
  cast --> store[store_dynamic_pose('rig','world', poses_f32, timestamps_us)]
```

Invariants: float64 end-to-end with a single cast to float32 at the edge; re-reference via
`se3_inverse(T_world_world_global)` (not `inv(poses[0])` + identity -- the canonical form
preserves the absolute frame-0 pose in `world→world_global`); `PoseInterpolator` for any
interpolation or extrapolation.

**Endpoint anchoring.** The NCore writer asserts `timestamps_us[0] == interval.start` and
`timestamps_us[-1] == interval.stop - 1` exactly. After dedup / sort / re-reference, add
boundary samples before casting. When a boundary lies inside the measured pose timestamp
range, evaluate it with `PoseInterpolator.interpolate_to_timestamps()` so the boundary pose
is consistent with the surrounding trajectory:

```python
interp = PoseInterpolator(poses_arr, uniq_ts)
boundary_ts = np.array([t_start_us, t_end_us], dtype=np.uint64)
boundary_ts = boundary_ts[(uniq_ts[0] <= boundary_ts) & (boundary_ts <= uniq_ts[-1])]
target_ts = np.unique(np.concatenate([uniq_ts, boundary_ts])).astype(np.uint64)
poses_arr = interp.interpolate_to_timestamps(target_ts, dtype=np.float64)
uniq_ts = target_ts
```

Use velocity extrapolation via `PoseInterpolator` instead of nearest-neighbor copies when
the boundary falls just outside the measured trajectory range and the gap is within the
accepted extrapolation limit. Otherwise, clip the sequence interval to the measured pose
coverage.

## Timestamps

- **Unit**: always microseconds (µs), dtype `uint64`.
- **Interval**: half-closed `[start, stop)` where `stop = last_timestamp_us + 1`. **Construction**: `HalfClosedInterval.from_start_end(start, end)` treats `end` as **inclusive** and internally stores `stop = end + 1`, so pass `last_timestamp_us` (not `last_timestamp_us + 1`). Passing `+1` yourself double-increments and fails the dynamic-pose coverage assertion.
- **Dynamic pose timestamps**: `timestamps_us[0]` must equal `interval.start`; `timestamps_us[-1]` must equal `interval.stop - 1` (i.e. exactly span the interval).
- **Frame timestamps**: each sensor frame has `start_timestamp_us` and `end_timestamp_us`; both must fall within the sequence interval.

### Sequence frame-range intersection

The converter must clip the NCore sequence to the **intersection** of every required
modality's timestamp coverage:

```
sequence_interval = intersect(
    pose_timestamps.envelope,
    cuboid_timestamps.envelope,            # if cuboids exist for the sequence
    *[cam.frame_timestamps.envelope for cam in cameras],
    *[lid.frame_timestamps.envelope for lid in lidars],
)
```

Preserve each observation's source frame key or index through filtering. For
every stream, record the retained key/index range and frame count, and detect
missing, duplicate, or nonmonotonic keys. Do not zip independently filtered
camera, LiDAR, or annotation arrays by position: a dropped or duplicated
observation changes every later positional pairing.

When streams have a documented shared key namespace, intersect the available
keys across required modalities first. Then retain the widest common key range
whose complete frame windows fit the timestamp-envelope intersection, and set
the sequence interval to span exactly those frames. Different starting keys,
subsampling, sensor restarts, and interior gaps must remain visible rather than
being renumbered into apparent correspondence.

When streams do not share documented keys, do not invent index correspondence
from similar rates or frame counts. Pair observations by timestamp using a
tolerance justified by the dataset's cadence, synchronization accuracy, and
frame timestamp semantics. Mixed-rate, asynchronous, and gappy streams are
valid on this path.

Use offset comparisons only as diagnostics. Compare equal-index,
adjacent-index, and nearest-timestamp frame midpoints across the sequence:

- A constant integer-frame shift suggests different index origins or unequal
  boundary clipping.
- A step change suggests an interior drop, duplicate, or sensor restart.
- A steadily changing offset suggests a rate or clock mismatch.

Confirm the cause against source keys and raw timestamps before changing
exposure or sweep timestamp semantics.

Frames where one or more required modalities have no data must be excluded even when
the rest of the modalities cover them. Half-coverage frames silently degrade
reconstruction:

- **Cuboids out of coverage**: NuRec skips any cuboid track whose
  `tracks_timestamps_us` range does not bracket the requested frame's
  `[start, end]`, with no warning. Frames at the edge of cuboid coverage render with
  no dynamic-object placement, so moving cars / pedestrians visible in that frame's
  sensor data leak into the static reconstruction as ghosts and motion smears.
- **Pose out of coverage**: caught by the NCore writer's pose / interval assertions
  (the writer requires `pose_timestamps[0] == interval.start` and
  `pose_timestamps[-1] == interval.stop - 1`). A converter that fails to clip will
  not produce a valid shard.
- **Camera or LiDAR out of coverage**: the affected sensor contributes no observation
  at the missing frame; NuRec either errors at sampling time or silently skips the
  frame for that sensor depending on the dataloader path.

Record the chosen interval and the modality that limited each side under
`sequence_frame_range_decision` in `critical_parameter_justifications`, including the
count of frames excluded by clipping and the cuboid track counts expected versus stored
(see ["Tracks lost to interval clipping"](#tracks-lost-to-interval-clipping)).

## LiDAR data

### LiDAR model-parameter source hierarchy

`row_elevations_rad`, `row_azimuth_offsets_rad` and per-ray `model_element` must come
directly from the dataset's per-beam calibration and per-ray metadata. Do NOT substitute a
synthetic `np.linspace(...)` grid for elevations or a direction-binning reconstruction for
`model_element`: a uniform grid is never a legitimate substitute for a real per-beam
elevation table, and direction binning discards per-beam firing-time offsets while collapsing
duplicate rays onto a single cell in a way that `check_lidar_model_accuracy` cannot
distinguish from a correct population.

Permitted sources (pick the highest-ranked one the dataset exposes):

| Rank | Source                                                                 | Fills                                                                                                                                                                                                        | When to use                                                                                                                                                                                                           |
| ---- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Per-point `(row, col)` indices in raw + calibration per-beam elevation | `row_elevations_rad` from calibration; `column_azimuths_rad` from the per-point azimuth stream (when shipped) or from a uniform angular step (see note below); `model_element` directly from per-ray indices | Preferred whenever per-ray metadata (ring/channel, azimuth, column) is exposed -- as embedded fields inside the primary point-cloud file, as separate companion files, or implicitly via a documented point ordering. |
| 2    | Calibration per-beam arrays + structured point ordering                | Calibration populates elevations & per-beam offsets; `model_element` derived from outer/inner index order (`row = i % n_rows`, `col = i // n_rows` or vice versa, verified empirically)                      | Raw point cloud is a flat list but its declared ordering is documented or can be inferred from azimuth monotonicity.                                                                                                  |

If the raw data appears to expose neither per-beam elevation nor per-ray indices, stop and
re-enumerate the dataset. Almost every spinning LiDAR ships both somewhere (calibration JSON,
datasheet, embedded field, or companion file); an apparent absence is almost always a parsing
gap. `warn_lidar_row_elevations_uniform` fires on linspace elevation
tables and `warn_lidar_model_element_all_zero` on `np.zeros((N, 2))` placeholder indices --
both catch converters that fell back to synthesis.

`warn_lidar_model_element_low_coverage` fires when the fraction of `(row, col)` cells
touched by at least one ray is below threshold. Interpretation: a converter that derives
`model_element` from real per-ray indices (Rank 1) typically achieves >50% coverage; below
~10% strongly indicates either a wrong `n_columns` choice (e.g. using the raw column count
when the sensor ships dual-return runs that should be collapsed to the unique-azimuth
count) or a phase-alignment bug that scatters rays into a thin diagonal band of cells.

`column_azimuths_rad` -- always derived from real per-frame data, never synthesized.
Spinning-LiDAR datasets ship per-frame absolute azimuths somewhere (embedded field, companion
`*azimuth*` file, or implicit through documented column ordering). Synthesizing a uniform
`np.linspace(0, 2*pi, n_columns)` is a synthesis artifact in exactly the same sense as a
synthetic `row_elevations_rad`: it discards the sensor's actual per-column firing structure
(dual-return run patterns, near-field channel bursts, non-uniform schedules) and replaces it
with a fiction that happens to share an average angular resolution.

**Phase-locked rotor.** When the rotor is phase-locked to the frame trigger (per-frame
azimuth schedules are identical), use any single frame's per-column azimuths directly as
`column_azimuths_rad`.

**Free-spinning rotor.** When per-frame azimuth schedules differ by a per-frame phase offset
(the inter-column structure -- run-length pattern, per-column step sizes, total angular
coverage -- is preserved across frames; only the absolute starting phase shifts), the
schedule is still a sensor intrinsic. Call the shared helper rather than reimplementing the
recipe:

```python
from ncore_<dataset>.utils import build_lidar_model_from_per_frame_azimuth

column_azimuths_rad, per_frame_phase_offsets_rad, model_element_col_indices = (
    build_lidar_model_from_per_frame_azimuth(
        per_frame_azimuth=...,         # list/array of per-frame absolute azimuth streams
        spinning_direction=...,        # 'cw' or 'ccw'
        per_beam_azimuth_offset=...,   # optional Δ table; pair with row_azimuth_offsets_rad=Δ (Pattern 1)
    )
)
```

The helper implements the algorithm below, self-validates NN-bin error against
`MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG`, and rejects schedules that would violate NCore's strict
monotonicity assertion. It handles cyclic-wrap dedup, integer phase recovery via FFT-based
circular cross-correlation, and modulo-`2*pi` subtraction -- the implementation pitfalls
listed below. Phase-locked rotors are handled by the same helper; the per-frame phase
offsets it returns are ~zero in that regime.

**Precondition.** The helper requires per-frame azimuth that contains **real per-ray Δ
structure** (within-column rays vary by Δ[ring]). For sensors that publish rotor-phase-only
streams (within-column rays share a single azimuth, so the per-frame azimuth carries only
the rotor schedule), the helper's stride detection cannot recover Δ; route to **Pattern 3
(Synthetic-uniform + Encode-in-column)** in the row_azimuth_offsets_rad section below
instead, with explicit Δ added to per-ray azimuth before NN-binning into the synthetic
schedule.

The algorithm description below remains as a fallback for environments where calling the
helper is not practical.

**Algorithm (for reference, in case you need to roll your own):**

1. Pick a reference frame (frame 0 is fine).
2. Extract its absolute per-column azimuth array from the per-frame stream.
3. Collapse duplicate azimuths from dual-return / near-field run patterns so the resulting
   array satisfies NCore's strict monotonicity in `spinning_direction` (verified at
   `RowOffsetStructuredSpinningLidarModelParameters.__post_init__` in
   [`types.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/types.py)).
   The collapsed length is the real `n_columns` -- the unique-azimuth count per rotation,
   not the raw column count. For example, on a 128-beam sensor with mixed 2x/4x dual-return
   run patterns this might be ~1700 unique azimuths instead of ~3600 raw columns; on a
   sensor with clean 2x dual-return runs it is exactly half the raw column count. The exact
   ratio is dataset-specific and must be derived empirically from the per-frame azimuth
   stream.
4. Use that array as `column_azimuths_rad`.
5. For every other frame, estimate the phase offset by cross-correlating the frame's
   azimuth pattern against the reference. Subtract the phase offset from each ray's
   measured azimuth (mod `2*pi`), then NN-bin against the reference schedule to populate
   `model_element[:, 1]`.

NN-bin error is bounded by half a column step -- well under 0.5° on any production spinning
LiDAR. The model's `column_azimuths_rad` now matches the sensor's actual firing pattern.

**Implementation pitfalls (when not using the helper).** Each of these silently breaks the
0.5° angular-error threshold or NCore's strict monotonicity assertion:

1. **Order-preserving dedup, not `np.unique`.** NCore requires monotonicity _in the
   spinning direction_, not numerical sort order. `np.unique` reorders azimuths
   numerically and discards the spin-direction wrap-around. Use a dict-based or
   `pandas.unique`-style dedup that preserves first-occurrence order in the per-frame
   firing sequence.
2. **Phase estimation by cross-correlation, not column-0 difference.** Frame N's column 0
   does not correspond to frame 0's column 0 -- the schedule is cyclically rotated by a
   few columns from frame to frame. Naive `phase = frame_N_az[0] - frame_0_az[0]`
   measures azimuth at index 0, but those are different logical columns of the same
   cyclic schedule. Cross-correlate the full azimuth array against the reference to
   recover the cyclic shift, then convert to azimuth offset.
3. **Subtract phase modulo `2*pi`.** `per_ray_az - phase` near `0` / `2*pi` wraps to
   negative values that NN-bin to the wrong end of the schedule. Always wrap with
   `np.mod(..., 2*np.pi)` before binning.
4. **Choose a row_azimuth_offsets_rad pattern up front.** For Pattern 1 (Sensor-native +
   Preserve), NN-bin `(per_ray_az - Δ[ring])` against the rotor schedule and store the
   real Δ in `row_azimuth_offsets_rad`. For Patterns 2 and 3 (Encode-in-column), NN-bin
   the full `(rotor + Δ)` per-ray azimuth against a dense schedule (sensor-native or
   synthetic) and store `row_azimuth_offsets_rad = 0`. See the row_azimuth_offsets_rad
   section below for the full taxonomy.

**Companion rule for `row_azimuth_offsets_rad`.** NCore's `elements_to_sensor_rays`
evaluates `azimuth(row, col) = column_azimuths_rad[col] + row_azimuth_offsets_rad[row]`
unconditionally (verified at
`RowOffsetStructuredSpinningLidar.elements_to_sensor_angles` in
[`lidar.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/sensors/lidar.py)
and the row-offset spinning model examples in
[`sensor_models.rst`](https://github.com/NVIDIA/ncore/blob/v19.8.0/docs/data/sensor_models.rst)),
and its FOV assertion `_init_angles_to_columns_map` validates the `n_rows × n_columns`
grid against parameter-derived FOV at every `(row, col)` cell (verified at
`RowOffsetStructuredSpinningLidar._init_angles_to_columns_map` in
[`lidar.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/sensors/lidar.py)).
Δ is always added; there is no conditional path. NCore's stated canonical target sensor
for non-zero Δ is **Hesai P128** (verified at
the Hesai P128 entries in
[`sensor_models.rst`](https://github.com/NVIDIA/ncore/blob/v19.8.0/docs/data/sensor_models.rst)
and the `RowOffsetStructuredSpinningLidarModelParameters` docstring in
[`types.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/types.py)).

NCore's writer requires only that the stored `direction` is `float32`, shape `[N, 3]`,
unit-norm to `1e-4` (verified at
`LidarSensorComponent.Writer.store_frame` in
[`components.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/v4/components.py));
it does **not** require stored direction to match `elements_to_sensor_rays(model_element)`
at any tolerance. Stored direction is the source of truth for point-cloud reconstruction
(`xyz_m = direction × distance` in `SequenceLoaderV4.RayBundleSensor.get_frame_point_cloud`
in [`compat.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/v4/compat.py));
`column_azimuths_rad` is consulted only for shutter-time / model-element paths.
`xyz_m` lives in the **same sensor frame** that `T_lidar_rig` is sensor-to-rig of
(see "Spinning LiDAR model-frame choice" above): under Option A this is the
yaw-corrected sensor frame whose +X aligns with rig +X; under Option B it is the
manufacturer's native sensor frame.

The converter author chooses across two binary axes:

- **Column schedule**: sensor-native (the manufacturer's rotor-phase schedule, possibly
  recovered from per-frame data) vs. synthetic-uniform (a fine-resolution uniform schedule
  the converter author constructs).
- **Per-beam offset Δ[ring] handling**: stored explicitly in `row_azimuth_offsets_rad`,
  OR encoded implicitly via per-ray column-index choice (NN-bin against a fine-enough
  schedule).

All three patterns below pair with **XYZ-derived `direction = xyz / ||xyz||`**
(sensor-frame XYZ published as a raw field or emitted by the dataset's reader with
all frame corrections applied — not hand-rolled from polar coordinates; see
"Spinning LiDAR model-frame choice").

Use the NCore model checks to choose a pattern.
Start with Pattern 1 when its calibration inputs are available. Construct
`RowOffsetStructuredSpinningLidarModel` with
`angles_to_columns_map_init=True` and then require a representative converted
frame to pass the existing 0.5° stored-versus-model direction check. If either
check fails, try Pattern 2 when the native schedule is recoverable. Use Pattern
3 if it is not, or if Pattern 2 still fails. For Pattern 3, start at native
column resolution and increase by the smallest integer factor that passes both
checks.

#### Pattern 1: Sensor-native + Preserve (default)

The canonical NCore-blessed pattern for any sensor whose rotor-phase schedule is
recoverable AND whose per-beam offsets Δ[ring] are published in manufacturer calibration.

- `column_azimuths_rad` = sensor's rotor-phase schedule (sensor-derived, monotonic in
  `spinning_direction`).
- `row_azimuth_offsets_rad` = real per-beam offsets Δ from manufacturer calibration (or
  `np.zeros(n_rows)` if the sensor genuinely has none).
- `model_element[:, 1]` = NN-bin of `(per_ray_az - row_azimuth_offsets_rad[ring])` against
  `column_azimuths_rad` -- i.e. NN-bin the **rotor-phase component**, not `(rotor + Δ)`.
- `direction = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)`.
- Helper: `bin_per_ray_to_columns` (in `ncore_template/ncore_example/utils.py`)
  with `per_beam_az_offset_rad=Δ` and `per_ray_beam_idx=ring_idx` to produce the rotor-only
  NN-bin output.
- Canonical sensors: Hesai P128 (cited above), Velodyne VLP/HDL, Ouster OS1/OS2
  (Δ ≈ ±0.5°). Also covers any spinning LiDAR whose calibration ships Δ = 0.

#### Pattern 2: Sensor-native + Encode-in-column

For sensors whose rotor schedule is recoverable, but whose explicit per-beam
offset representation is unavailable or fails the model checks above.

- `column_azimuths_rad` = sensor's full schedule.
- `model_element[:, 1]` = NN-bin of full per-ray azimuth `(rotor + Δ[ring])` against
  `column_azimuths_rad`.
- `row_azimuth_offsets_rad = np.zeros(n_rows)`.
- `direction = xyz / ||xyz||`.
- Helper: `bin_per_ray_to_columns` (in `ncore_template/ncore_example/utils.py`)
  with `per_beam_az_offset_rad=None` (full per-ray azimuth in, no rotor/Δ split).
- Δ is encoded by which column-index each ray lands in. Resolution loss is bounded by
  half the column step.

#### Pattern 3: Synthetic-uniform + Encode-in-column (fallback)

For sensors whose native rotor schedule cannot be cleanly recovered: rotor-phase-only
streams (within-column rays share the same azimuth -- the per-frame extractor has no
per-ray Δ structure to extract), dual-return / near-field run patterns that scramble the
schedule, or native-resolution encode-in-column parameters that fail the model checks.

- `column_azimuths_rad` = synthetic uniform fine-resolution schedule with `n_columns`
  chosen as the smallest integer multiple of the native column count that passes the
  model checks above. The schedule must satisfy NCore's monotonicity rule (relative angle
  along `spinning_direction` strictly increasing, no wrap at the start element; verified
  at
  `RowOffsetStructuredSpinningLidarModelParameters.__post_init__` in
  [`types.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/types.py)):
  - CCW (`spinning_direction = +1`):
    `column_azimuths_rad = (np.arange(n_columns) * step) % (2*np.pi)`.
  - CW (`spinning_direction = -1`):
    `column_azimuths_rad = ((-np.arange(n_columns) * step) % (2*np.pi))`.
- `model_element[:, 1]` = NN-bin of `(rotor + Δ[ring])` against the synthetic schedule.
- `row_azimuth_offsets_rad = np.zeros(n_rows)`.
- `direction = xyz / ||xyz||` (mandatory; otherwise Δ is lost on the way to disk).
- Helper: `bin_per_ray_to_columns` (in `ncore_template/ncore_example/utils.py`)
  with `per_beam_az_offset_rad=None`.

#### Trade-off subsection

- **Pattern 1** is the default whenever the sensor publishes a Δ table and the rotor
  schedule is recoverable. Model parameters and stored direction agree to manufacturer
  precision; no resolution loss. Under NCore's writer contract, model-derived directions
  and XYZ-derived directions are interchangeable here -- but XYZ-derived is preferred
  for consistency with the other patterns and with NCore's reference converters.
- **Pattern 2** is the option when explicit row offsets are unavailable or fail the FOV
  self-consistency check, and the sensor's rotor schedule is dense enough that half its
  column step stays under the 0.5° model-error limit -- a native step finer than 1°.
  Leave real margin: at 1° the column binning alone consumes the whole budget, which the
  elevation table and row-offset handling draw on as well.
  Resolution loss in `model_element[:, 1]` is bounded by **half the column step**, not
  by `|Δ|`. Stored XYZ-derived direction carries the true per-ray geometry independently
  of column-bin precision.
- **Pattern 3** is the fallback when Patterns 1 and 2 are infeasible: rotor-phase-only
  streams, dual-return / near-field run patterns that scramble the native schedule, or
  native-resolution model error above 0.5°. Resolution loss is bounded by **half the
  synthetic column step** `π / n_columns`, not by `|Δ|`. Same direction-precision
  properties as Pattern 2.

Never synthesize per-beam offsets when calibration omits them -- default to zeros and use
Pattern 2 or Pattern 3 (encode Δ via column-index choice). Never synthesize a
heading-unaware `linspace(0, 2*np.pi)` placeholder; Pattern 3's synthetic schedule is
principled because the start point satisfies NCore's monotonicity rule and the schedule
is paired with per-ray NN-binning of `(rotor + Δ[ring])` so the column-index encodes
the true firing geometry.

Record the chosen rank per sensor under `lidar_model_source` in
`critical_parameter_justifications` with evidence = calibration path + field name.

### Per-ray metadata discovery

Before authoring `_write_spinning_lidar_frame`, locate every per-ray metadata stream the
dataset exposes. They appear in one of three forms -- check all three:

1. **Embedded fields inside the primary point-cloud payload**: extra named fields or
   channels beyond `(x, y, z)`. Depending on the container format this can be additional
   columns in a structured per-point record (e.g. `channel`, `ring`, `ring_idx`, `azimuth`,
   `time`, `timestamp`), extra channels alongside `xyz` in a range image, or additional
   arrays beyond `xyz` in a multi-array archive -- inspect the container's field / channel
   / key listing.
2. **Separate companion files** next to the primary point cloud: additional per-frame
   files (e.g. named `*ring*`, `*azimuth*`, `*timestamp*`) sitting in the same directory
   as the point cloud itself. Enumerate the frame directory and check every file whose
   length or shape is compatible with the point count.
3. **Implicit structured ordering**: a flat `(N,)` point list that is actually a
   `(n_rows x n_columns)` grid stored row- or column-major. This is common when the
   container only exposes a one-dimensional point count and no grid shape. Verify by
   checking whether consecutive points at a given stride share an azimuth (column-major)
   or an elevation (row-major). When this holds, `model_element` comes from index
   arithmetic with no extra data needed.

Common naming patterns across spinning-LiDAR datasets, applicable to either embedded fields
(form 1) or separate companion files (form 2):

| Pattern                   | Typical content                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| `*ring*`, `*channel*`     | Per-point ring / channel / beam index (maps to `model_element[:, 0]`).                                  |
| `*azimuth*`, `*azim*`     | Per-point azimuth angle (derives or replaces `column_azimuths_rad`; binnable to `model_element[:, 1]`). |
| `*column*`, `*col*`       | Per-point column index (maps directly to `model_element[:, 1]`).                                        |
| `*timestamp*`, `*time*`   | Per-point firing time (populates the per-point `timestamp_us` field).                                   |
| `*intensity*`, `*return*` | Per-point intensity / return flag (populates stored `intensity` field).                                 |

### Sweep timestamp convention

`start_timestamp_us` = first-column fire time; `end_timestamp_us` = last-column fire time.
Per-point `timestamp_us` spans `[start, end]` monotonically.

**Common mistake**: some datasets provide a "frame timestamp" at the sweep **middle** (e.g.
the time of the vehicle pose). Centering the window around it (`frame_ts ± half_sweep`)
shifts every per-ray timestamp by ~50 ms at 10 Hz. NuRec then motion-compensates every point
with a pose from the wrong time. Always check the dataset docs to determine whether the
frame timestamp is start, center, or end, and reinterpret as the start accordingly.

### LiDAR frame timestamp sourcing

LiDAR `frame_timestamps_us` must be on the **same time base** as the rig pose trajectory. If
multiple timestamp sources exist (frame-level metadata, per-point timestamps in raw or
compensated data), cross-check: compare sweep-level values against the min/max of per-point
timestamps for the same frame. Record the finding in `critical_parameter_justifications`
under `lidar_frame_timestamp_meaning`.

### Motion compensation

NCore V4 expects **uncompensated** LiDAR data -- points in the sensor frame at each point's
own measurement time. NuRec applies motion compensation internally from per-point
timestamps + the pose trajectory.

### LiDAR direction derivation

NCore V4's `direction` is per-ray unit-norm ray direction **in sensor coordinates**. The
derivation depends on the raw data format.

**Recommendation: always store XYZ-derived direction (Case 2 / Case 3 below).**
NCore's writer accepts any unit-norm direction (verified at
`LidarSensorComponent.Writer.store_frame` in
[`components.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/v4/components.py));
the choice is a converter-quality matter, not an NCore-required check.

**Case 1 — polar range image, model-derived direction (preserved for completeness only).**
The intrinsic beam model gives:

```python
az = column_azimuths_rad[col_idx] + row_azimuth_offsets_rad[row_idx]
elev = row_elevations_rad[row_idx]
cos_e = np.cos(elev)
direction = np.stack([cos_e * np.cos(az), cos_e * np.sin(az), np.sin(elev)], axis=-1)
distance = range_image[row_idx, col_idx]
```

- **Risk specific to Case 1.** When `row_azimuth_offsets_rad = 0` AND
  `column_azimuths_rad` is rotor-phase-only (does not encode Δ via dense column-index --
  i.e. you are _not_ in Pattern 2 or Pattern 3), Case 1 silently drops Δ, producing
  physically wrong directions. The stored `model_element` -- itself derived from the
  same model parameters -- agrees with the model-derived direction trivially, so
  `check_lidar_model_accuracy` passes tautologically and the bug is invisible to the
  ERROR-level accuracy check. The new `warn_lidar_direction_collapsed_to_model`
  WARNING flags this collapse (median + P99 angular error < 0.01° AND
  `row_azimuth_offsets_rad ≈ 0`).
- Do **not** call `elements_to_sensor_rays` to produce stored directions -- it is for
  validation only; using it couples `column_azimuths_rad` to `T_lidar_rig` in a way that
  silently breaks if only one is updated.
- Do **not** route through world-frame cartesian (via per-pixel poses or a reader's cartesian
  helper) and then decompensate back. The round-trip introduces diagonal-streaking
  numerical error. The beam model gives you sensor-frame directions directly.
- Per-ray pose grids (per-pixel 4×4 or 6-tuple) encode where the sensor was when each ray
  fired -- useful for world-frame reconstruction, not needed for NCore storage.

**Case 2 — pre-compensated world-frame XYZ (no range image).** Decompensate to sensor frame:
`T_world_sensor_end = inv(T_sensor_rig) @ inv(T_rig_world_end)` applied to the world points,
then `MotionCompensator.motion_decompensate_points(sensor_id, xyz_sensorend, timestamp_us,
frame_start_timestamp_us, frame_end_timestamp_us)`, then
`distance = ||xyz||`, `direction = xyz / distance[:, None]`. Construct the compensator via
`MotionCompensator.from_sensor_rig(sensor_id, T_sensor_rig, T_rig_worlds,
T_rig_worlds_timestamps_us)`.

**Case 3 — sensor-frame XYZ (no compensation applied).** Normalize:
`direction = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)`.

### Stored ray-bundle fields

For dtype / shape details, see NCore data-formats spec. Converter-relevant constraints:

- `direction` must be unit-norm (validate with `np.allclose(norms, 1.0, atol=1e-3)`).
- `model_element[:, 0] < n_rows`, `model_element[:, 1] < n_columns`.
- Angular error between stored directions and the model-predicted directions (from
  `model_element`) must be ≤ 0.5° for NuRec viability. Validation uses
  `RowOffsetStructuredSpinningLidarModel.elements_to_sensor_rays` for the comparison -- a
  mismatch with directions computed via direct trig from the same `column_azimuths_rad` /
  `row_elevations_rad` indicates an index-mapping error in `model_element`.
- **NCore strict in-FOV assertion.** Each stored ray's direction must fall inside the
  model FOV defined by `(row_elevations_rad, column_azimuths_rad, row_azimuth_offsets_rad)`
  -- NCore asserts this strictly per `(row, col)` cell (verified at
  `RowOffsetStructuredSpinningLidar._init_angles_to_columns_map` in
  [`lidar.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/sensors/lidar.py)).
  Calibrated per-beam elevation tables are nominal; real measurements scatter slightly past
  the table extremes due to sensor noise and low-precision storage formats. A small
  fraction of rays (empirically O(0.1-1%)) will be flagged. Drop these rays before write;
  do **not** widen the elevation table to admit them, and do **not** clip directions in
  place (clipping moves the ray off its measured azimuth/elevation pair, breaking
  `model_element` fidelity).

### Intensity

- Range `[0.0, 1.0]`, dtype `float32`. Normalize from sensor-native range if needed.
- Use `np.tanh(raw_intensity)` when raw intensity is an unbounded positive float (common for
  spinning LiDARs that report calibrated reflectance). This preserves dynamic range for small
  values. Linear normalization (`raw / constant`) is appropriate only when the sensor's maximum
  intensity is well-defined and documented (e.g. 255 for uint8 returns). Dividing by an
  arbitrary large constant (e.g. 65535) when typical raw values are in single digits destroys
  dynamic range and produces near-zero intensities across the entire dataset.

### Per-point timestamps

- Each point has a `timestamp_us` that must fall within the frame's `[start, end]` bounds.
- Per-ray ts MUST come from a **global column-to-time mapping**:
  `t_ray = frame_start_ts + (col_idx / n_columns) * sweep_duration_us`. Do **not**
  normalize by per-frame min/max columns -- many raw arrays carry overlap columns
  from adjacent sweeps that inflate the denominator and mismap rays by up to one
  sweep. Filter to the primary `n_columns` of the current sweep (read from the
  configured `RowOffsetStructuredSpinningLidarModelParameters.n_columns`, never
  hardcoded) **before** deriving per-ray ts and `model_element` column indices.
- Source per-point timestamps carry one of two epochs: **absolute**, or
  **frame-relative** -- signed offsets from the frame timestamp, or a count restarting
  at zero each frame. Establish which before using them across frames. A fit _within_
  one frame stays well-posed under either epoch; one that spans frames does not, because
  frame-relative values retrace the same span every frame and make the fitted rotor rate
  arbitrary.
- **Derive `spinning_frequency_hz` from the frame window**, never from a fit over
  per-point timestamps. For full-rotation frames the window _is_ one revolution, so the
  rate is `1 / (frame_end - frame_start)`; `warn_lidar_spin_frequency_mismatch` flags a
  declared value more than 10% away from it. Do not substitute the frame cadence either:
  cadence equals the rotor rate only when every revolution is stored, and a dataset that
  stores every other revolution publishes frames at half the rotor rate.

### LiDAR frame-window clamping

> **Sequence-interval rule.** Choose a sequence interval that fully contains
> every retained LiDAR frame, or drop LiDAR frames whose nominal
> `[sweep_start, sweep_end]` is not fully contained in the sequence interval.
> NCore writers assert that both `frame_timestamps_us` entries are contained
> in the sequence interval and that every stored per-ray `timestamp_us` falls
> inside the stored frame window; see `BaseSensorComponentWriter._store_base_frame`
> and `LidarSensorComponent.Writer.store_frame` in
> [`components.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/v4/components.py).

The NCore V4 writer asserts per-ray `timestamp_us` falls inside the frame's
`[frame_timestamps_us[0], frame_timestamps_us[1]]` window. When per-ray timestamps are
derived from `frame_start_us + col_time_offset_us[c]` (with per-column rounding and,
for many datasets, overlap rays that extend past the nominal column count), individual
rays can land a few microseconds outside the nominal sweep window. Filter those rays
or drop the frame; do not widen or clamp a frame window in a way that violates the
sequence-interval rule above.

**Retained-frame timestamp recipe** (apply this to every per-sweep frame-window
derivation):

```python
# interval is the chosen sequence_timestamp_interval_us (HalfClosedInterval).

# 1. Nominal frame window from the sensor's frame-level timestamp + sweep duration.
frame_start = int(fr_ts)
frame_end = int(fr_ts + sweep_duration_us - 1)

# 2. Keep only frames whose frame_timestamps_us can satisfy NCore's
#    sequence-interval assertion.
#    HalfClosedInterval is [start, stop), so interval.stop - 1 is the
#    largest valid frame_end.
if frame_start < interval.start or frame_end >= interval.stop:
    continue

# 3. Keep only rays whose timestamp_us can satisfy NCore's frame-window assertion.
valid_ray = (per_ray_ts >= frame_start) & (per_ray_ts <= frame_end)
if not np.any(valid_ray):
    continue
per_ray_ts = per_ray_ts[valid_ray]
# Apply the same mask to direction, distance, intensity, model_element, and any
# other per-ray arrays before calling LidarSensorComponent.Writer.store_frame.
```

Do not clamp boundary frame timestamps after `sequence_frame_range_decision` clips the
sequence interval (see ["Sequence frame-range intersection"](#sequence-frame-range-intersection)).
Re-apply the containment check above and drop any frame whose nominal
`frame_timestamps_us` no longer fits.

## Cameras

### Frame timestamps

The NCore V4 format stores `frames_timestamps_us: [N, 2] uint64 (start, end per frame)`
([NCore Data Formats](https://nvidia.github.io/ncore/data/formats.html)). Each camera frame has
its own **exposure start** and **exposure end** timestamp -- not the global frame timestamp.

Two decisions must be made for each camera, in this order:

1. **Shutter type** (rolling vs global; for rolling, also the direction).
2. **T_rs value** -- the rolling-shutter readout traversal time -- when shutter type is rolling.

Decision (1) is determined by the dataset's own documentation (`data.description`, any
`data.docs` entry, or a vendor datasheet named in either). When the documentation states or
implies the shutter is rolling -- with or without naming a specific `T_rs` value -- the
converter MUST use one of NCore's four directional rolling variants:
`ROLLING_TOP_TO_BOTTOM`, `ROLLING_BOTTOM_TO_TOP`, `ROLLING_LEFT_TO_RIGHT`, or
`ROLLING_RIGHT_TO_LEFT`. `ShutterType.GLOBAL` is the correct choice **only** when (a) the
documentation is silent on shutter type AND (b) no source provides `T_rs` (see the "T_rs not
published" branch below). Decision (1) is independent of `T_rs` source quality:
even when the only available `T_rs` estimate comes from a non-canonical source, the documented
shutter type stands. Downgrading to `ShutterType.GLOBAL` because
the operator distrusts the T_rs source's authority is a **silent contract violation**: it
discards documented sensor behavior and produces uniform-row sampling that contradicts the
dataset's own description. T_rs precision is a separate, lower-stakes decision governed by
the bullets below.

- **Rolling-shutter cameras**: start = trigger time + half shutter duration; end = readout done
  time - half shutter duration. These are exposure-centered bounds. The NuRec data quality guide
  requires _"accurate per-sensor start-of-frame and end-of-frame timestamps computed from
  per-imager rolling-shutter offset and delay timings."_
- **Global-shutter cameras**: start == end (instantaneous capture) is valid. A 1 µs delta is
  also acceptable.
- **Cameras with no documented shutter type AND no T_rs in any consulted source**: use
  `ShutterType.GLOBAL` with `start == end` (set both to the frame's readout-done timestamp).
  This is the only branch that selects `ShutterType.GLOBAL`, and it applies **only** when
  (a) the documentation is silent on shutter type **and** (b) no source provides T_rs. If the
  documentation states or implies a rolling shutter, keep the documented directional rolling
  variant per Decision (1) even when T_rs is unpublished — do **not** downgrade it to global;
  if any source provides T_rs, route to the next bullet instead. Do **not** guess T_rs — a
  wrong value introduces row-dependent position errors that are inconsistent across multiple
  cameras and cause multi-camera blur on dynamic objects. A global-shutter assumption is
  uniformly wrong across all cameras, which is far less harmful to NuRec's joint optimization
  than a wrong T_rs that biases each camera differently. Record the approximation in the
  critical-parameter decisions under `shutter_type_and_direction` with a note that the shutter
  type was undocumented, T_rs was not published in any consulted source, and global shutter
  was used as a fallback.
- **Rolling-shutter cameras where T_rs is provided in `data.description`**: prefer any
  operator-supplied estimate (from dataset documentation, a companion paper, an internal
  measurement, or any other stated source) over `ShutterType.GLOBAL`. A uniform
  approximation stated by the operator is consistently applied across all cameras, which
  is far less harmful to NuRec's joint optimization than a global-shutter assumption that
  ignores row displacement entirely. Use the stated shutter type and direction; derive
  `start` and `end` from the estimate per the rolling-shutter formula above. Record the
  source of the estimate under `shutter_type_and_direction` evidence.
- Each camera's **end timestamp must be unique** within that camera's writer -- no two frames
  from the same camera can share the same end timestamp.
- Using the global frame timestamp (e.g. `frame.timestamp_micros`) for all cameras is wrong.
  Each camera has its own trigger and readout timing. Getting this wrong means NuRec handles
  rolling-shutter compensation at incorrect times, causing motion blur or ghosting.

### Rolling-shutter direction enum mapping

Source datasets define their own rolling-shutter direction enums with integer values that do
**not** necessarily match NCore's `ShutterType` enum. The converter must map each source enum
value to the correct `ShutterType` variant **by name**, not by casting the integer.

NCore's `ShutterType` (declared via `auto()` starting at 1):

| `ShutterType` variant   | Integer |
| ----------------------- | ------- |
| `ROLLING_TOP_TO_BOTTOM` | 1       |
| `ROLLING_LEFT_TO_RIGHT` | 2       |
| `ROLLING_BOTTOM_TO_TOP` | 3       |
| `ROLLING_RIGHT_TO_LEFT` | 4       |
| `GLOBAL`                | 5       |

An off-by-one mapping (e.g. source enum `TOP_TO_BOTTOM=1` cast directly as `ShutterType(1)`
happens to work for some datasets but `GLOBAL_SHUTTER=5` cast as `ShutterType(5)` is also
correct by coincidence -- other datasets may not align). Always build an explicit name-based
mapping dictionary. An incorrect mapping silently marks rolling-shutter cameras as global,
disabling NuRec's per-scanline pose interpolation entirely: `compute_relative_frame_time`
returns 0 for every pixel, so each ray is rendered at the frame-start pose instead of its
row-dependent pose.

### Image format

- JPEG or PNG bytes. Store as `bytes` via `camera_writer.store_frame(image_data=...)`.
- `shutter_type`: `ShutterType.GLOBAL` or one of the `ROLLING_*` directional variants
  (`ROLLING_TOP_TO_BOTTOM`, `ROLLING_BOTTOM_TO_TOP`, `ROLLING_LEFT_TO_RIGHT`,
  `ROLLING_RIGHT_TO_LEFT`). Affects NuRec motion blur handling.

### Intrinsics

- One of: `OpenCVPinholeCameraModelParameters`, `OpenCVFisheyeCameraModelParameters`, `FThetaCameraModelParameters`.
- Must include `width` and `height` matching the image dimensions.
- For the full list of supported models and their mathematical specifications, see [NCore sensor models](https://nvidia.github.io/ncore/data/sensor_models.html).

### Camera model selection

The model class must match the **mathematical form of the dataset's distortion polynomial**, not
the camera's nominal field of view. The three NCore classes correspond to three mutually
incompatible polynomials (see [NCore sensor models](https://nvidia.github.io/ncore/data/sensor_models.html)
for the canonical equations):

| NCore class                          | Polynomial (per [NCore sensor models](https://nvidia.github.io/ncore/data/sensor_models.html))                                                                                                                                                                                                                                                                                                                                                                                                 | Coefficient meaning                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OpenCVPinholeCameraModelParameters` | Brown-Conrady, **rational radial form**: starting from a camera-frame ray `[x, y, z]`, normalize as `x' = x/z, y' = y/z, r^2 = x'^2 + y'^2`; then `d_r = (1 + k1*r^2 + k2*r^4 + k3*r^6) / (1 + k4*r^2 + k5*r^4 + k6*r^6)`; pixel coords are `[fx*(x'*d_r + dx_t + dx_p) + cx, fy*(y'*d_r + dy_t + dy_p) + cy]` where `dx/dy_t` are tangential and `dx/dy_p` are thin-prism. **k4, k5, k6 are denominator coefficients, not numerator extensions.**                                             | `radial_coeffs` are k1..k6 (3 numerator + 3 denominator), `tangential_coeffs` are p1..p2, `thin_prism_coeffs` are s1..s4                                                                                                                                                                                                                                                                                                                |
| `OpenCVFisheyeCameraModelParameters` | Kannala-Brandt: from a camera-frame ray `[x, y, z]`, take `theta = arctan2(sqrt(x^2 + y^2), z)` and per-axis normalized direction `x_n = x / sqrt(x^2 + y^2), y_n = y / sqrt(x^2 + y^2)`; the distorted angle is `delta(theta) = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)`; pixel coords are `[fx*delta(theta)*x_n + cx, fy*delta(theta)*y_n + cy]`. The radial polynomial is in **incidence angle theta from the optical axis**, not in pixel/normalized image radius. | `radial_coeffs` are k1..k4 (theta-domain), no p1/p2, `max_angle` is the half-angle from the optical axis to the farthest image corner — **always compute it from the calibration using `OpenCVFisheyeCameraModelParameters.compute_max_angle()` (`ncore.data`), never from the sensor spec's declared FOV** (the spec is a slot label, not the optical half-angle; cameras in the "same" slot can differ in focal length by 4× or more) |
| `FThetaCameraModelParameters`        | Forward / backward polynomial pair between incidence angle and pixel-distance from principal point. The forward map (angle -> pixel-distance) is `angle_to_pixeldist_poly`; its inverse (pixel-distance -> angle) is `pixeldist_to_angle_poly`. `reference_poly` names which of the two is the calibration's exact polynomial; the other is an approximate inverse computed during fitting.                                                                                                    | `reference_poly` (`PolynomialType.ANGLE_TO_PIXELDIST` or `PolynomialType.PIXELDIST_TO_ANGLE`), `pixeldist_to_angle_poly` (6 coeffs), `angle_to_pixeldist_poly` (6 coeffs), `max_angle` — read from vendor calibration data (see note below)                                                                                                                                                                                             |

For OpenCV fisheye parameters, `OpenCVFisheyeCameraModelParameters` lives in
`ncore.data`. Use keyword arguments when computing `max_angle`; see the
public NCore
[`OpenCVFisheyeCameraModelParameters.compute_max_angle()` implementation](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/types.py)
for the authoritative signature:

```python
max_angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
    resolution=resolution,
    focal_length=focal_length,
    principal_point=principal_point,
    radial_coeffs=radial_coeffs,
)
```

**`max_angle` for `FThetaCameraModelParameters` must be read from the vendor calibration data, not computed from the image boundary.**
FTheta calibration tools (Scaramuzza's OcamCalib, MIPI, and similar) always export `max_angle` as a direct calibration output — it is the angle at which the polynomial fit was performed. Read it from the calibration JSON alongside the polynomial coefficients via `FThetaCameraModelParameters.from_dict(model_parameters)`. Do not substitute a value derived from image corners: the polynomial's valid domain is what matters, and that is what the calibration tool already reports.

**FOV is not a sufficient signal.** The k1..k4 coefficients in a 120 deg "narrow" camera's
calibration are not necessarily Brown-Conrady -- some platforms ship the same Kannala-Brandt
theta-domain polynomial across their entire camera array regardless of FOV. Mis-classifying
the polynomial undistorts incorrectly and produces severe edge warping, so the choice must
be evidence-based, not FOV-heuristic.

When the dataset's documentation names the polynomial form (e.g. "k1..k4 are Kannala-Brandt
theta-domain coefficients" in `data.description` or `data.docs`), prefer that as the primary
evidence. `camera_model_family` in `critical_parameter_justifications` records the chosen
NCore class per camera id and cites the concrete evidence -- documentation, vendor
calibration file, or the empirical disambiguation below -- so a future agent run cannot
silently pick the wrong class on the same data.

When the polynomial form is genuinely ambiguous from the dataset's own documentation,
disambiguate empirically against the data itself: project a static landmark (LiDAR returns on a
known surface, a 3D cuboid corner, or a known fiducial) into the image with each candidate
model, and compare to the underlying pixels. The model whose projections land on the correct
pixels is the right one; the wrong model produces visible drift that grows with image radius
(small near the principal point, large near the edges).

#### Projection sample construction

**In-converter camera projection verification with `verify_camera_model_projection()`.**
Once you have constructed any NCore camera model parameters, compare them against
source-derived image-point references before writing to the shard. The template helper
`verify_camera_model_projection()` instantiates `CameraModel.from_parameters(params, device="cpu")`
and calls `camera_rays_to_image_points()`. A wrong model family, coefficient ordering, or FOV
boundary then fails during conversion rather than after a multi-hour NuRec run.

The expected image points must come from the source dataset's camera math, not from the NCore
parameters you just built. Express them in NCore's image-point convention (continuous
coordinates; the center of integer pixel `(i, j)` is `(i + 0.5, j + 0.5)`) so you do not compare
pixel-corner indices against NCore image points and pick up a constant 0.5 px bias. Include
center-to-corner samples so errors that grow with image radius are visible. Bound synthetic
sample radii to the nearest image edge, for example `min(cx, cy, W - cx, H - cy)`, rather
than a fixed incidence angle; off-center principal points and telephoto cameras can otherwise
produce valid camera rays whose projected image points fall outside the frame.

For `OpenCVFisheyeCameraModelParameters`, convert that pixel bound into an incidence angle
rather than sampling out to `max_angle`, which is the angle to the farthest image _corner_ and
so puts probes outside the frame along the short axis. NCore's
`compute_max_angle_with_monotonicity(fw_poly_coeffs, max_radius)` in
[`util.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/util.py) inverts the
forward Kannala-Brandt polynomial (coefficients `[0, 1, 0, k1, 0, k2, 0, k3, 0, k4]`) and
returns the smaller of the requested-radius angle and the angle at which the polynomial stops
increasing. Pass the focal-normalized inscribed radius,
`min(cx / fx, cy / fy, (W - cx) / fx, (H - cy) / fy)`. That per-axis normalization is the same
one `OpenCVFisheyeCameraModelParameters.compute_max_angle()` applies before calling the helper
with the corner radius, so unequal `fx` / `fy` are handled rather than approximated. The
helper is an implementation-level API rather than part of `ncore.data`; the public
`compute_max_angle()` cannot substitute because it hardcodes the corner radius.

Keep this check active. If calibration is shared across many sequences, run it once when the
converter loads or constructs calibration, cache the verified camera parameters, and reuse them
for each sequence.

```python
import numpy as np
from ncore_example.utils import verify_camera_model_projection

# params: any NCore camera model parameters instance you just constructed.
# cam_rays: center-to-corner camera-frame rays.
# expected_image_points: source-derived image points for the same rays
#   (NCore convention: pixel center = index + 0.5).
verify_camera_model_projection(params, cam_rays, expected_image_points, tol_px=1.0)
```

A correct model projects the source-derived samples to their expected image-point locations.
Symptoms of wrong parameters: invalid edge samples, rays projecting to extreme corners when
they should be near-center, or a systematic drift that grows with radius (wrong model class or
wrong coefficients).

## Masks

- Per-camera dictionary of mask name -> PIL Image.
- The NuRec data quality guide **requires** binary ego-vehicle masks per camera for best
  reconstruction quality. Without them, NuRec may include ego-vehicle pixels (e.g. hood,
  bumper, roof rack) in the reconstruction, degrading quality near camera boundaries.
- Empty dict `{}` is a fallback that won't break NuRec, but should only be used when
  the dataset provides no masks and they cannot be generated.
- Common masks: `"ego"`, `"sky"`, `"static_structure"`. The ego-vehicle mask **must**
  be stored under the key `"ego"` — NuRec looks up the ego mask by this exact key
  name. Any other key name (e.g. `"ego_vehicle"`) will not be recognized and will
  cause silent fallback to model-computed ego masks.
- **Polarity (positive convention).** A named mask uses _positive_ polarity: pixel
  values `1` (or `255` for `uint8`) mean the pixel **belongs to the named class**
  (e.g. `ego = 1` is a pixel of the ego vehicle); `0` means it does not.
  Some upstream perception tooling stores the _inverse_ (`1` = "valid pixel for
  loss", i.e. the complement of `ego`); invert before write. NCore itself
  does not enforce mask polarity, but NuRec's published `ncore-aux-data --ego-mask`
  output and the Cosmos-shard `egomask` write path both follow positive polarity
  (`True` belongs to the named class, written as `255` for `uint8` masks).

### Mask encoding domain

A logically-binary source mask in `mode='L'` may be stored on disk as `{0, 1}` (binary
integer-valued -- the producer skipped the scale-up to 255), `{0, 255}` (paletted binary),
or `{0..255}` (anti-aliased / soft, with intermediate values along object boundaries). All
three carry the same filename and PIL mode; the encoding only shows up in `np.unique(arr)`.

A polarity-inversion threshold hard-coded at `< 128` works on `{0, 255}` and `{0..255}` but
silently degenerates on `{0, 1}`: both values fall below the cutoff so every pixel is
selected. Threshold relative to the file's own min and max instead:

```python
mid = (int(arr.min()) + int(arr.max())) // 2
mask = (arr <= mid).astype(np.uint8) * 255   # picks the darker of the two values
```

This selects exactly the `0` pixels for `{0, 1}` (`mid = 0`), exactly the `0` pixels for
`{0, 255}` (`mid = 127`), and the natural midpoint between the two clusters for
`{0..255}`. Genuinely-soft masks (probability maps, distance fields) are not yes/no
labelings and need per-dataset quantization, not midpoint thresholding.

`camera_static_mask_handling` in `critical_parameter_justifications` records the encoding
domain (verbatim `np.unique(arr)` on a representative file) and polarity per
`(camera_id, mask_name)` pair the converter writes; this measurement, not an assumption,
is what the converter must use to choose its threshold. When the operator already knows
the encoding domain or polarity, naming it in `data.description` removes an exploration
round-trip but is not required.

## Cuboids (annotations)

See NCore data-formats spec for the `CuboidTrackObservation` schema. Converter-relevant
constraints:

### Timestamp constraints

Both `timestamp_us` and `reference_frame_timestamp_us` must lie inside the sequence interval
(`store_observations` asserts this). `PoseGraphInterpolator` handles arbitrary timestamps
via SLERP/linear, so exact matches to pose waypoints are **not** required. Use the
annotation-native timestamp directly.

That tolerance is NCore's, not NuRec's. Placing a cuboid evaluates its
`reference_frame_id`→`world` pose at `reference_frame_timestamp_us`. From `rig` and from
any sensor frame that path crosses the dynamic `rig`→`world` edge, so
`check_cuboid_pose_gap` hard-fails those when the nearest waypoint is more than 1 ms away;
a sensor frame's static sensor→rig pose prefixes the traversal rather than replacing it.
Give every such reference timestamp a waypoint -- the source's own ego pose where it
publishes one, otherwise a `PoseInterpolator` sample of the trajectory. Cuboids anchored in
`world` or `world_global` are exempt, reaching `world` by an identity or a static edge with
no pose evaluated at the reference timestamp.

Additionally, `timestamp_us` should fall inside some sensor frame's **inclusive**
`[frame_start, frame_end]`. Downstream consumers filter on the half-closed
`[frame_start, frame_end + 1)`, silently dropping `timestamp_us == frame_end + 1`
(a common off-by-one when anchoring to a sweep-end pose timestamp). Pick
`frame_start`, the midpoint, or `frame_end`. Not caught by the shard validator
(`//skills/nurec-onboard/ncore_validation:validate`); verify world-space cuboid visibility in
`ncore_vis` after conversion.

### Tracks lost to interval clipping

Because every observation timestamp must lie inside the sequence interval, clipping the
interval drops observations, and a track whose observations all fall outside vanishes with
no error from either the writer or the shard validator -- both see only what was stored.

Stationary objects are the usual casualty. Annotation formats often encode a parked car as
two samples at the ends of its visibility span rather than one per frame, so an interval
that begins after the first sample and ends before the second removes the track entirely,
even though the object is present in every retained frame. What reconstructs is a car baked
into the static scene, and it cannot be edited or removed downstream.

**Store source observations, plus crossed boundaries.** Keep each source observation that
falls inside the stored sensor coverage at its native timestamp, and add one interpolated
observation at each coverage boundary the track's source span crosses, however many native
observations remain inside. NuRec places a track only in frames within its own first-to-last
timestamp span, and at most extends that span by one step of the adjacent sample spacing, so
on an irregular cadence the retained samples alone can stop short of a boundary the object is
still visible at. Add nothing else: never resample onto a per-frame grid. Source timestamps
that repeat within one track are a source defect to resolve, not to collapse silently.
`cuboid_track_sample_timestamps` in the template `utils.py` implements this selection and
raises on repeated timestamps. `check_cuboid_track_timestamps` then rejects a track whose
stored timestamps differ from that selection; call it on every track before storing:

```python
coverage_lo = min(frame_starts.min() for frame_starts in stored_sensor_frame_starts)
coverage_hi = max(frame_ends.max() for frame_ends in stored_sensor_frame_ends)

sample_us = cuboid_track_sample_timestamps(native_us, coverage_lo, coverage_hi)
# ... build this track's observations at sample_us ...
check_cuboid_track_timestamps(track_id, stored_us, native_us, coverage_lo, coverage_hi)
```

Evaluate a boundary pose by interpolating between the bracketing source samples rather than
carrying an outside sample's pose across — for a stationary object the two are identical, but
for one moving at 10 m/s a 50 ms offset is half a meter of error. The parked car above then
stores exactly two observations and its span brackets every retained frame.

Use stored sensor coverage rather than raw sequence-interval endpoints, which
can sit outside every retained frame on a staggered rig.

A track that still ends up with fewer than two observations -- a single source sample that
crosses no boundary -- is valid NCore data, but NuRec needs two distinct timestamps to build a
trajectory. It is omitted, baking the object into the static scene, when its label source is
selected alone; another selected source contributing a distinct timestamp for the same
`track_id` can keep it. `warn_cuboid_tracks_below_two_observations` reports these per
component and label source.

NuRec merges observations by `track_id` once label sources are filtered and silently keeps
only one observation at a repeated timestamp. Two severities follow:

- Within one exact label source and version, a repeated `(track_id, timestamp_us)` is always
  destructive; `check_cuboid_duplicate_observation_timestamps` fails on it.
- Across different sources, or different versions of one source, the collision matters only
  when those sources are selected together -- an unversioned selector admits every version.
  `warn_cuboid_cross_source_timestamp_collisions` warns on it.

Do not resolve either by silently deduplicating source observations. Distinct timestamps
within one sensor frame are valid NCore data.

Record these under `sequence_frame_range_decision`, counting a track as one cuboid
component, track ID, label source, and source version, so alternative annotations of one
object are counted separately:

- `source_track_count_in_coverage` -- source tracks whose first-to-last timestamp span
  intersects `[coverage_lo, coverage_hi]`, which are exactly the tracks the selection above
  leaves non-empty. Scope it to the coverage, not to the whole source log: a short clip out
  of a long log legitimately contains a fraction of the log's tracks, and counting all of
  them turns every run into an apparent mass loss that hides the real one.
- `stored_track_count` -- tracks present in the shard's cuboid components.
- `excluded_tracks` -- source tracks in coverage the converter deliberately did not store,
  as a mapping from reason to count (`{}` when none), such as a class out of scope.
- `source_observation_count_in_coverage` -- source observations whose timestamp falls inside
  `[coverage_lo, coverage_hi]`, summed over the stored tracks.
- `stored_observation_count` -- observations present in the shard's cuboid components.

Every source track in coverage must be stored or excluded with a reason:
`source_track_count_in_coverage == stored_track_count + sum(excluded_tracks)`; an
unaccounted track is a clipping loss. Boundary observations are the only additions the rule
above permits, at most two per track, so
`stored_observation_count <= source_observation_count_in_coverage + 2 * stored_track_count`.
The track counts alone cannot catch the opposite failure — a converter that multiplies
observations keeps every track and looks healthy on them.

`check_cuboid_track_timestamps` is the per-track guarantee; record how many stored tracks it
checked. The build validator checks the recorded counts and the bound against the store, and
compares a representative sample of source tracks exactly against their stored timestamps
(see [`critical-parameters.md`](critical-parameters.md#entry-schema)). An unsampled track
that multiplies its observations within the aggregate bound is caught only by the converter
check.

### Centroid convention

`BBox3.centroid` must be the **geometric center** of the bounding box. The NuRec data
quality guide requires _"Object poses should represent the frame located at the center of the
bounding aligned with the boxes principal directions, without any offsets."_

Common source-dataset deviations to fix before storing:

- **Bottom-center**: add `dim_z / 2` to `centroid_z`.
- **Corner origin**: add half each dimension along the appropriate axis.

**Empirical verification** (required when documentation is ambiguous): for vehicle-class
cuboids, compute `mean(position.z) - mean(height)/2`. In a ground-level world frame this
approximates the average vehicle bottom. Near 0 → geometric center (no adjustment). Near
`mean(height)/2` above ground → bottom-center (adjustment needed). Cross-check with ≥ 2
object classes. Record in `critical_parameter_justifications` under
`cuboid_centroid_convention`.

### Reference frame consistency

`CuboidTrackObservation.reference_frame_id` must name the **exact** frame the stored
`centroid` coordinates live in. NCore's data contract enforces this: `BBox3.centroid` is
defined as living "in the frame of reference"
(`BBox3.centroid` in
[`types.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/types.py)),
and `CuboidTrackObservation.transform()` calls
`pose_graph.evaluate_poses(reference_frame_id, anchor_frame_id, …)`
(`CuboidTrackObservation.transform` in
[`types.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/types.py))
— the frame ID **must** be a registered node in the pose graph or the transform fails
(`PoseGraphInterpolator.evaluate_poses` in
[`transformations.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/common/transformations.py)
raises on unknown nodes or missing paths).

If the converter rebases the world (Z shift, re-anchor at frame 0, axis swap, merge of
multiple sensor frames into a single `world_global`), cuboid coordinates must either be
transformed into the new frame _or_ labeled with the original frame (which must itself be a
registered node). Storing cuboids under `reference_frame_id="world_global"` while their
coordinates are still authored in `world` (or vice versa) produces two visible symptoms in
`ncore_vis`:

- **Vertical mismatch**: cuboids appear consistently above or below their LiDAR returns by
  a fixed amount across the sequence. The named `reference_frame_id` differs in Z anchor
  from the frame the centroid coordinates were authored in.
- **Horizontal-axis flip**: cuboids appear rotated relative to ego's forward trajectory
  while individual cuboid yaws look correct in isolation. The named `reference_frame_id`
  is registered in the pose graph but its X/Y axes don't agree with the rig's actual
  direction of travel (typically because `world_global` was re-anchored at a frame whose
  rig→world rotation differs from the dynamic rig→world used elsewhere in the pipeline).

NCore source is agnostic about which `dim` axis is length / width / height and about the
`rot=(0,0,0)` "forward" direction
(`BBox3.dim` and `BBox3.rot` in
[`types.py`](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/types.py)
only says `dim` is "extents in the local frame" and `rot` is "XYZ Euler angles orienting
the local bounding-box frame to the frame of reference"). The forward-axis flip and Z
mismatch are reference-frame issues, not local-frame convention bugs — fix them by
transforming the centroid into the named frame, or by re-labeling the named frame, not by
swapping `dim` axes or reinterpreting `rot[2]`.

### Track the world frame at every step

Companion to "Track the rig frame at every step" above, applied to world frames. NCore
treats names such as `world`, `world_global`, and any converter-authored intermediate
world frames as ordinary pose-graph nodes; the converter defines their origins and axes by
the static/dynamic poses it registers. The bug shape these labels invite is two world
frames with the same name but different origins / axes — one defined where the centroid
was authored, the other registered in the pose graph under the same name.

Recommended pattern: every variable holding a 3D position or a `T_*_world*` matrix should
carry a one-line comment naming the frame and what its origin / axes are. For example:

```python
# in source pose frame: origin = sensor mount, axes match the source's native
# convention (cross-reference data.docs); NOT yet shifted by mount height.
T_world_lidar_anchored = np.linalg.inv(lidar_world_pose) @ ...

# in `world_global`: T_world_world_global was built by re-anchoring at unique_poses[0]
# (frame 0) AFTER shifting Z by sensor_mount_height_m, so this frame's Z origin is on the
# ground. Cuboids stored here must be in the same frame; cuboids still in
# source pose-frame coordinates appear shifted vertically by the mount height.
T_rig_world_global = T_world_world_global @ T_rig_world
```

Cuboid coordinates and the world frame they are stored under (`reference_frame_id`) must
agree on origin, orientation, **and** Z anchor. All three discrepancies surface in
`ncore_vis` per "Reference frame consistency" above — a constant Z offset, an axis-flipped
forward direction, or both.

### LabelSource

`LabelSource` tags the **origin pipeline** of the labels, not their quality. NCore's
`EXTERNAL` docstring ("from an unspecified external source, e.g., from third-party
processes") is the controlling definition; third-party dataset labels map to `EXTERNAL` even
when they were human-annotated upstream.

| Origin relative to this converter                 | `LabelSource`   |
| ------------------------------------------------- | --------------- |
| Authored by the same team writing the NCore shard | `GT_ANNOTATION` |
| Third-party / upstream dataset, any quality       | `EXTERNAL`      |
| Upstream is explicitly an autolabel system        | `AUTOLABEL`     |
| Synthetic simulator                               | `GT_SYNTHETIC`  |

Do **not** pick a different value to work around downstream filtering -- fix the consumer
(`run-nurec` `track_label_sources`). `source_version` is a free-form string; leave `None`
unless the source exposes a stable version identifier.

## Component group profiles

| Profile            | Behavior                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `default`          | All components in one group                                                                |
| `separate-sensors` | One group per sensor type (cameras, lidars, radars) plus shared (poses, intrinsics, masks) |
| `separate-all`     | One group per individual sensor                                                            |

`separate-sensors` is recommended for NuRec.

## Store types

| Type        | Description                                                      |
| ----------- | ---------------------------------------------------------------- |
| `itar`      | Indexed tar archive; fast sequential read, recommended for NuRec |
| `directory` | Plain directory; easier to debug                                 |

## Data quality requirements for NuRec

Full accuracy targets live in the NuRec "Ensure Data Quality" guide -- consult it for the
authoritative thresholds. A converter cannot improve on data it doesn't have, but it MUST
avoid silently degrading it:

- Preserve **original** sensor resolution and **full** temporal rate; do not downsample.
- Store **raw distorted** images with distortion modeled by the intrinsic camera params; do
  not undistort / rectify in the converter.
- Provide per-sensor `frame_timestamps_us = [exposure_start, exposure_end]` and
  rolling-shutter direction -- see "Frame timestamps" and
  "Rolling-shutter direction enum mapping".
- Rig trajectory must cover the full range of sensor frame timestamps (enforced by the
  NCore V4 writer via the interval / endpoint assertions).
- Sequence interval must be clipped to the intersection of all required-modality
  coverages -- pose, cuboids, and per-sensor frame timestamps. NuRec silently drops
  cuboid tracks at frames outside their `tracks_timestamps_us` range, so half-coverage
  edges quietly degrade reconstruction. See "Sequence frame-range intersection" above.
- Cuboid `timestamp_us` is the absolute observation time with no offsets; pass the dataset
  timestamp through unchanged.
- Provide per-camera ego-vehicle masks when available (see "Masks").

## NCore convention checklist

Pre-conversion sanity check before running the converter. Each item is enforced
by the NCore V4 writer or the shard validator at
`//skills/nurec-onboard/ncore_validation:validate`. The linked section is the single source
of truth for the specifics -- this gate only names what to re-check, so it cannot
drift from the rules it points at:

- **Frames** -- pose naming and camera / LiDAR frame conventions:
  [Coordinate frames](#coordinate-frames).
- **Timestamps** -- timestamp dtype and sequence-interval construction:
  [Timestamps](#timestamps).
- **Pose dtypes** -- required dtype per pose edge: [Pose dtypes](#pose-dtypes).
- **Pose trajectory** -- multi-source assembly and waypoint density:
  [Rig-pose pipeline](#rig-pose-pipeline-canonical-data-flow),
  [Pose trajectory density](#pose-trajectory-density).
- **Camera `frame_timestamps_us`** -- per-frame exposure-window contract:
  [Frame timestamps](#frame-timestamps).
- **LiDAR frames** -- frame contents and spinning-schedule monotonicity:
  [LiDAR data](#lidar-data).
- **Cuboids** -- reference frame and in-interval timestamps:
  [Cuboids (annotations)](#cuboids-annotations).
