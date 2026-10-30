<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Critical Parameter Justifications

Schema reference for the critical-parameter justification step in
[`SKILL.md`](SKILL.md) raw-data exploration / critical-parameter justification step. Enforced by `validators/build_ncore_converter_validator.py`
at Scaffold Mode's Justifications validator.

## Contents

- [Entry schema](#entry-schema)
- [Authoritative source sub-field](#authoritative-source-sub-field)
- [Evidence](#evidence)
- [Required parameters](#required-parameters)

## Entry schema

```yaml
- parameter: <string> # one of the names in the table below
  value: <any> # chosen value -- see "Value field" below
  evidence: <string> # see "Evidence" below
  wrong_value_consequence: <string> # one-sentence failure mode (the agent's own words)
```

### Value field

`value:` accepts any YAML type -- string, number, list, mapping -- so each
parameter can use the shape that best represents its chosen value. Use the
parameter's natural structural form rather than flattening it into a
block-literal YAML string; structured values let downstream tooling
introspect dimensions and types instead of re-parsing free text.

| Parameter shape      | Natural YAML form       | Example                                                                                                                |
| -------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Enum / string        | bare string             | `value: "cw"` (for `spinning_direction`)                                                                               |
| 4x4 transform        | list of 4 lists         | `value: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.81], [0.0, 0.0, 0.0, 1.0]]` (for `T_lidar_rig`) |
| Per-camera mapping   | dict keyed by camera id | `value: {front: [[...]], rear: [[...]]}` (for `T_camera_rig`)                                                          |
| List of records      | list of dicts           | `value: [{camera_id: front, model: opencv_pinhole}, ...]` (for `camera_model_family`)                                  |
| Per-sensor decisions | dict-of-lists           | `value: {cameras: [...], lidars: [...]}` (for `sensor_inventory_decisions`)                                            |

The validator (`_value_field_diagnostic` in
`validators/build_ncore_converter_validator.py`) rejects `null`, blank
strings, placeholder markers (`FILL IN`, `TODO`, `TBD`, `XXX`,
`[ ... ]`), and empty containers (`[]`, `{}`). Parameters listed in
`PARAMETERS_ALLOWING_EMPTY_VALUE` (same module) are the sole exception:
they may carry an empty list (`value: []`) when the dataset
legitimately has nothing to populate them; empty dicts remain
rejected even for those parameters. All other shapes pass the type
check; finer-grained per-parameter constraints (e.g. the
`T_camera_rig` per-camera coverage check) are enforced separately.

`T_lidar_rig` and `spinning_lidar_model_frame_choice` additionally require
a single-ray `worked_example`:

```yaml
- parameter: T_lidar_rig # or spinning_lidar_model_frame_choice
  # value, evidence, wrong_value_consequence: see "Entry schema" above.
  worked_example:
    row_index: <int>
    column_index: <int>
    # Unit vector in rig frame.
    d_rig_expected: [<float>, <float>, <float>]
    tolerance_deg: <float>
    # Optional; required when the store has more than one spinning LiDAR.
    lidar_id: <string>
```

The validator predicts `d_rig` from the stored intrinsics + `T_lidar_rig` and
fails if the residual exceeds `tolerance_deg`.

`lidar_id` pins the verification to a specific spinning LiDAR. Single-LiDAR
stores can omit it; stores with more than one spinning LiDAR fail the validator
without it (the validator refuses to pick arbitrarily, so multi-LiDAR
converters cannot silently verify against the wrong sensor).

`T_camera_rig` additionally requires a per-camera `worked_example` covering
**every** camera the converter writes a `T_camera_rig` for. The validator rejects
missing or extra camera ids (set-equality with `T_camera_rig.value.keys()`).
Each entry has a required rotation block and a translation block that is
conditionally mandatory (see below):

```yaml
- parameter: T_camera_rig
  # value, evidence, wrong_value_consequence: see "Entry schema" above.
  worked_example:
    per_camera:
      - camera_id: <string>
        # Required rotation block.
        # Any unit-norm camera-frame vector the operator can assert a
        # rig-frame counterpart for (NCore camera convention:
        # camera +Z = optical axis = [0, 0, 1]).
        axis_in_camera: [<float>, <float>, <float>]
        # Operator's rig-frame prediction (NCore rig convention:
        # rig +X forward, +Y left, +Z up).
        axis_in_rig_expected: [<float>, <float>, <float>]
        # Angular tolerance, e.g. 5.0.
        tolerance_deg: <float>
        # Translation block. Mandatory when
        # authoritative_source.verification.per_image_data_available is true
        # (see Authoritative source sub-field below); optional otherwise.
        # If declared, all three fields must be present.
        # Mount position in rig-frame meters.
        position_in_rig_expected: [<float>, <float>, <float>]
        # Translation tolerance, e.g. 0.1.
        position_tolerance_m: <float>
        # Free text describing how the prediction was obtained. By
        # convention the derivation cites a stream OTHER than
        # authoritative_source.used_by_converter so the cross-check is
        # independent (anti-circularity guard).
        position_in_rig_expected_derivation: <string>
```

The validator loads `T_camera_rig` for `camera_id` from the store, rotates
`axis_in_camera` through `R`, and asserts the angular distance to
`axis_in_rig_expected` is at most `tolerance_deg`. When the translation
block is declared in full, it additionally asserts
`||T.t - position_in_rig_expected|| <= position_tolerance_m`.

**Why the translation block is conditionally mandatory.** A passing
rotation check does not certify translation. Two failure modes hide
behind it: an empirical `T_camera_rig` with SVD-mean translation noise
(rotations match per-image data, translations are several decimetres
off the static source); and a static-extrinsic translation block that
lives in a frame other than the rig the converter built (which puts
the sensor at the wrong rig-relative position even when rotation is
correct). The two cases need predictions from **different sources**:
the first wants the static source as the prediction, the second wants a
per-image-derived position. The `position_in_rig_expected_derivation`
free-text records which is being tested and is the anti-circularity
guard — a converter that derives the prediction from the same source it
loaded `T_camera_rig` from can pass the check trivially without testing
either failure mode. The diagnosis-YAML reviewer reads the derivation
to confirm independence.

Partial declarations of the translation block (any subset of the three
fields) fail the validator; declare all three or none.

`camera_model_family` additionally requires a per-camera projection
`worked_example` covering every camera declared in `camera_model_family.value`.
Each entry supplies source-derived camera-frame rays and expected NCore
image-point coordinates. The source samples should include off-center or
near-edge rays where pinhole/fisheye/ftheta errors are largest; LiDAR-to-image
correspondence samples are a good source when available. When generating
synthetic samples from intrinsics, follow the sample-radius bound in
[`conventions.md`](conventions.md#projection-sample-construction).

```yaml
- parameter: camera_model_family
  # value, evidence, wrong_value_consequence: see "Entry schema" above.
  worked_example:
    per_camera:
      - camera_id: <string>
        model: opencv_pinhole | opencv_fisheye | ftheta
        # List of camera-frame rays, shape (N, 3).
        camera_rays:
          - [<float>, <float>, <float>]
        # Source-derived NCore image points, shape (N, 2).
        expected_image_points:
          - [<float>, <float>]
        tolerance_px: <float>
        # E.g. "LiDAR-to-image correspondences from source calibration QA".
        sample_source: <string>
```

The validator loads the stored NCore camera model parameters for each camera,
checks that the stored parameter class matches `model`, projects
`camera_rays` with `CameraModel.camera_rays_to_image_points`, and asserts the
maximum image-point error is at most `tolerance_px`. This catches silent
camera-model-family errors such as fitting a theta-domain fisheye calibration
as a Brown-Conrady pinhole model.

`sequence_frame_range_decision` additionally requires a `worked_example`
recording how many tracks passed `check_cuboid_track_timestamps` during
conversion and a representative sample of source tracks read directly from the
raw annotations. Sample at least one track whenever `stored_track_count` is
nonzero, and prefer tracks that cross a coverage boundary, such as a stationary
object stored as two endpoints, and densely annotated moving tracks.

```yaml
- parameter: sequence_frame_range_decision
  # value, evidence, wrong_value_consequence: see "Entry schema" above.
  worked_example:
    # Tracks passed through check_cuboid_track_timestamps.
    tracks_checked: <int>
    sampled_tracks:
      # Cuboid component name in the shard.
      - component: <string>
        # Track id as stored in the shard.
        track_id: <string>
        # Stored LabelSource name, e.g. EXTERNAL.
        label_source: <string>
        # Stored source version, or null.
        source_version: <string|null>
        # Every source timestamp of the track, unclipped.
        source_timestamps_us: [<int>]
        # E.g. "tracks/<file>.pkl, read with <reader>".
        sample_source: <string>
```

The validator requires `tracks_checked` to equal `stored_track_count` and every
source track in coverage to be stored or excluded. Against the store it checks
the recorded stored counts and the observation bound, and that each sampled
track stores exactly the selection described in
[`conventions.md`](conventions.md#tracks-lost-to-interval-clipping), computed
against the store's sensor coverage. A resampled, padded, repeated, or dropped
observation on a sampled track fails. Unsampled tracks are checked only in
aggregate, so the converter must run `check_cuboid_track_timestamps` on every
track.

## Authoritative source sub-field

`T_lidar_rig` and `T_camera_rig` additionally require an `authoritative_source`
sub-block recording the chain from the operator-declared source in the input
YAML to whatever the converter actually loaded, plus an independent
verification block:

```yaml
- parameter: T_camera_rig # or T_lidar_rig
  # value, evidence, wrong_value_consequence, worked_example: see above.
  authoritative_source:
    # The operator-declared authoritative source from data.description /
    # data.docs (verbatim quote with path, e.g. "data.description:
    # 'authoritative source for T_camera_rig is
    # static_extrinsic_calibration.yaml'"). When the input is silent, MUST
    # explicitly say "none -- no authoritative source declared".
    declared_in_input: <string>
    # What the converter actually loaded, including any composed basis
    # change. E.g. "static_extrinsic_calibration.yaml composed with
    # R_OPENCV_FROM_CAMYAML"; or "empirical SVD-mean of inv(T_rig_world) @
    # T_camera_world over per-image poses" if the converter went empirical.
    used_by_converter: <string>
    # true = converter used the declared source as-is (with any documented
    # basis change). false = converter ignored or replaced the declared
    # source.
    source_match: <bool>
    # Required sub-block. Operator-supplied authoritative-source claims are
    # HYPOTHESES; per-image pose data, when the dataset ships it, is
    # independently authoritative. This block records what cross-check the
    # converter ran rather than just asserting that the operator's claim
    # was followed.
    verification:
      # Does the dataset ship per-image (or per-frame) pose streams the
      # converter can derive an independent prediction from? Declared as a
      # fact about the data.
      per_image_data_available: <bool>
      # Free text describing the cross-check, with concrete numerical
      # residuals when per_image_data_available is true (e.g. "<sensor>
      # per-image pose at frame 0 transformed into rig via
      # inv(T_rig_world_0); residual <dx, dy, dz> m against <calib_file>
      # translation; <within|exceeds> calibration tolerance"). When false,
      # MUST explicitly state that (e.g. "no per-image pose data available
      # for this sensor; trusted static source as the only available
      # evidence") rather than leave it blank.
      summary: <string>
```

`source_match: false` is recorded for workflow review, not auto-failed.
`verification.summary` is required, non-placeholder text but is not parsed;
`per_image_data_available: true` makes the `T_camera_rig` translation block
mandatory. See [`conventions.md`](conventions.md#static-extrinsics-over-empirical-solving).

## Evidence

Each entry's `evidence` must be concrete -- one of:

- A `data.docs` quote with section / page anchor.
- An `explore_<data.name>_meta.json` key.
- An inline numeric measurement (residual, histogram bucket, offset, ...).

`T_lidar_rig` evidence must additionally include a numeric cuboid-overlap or
LiDAR-to-camera reprojection residual when the value is identity / near-identity
(identity is rarely correct -- a non-zero z is normal from sensor height alone).

Placeholders (`FILL IN`, `TODO`, `TBD`, `[ ... ]`, blank) fail the validator.

## Required parameters

| Parameter                           | Value domain                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Rationale (`conventions.md` anchor)                                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `spinning_direction`                | `"cw"` \| `"ccw"`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Spinning LiDAR model parameter pitfalls                                                                             |
| `row_elevations_rad`                | strictly descending per-beam angles                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Spinning LiDAR model parameter pitfalls; LiDAR model-parameter source hierarchy                                     |
| `column_azimuths_rad`               | per-column angles, includes any heading correction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Canonical azimuth-correction formula; Spinning LiDAR model parameter pitfalls                                       |
| `row_azimuth_offsets_rad`           | per-beam offsets; zeros if absent                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Spinning LiDAR model parameter pitfalls                                                                             |
| `spinning_lidar_model_frame_choice` | `"option_a_rig_aligned"` \| `"option_b_native_aligned"`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Spinning LiDAR model-frame choice                                                                                   |
| `sensor_inventory_decisions`        | dict-of-lists `{<modality_group>: [{path, include, decision_source, reason}, ...]}` keyed by modality (`cameras`, `lidars`, `radars`, `poses`, `annotations`, `other`); `decision_source` ∈ {`agent_decided`, `operator_include`, `operator_exclude`} -- one entry per raw subdirectory; populated from the raw-data exploration step                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | --                                                                                                                  |
| `sequence_frame_range_decision`     | `{start_us, stop_us, limited_by_start, limited_by_end, excluded_frame_count, source_track_count_in_coverage, stored_track_count, excluded_tracks, source_observation_count_in_coverage, stored_observation_count, rationale}` -- intersection of every required modality envelope. Source tracks in coverage equal stored plus excluded tracks. `stored_observation_count <= source_observation_count_in_coverage + 2 * stored_track_count`; requires a `worked_example` with `tracks_checked` and sampled tracks.                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Sequence frame-range intersection; Tracks lost to interval clipping                                                 |
| `lidar_model_source`                | per LiDAR `{lidar_id, rank, row_elevations_source, column_azimuths_source, row_azimuth_offsets_source, model_element_source}`; `rank` ∈ 1..2; synthesis is **not** allowed for `row_elevations_rad` or `model_element`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | LiDAR model-parameter source hierarchy                                                                              |
| `T_lidar_rig`                       | 4x4 SE(3); requires `worked_example`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Spinning LiDAR model-frame choice                                                                                   |
| `camera_frame_convention`           | `"ncore_native"` (OpenCV: +Z fwd, +X right, +Y down) \| other                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Camera sensor frame                                                                                                 |
| `T_camera_rig`                      | 4x4 SE(3) including any basis change above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Camera sensor frame                                                                                                 |
| `rig_frame_convention`              | NCore: +X fwd, +Y left, +Z up                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Coordinate frames                                                                                                   |
| `cuboid_dim_axis_ordering`          | mapping of dataset dim fields → `BBox3.dim[0..2]`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Cuboids (annotations)                                                                                               |
| `cuboid_centroid_convention`        | `"geometric_center"` \| `"bottom_center"` \| `"corner"`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Centroid convention                                                                                                 |
| `cuboid_world_anchored_frame`       | `"world"` \| `"world_global"` \| `"rig"`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | World frames; World re-referencing and world-anchored data                                                          |
| `timestamp_unit`                    | NCore requires microseconds (`uint64`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Timestamps                                                                                                          |
| `camera_timestamp_meaning`          | `trigger` \| `exposure_center` \| `exposure_end` \| `readout_end`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Frame timestamps                                                                                                    |
| `lidar_frame_timestamp_meaning`     | `sweep_start` \| `sweep_center` \| `sweep_end` \| `pose_ref`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Sweep timestamp convention; LiDAR frame timestamp sourcing                                                          |
| `shutter_type_and_direction`        | `GLOBAL` \| `ROLLING_TOP_TO_BOTTOM` \| `ROLLING_BOTTOM_TO_TOP` \| ...                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Rolling-shutter direction enum mapping                                                                              |
| `lidar_raw_format`                  | `sensor_frame_xyz` \| `world_frame_xyz` \| `ego_compensated_xyz` \| `polar_with_per_ray_metadata` \| `polar_without_per_ray_metadata`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | LiDAR direction derivation                                                                                          |
| `lidar_motion_compensation`         | uncompensated raw sensor-frame data (required)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Motion compensation                                                                                                 |
| `image_distortion_state`            | `distorted` (raw) \| `undistorted`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Image format; Intrinsics                                                                                            |
| `camera_model_family`               | per camera `{camera_id, model}`; `model` ∈ `opencv_pinhole` \| `opencv_fisheye` \| `ftheta` -- one entry per included camera id, naming the chosen NCore intrinsic class; requires a per-camera projection `worked_example`. The choice must follow the dataset's distortion polynomial form (Brown-Conrady → pinhole, Kannala-Brandt theta-domain → fisheye, fitted forward/backward polynomial pair → ftheta), **not** a FOV heuristic.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Camera model selection                                                                                              |
| `camera_static_mask_handling`       | list of `{camera_id, mask_name, source_path, pil_mode, encoding_domain, np_unique, polarity}` -- one entry per `(camera_id, mask_name)` pair the converter writes. `mask_name` is the NCore name (`ego`, `sky`, `static_structure`, ...). `encoding_domain` ∈ `binary_01` \| `binary_0_255` \| `antialiased_0_255` and **must** match the actual `np.unique(arr)` measured on a representative file (recorded verbatim in `np_unique`); a stated domain that disagrees with the measurement fails the validator. `polarity` ∈ `positive` (raw value 1/255 = belongs to named class) \| `inverse` (raw value 1/255 = does NOT belong to named class) -- inverse must be flipped before write to match NCore's positive convention. Empty list `[]` is permitted **only** when the dataset ships no per-camera mask files; in that case `evidence` must cite the probe enumeration and `wrong_value_consequence` must reference the NuRec quality impact of writing `{}`. | Masks; Mask encoding domain                                                                                         |
| `pose_trajectory_source`            | names of every source combined (per-image, frame-level, IMU/GPS, ...)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Pose trajectory density; Rig-pose pipeline                                                                          |
| `lidar_point_frame`                 | `sensor_frame_uncompensated` (required)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | LiDAR sensor frame; Motion compensation                                                                             |
| `intensity_normalization_method`    | `tanh` \| `linear` / `scale` \| `none` (raw already in [0,1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Intensity                                                                                                           |
| `nurec_lidar_selection`             | `{selected_id, excluded_ids, rationale, operator_override}` -- `selected_id` is the single LiDAR id passed to NuRec; `excluded_ids` is a flat list of LiDAR id strings (every other shard LiDAR), with per-unit reasons consolidated into `rationale` (one narrative covering both the selection and the exclusions); `operator_override` is `true` when the id came from the input                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | NuRec trains on exactly one LiDAR per run; an unchecked multi-LiDAR `dataset.lidar_ids` causes a training assertion |
