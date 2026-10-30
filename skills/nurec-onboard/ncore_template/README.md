<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore V4 converter template

Self-contained Bazel workspace for building a custom **NCore V4** (component-based zarr/itar) converter for use with **NuRec** (Neural Reconstructions). Materialized into each operator workdir by `run.sh`. The workdir ships with a CLI plus a placeholder converter package (`ncore_example/`) that the agent fills in per dataset; shard validation and metrics evaluation are provided as a platform service by NuRec Onboard and are not part of the materialized workdir.

**NCore dependency:** This workspace depends on [nvidia-ncore](https://pypi.org/project/nvidia-ncore/) (`pip install nvidia-ncore`). Bazel installs it automatically from PyPI via the pip requirements in [deps/pip/requirements_3_13.txt](deps/pip/requirements_3_13.txt). See the [NVIDIA/ncore](https://github.com/NVIDIA/ncore) GitHub repo for source code, docs, and tools.

## Prerequisites

- **Python 3.13**
- **Bazel** (Bzlmod; see [.bazelversion](.bazelversion))

## Build and run

```bash
bazel run //ncore_example:convert -- --output-dir=/tmp/ncore_out example
```

This writes valid V4 output (one sequence named `example_sequence_001` with poses, intrinsics, masks, camera frames for three camera models (pinhole, fisheye, ftheta), a LiDAR frame, a radar frame, cuboids, and sequence meta) to `/tmp/ncore_out` using synthetic data. The bundled `example` subcommand ignores `--root-dir` and `get_sequence_ids` returns a fixed id; real converters wired into the same CLI use `--root-dir` to discover sequences. Subcommand options include `--store-type {itar,directory}`, `--profile {default,separate-sensors,separate-all}`, and `--sequence-meta` / `--no-sequence-meta`.

**Time-range restriction** — use `--seek-sec` and `--duration-sec` to convert a sub-interval of the sequence (same seek/duration convention used by NuRec):

```bash
bazel run //ncore_example:convert -- \
    --root-dir /path/to/raw --output-dir /path/to/out \
    --seek-sec 5.0 --duration-sec 10.0 \
    example
```

`--seek-sec` skips the given number of seconds from the first pose timestamp; `--duration-sec` caps the total converted duration. Both are optional; when omitted the full sequence is converted. The restricted interval is applied inside `_compute_sequence_interval` via NCore's `time_bounds()` utility, and all `_write_*` methods filter their source frames to that interval.

---

## Adding an NCore V4 converter

This section explains how to add a new data-format converter so that your proprietary AV sensor data is written as **NCore V4** for **NuRec**. **Start from the skeleton** in [ncore_example/converter.py](ncore_example/converter.py): it shows the full contract (Config, `get_sequence_ids`, `from_config`, `convert_sequence`) and demonstrates all V4 component types (Poses, Intrinsics, Masks, Camera, LiDAR, Radar, Cuboids) with placeholder data. Each component type is written in a dedicated `_write_*` method for clarity. The same file also exposes the Click CLI that `bazel run //ncore_example:convert` invokes.

### NCore V4 quick reference

- **Component groups:** V4 stores are organized into component groups (e.g. one group per sensor or shared). Each group holds components (Poses, Intrinsics, Camera, Lidar, Radar, Cuboids, Masks, etc.).
- **Sequence metadata:** Every sequence has `sequence_id` and `sequence_timestamp_interval_us` (a half-closed time range in microseconds). This metadata is required for NuRec and ncore_vis.
- **Main components:**
  - **Poses** – Rig→world (dynamic), sensor→rig (static), optional world→world_global (static).
  - **Intrinsics** – Camera and LiDAR intrinsics.
  - **Camera / Lidar / Radar** – Sensor frames (images, point clouds, etc.).
  - **Cuboids** – 3D boxes (e.g. objects).
  - **Masks** – Optional in the NCore schema, but **required by NuRec**: register a `MasksComponent` and call `store_camera_masks` per camera (use `{}` if you have no mask images yet).

References: [NCore data formats](https://nvidia.github.io/ncore/data/formats.html), [data.v4 API](https://nvidia.github.io/ncore/apis/data.html), [conventions](https://nvidia.github.io/ncore/data/conventions.html).

### V4 output reference: what a shard looks like

When implementing your converter, use these as working references for the expected NCore V4 layout and content:

**1. NCore format documentation (structure of a V4 shard)**  
The canonical description of the V4 component-store layout is in the NCore docs:

- **[Data formats](https://nvidia.github.io/ncore/data/formats.html)** (including [formats.rst](https://github.com/NVIDIA/ncore/blob/v19.8.0/docs/data/formats.rst) in the NCore repo): component group hierarchy, root-level `sequence_meta_data` and `sequence_timestamp_interval_us`, and the layout of each component type (Poses, Intrinsics, Camera, Lidar, Radar, Cuboids, Masks). It also describes the **itar** (single-file) vs **directory** store options.

Use this to confirm directory structure, attribute names, and array shapes (e.g. poses as 4×4, timestamps in microseconds).

**2. Example V4 data**

- **Template output:** Running the skeleton converter produces minimal but valid V4 under your `--output-dir`. Inspect the resulting directory (or `.zarr.itar` file) and the `{sequence_id}.json` sequence meta to see a concrete example.
- **NCore test-data-v4:** The NCore repo uses a **test-data-v4** archive (fetched from GitHub Packages in `MODULE.bazel`) as real V4 data in its tests. If you have access to the NCore repo and GitHub Packages (see [NCore CONTRIBUTING](https://github.com/NVIDIA/ncore/blob/v19.8.0/CONTRIBUTING.md)), you can use that archive as another reference.
- **Reference converters:** The [Waymo](https://github.com/NVIDIA/ncore/blob/v19.8.0/tools/data_converter/waymo/converter.py), [PAI](https://github.com/NVIDIA/ncore/blob/v19.8.0/tools/data_converter/pai/converter.py), and [Colmap](https://github.com/NVIDIA/ncore/blob/v19.8.0/tools/data_converter/colmap/converter.py) converters in the NCore repo produce full V4 output (poses, cameras, lidar, etc.). Running one of them (when you have the source data) gives you a complete example shard to compare against.

**3. Code examples**

- **[components_test.py](https://github.com/NVIDIA/ncore/blob/v19.8.0/ncore/impl/data/v4/components_test.py):** Full example of writing V4 data and extending it with a custom component (`TestDataNewComponent`). Useful for seeing the exact writer calls and store layout in code.

### Step 1: Add a Config subclass

Copy and adapt from `ExampleConverterConfig` in [converter.py](ncore_example/converter.py). Extend `FileBasedDataConverterConfig` and add the same NCore-style options used by the existing NCore converters ([Waymo](https://github.com/NVIDIA/ncore/blob/v19.8.0/tools/data_converter/waymo/converter.py), [Colmap](https://github.com/NVIDIA/ncore/blob/v19.8.0/tools/data_converter/colmap/converter.py), [PAI](https://github.com/NVIDIA/ncore/blob/v19.8.0/tools/data_converter/pai/converter.py)):

- `store_type`: `"itar"` | `"directory"` (default `"itar"`).
- `component_group_profile`: `"default"` | `"separate-sensors"` | `"separate-all"` (default `"separate-sensors"`).
- `store_sequence_meta`: `bool` (default `True`) – whether to write `{sequence_id}.json` after conversion.

You inherit `root_dir`, `output_dir`, and sensor selection from `FileBasedDataConverterConfig`.

### Step 2: Implement the converter

Subclass `FileBasedDataConverter` and implement (see `ExampleConverter` in [converter.py](ncore_example/converter.py)):

1. **`get_sequence_ids(config)`** – Return the list of sequence identifiers (e.g. paths or IDs) to convert. Typically you scan `config.root_dir` (subdirs, manifest file, etc.).
2. **`from_config(config)`** – Return an instance of your converter built from the config.
3. **`convert_sequence(sequence_id)`** – Run conversion for one sequence (see below).

### Step 3: Implement `convert_sequence`

Follow the structure of `ExampleConverter.convert_sequence` in [converter.py](ncore_example/converter.py). For each sequence:

1. **Create the writer**  
   `SequenceComponentGroupsWriter(output_dir_path, store_base_name, sequence_id, sequence_timestamp_interval_us, store_type, generic_meta_data)`. (See the example's `store_writer` setup.)

2. **Component group layout**  
   Use `ComponentGroupAssignments.create(camera_ids, lidar_ids, radar_ids, point_clouds_ids, camera_labels_ids, profile=...)` so the layout matches what NuRec expects (same pattern as the [Waymo V4 converter](https://github.com/NVIDIA/ncore/blob/v19.8.0/tools/data_converter/waymo/converter.py) and the [skeleton converter](ncore_example/converter.py)).

3. **Register and use V4 component writers** from `ncore.data.v4` — each in its own `_write_*` method (see the skeleton for the full pattern):

   - **Poses** (`_write_poses`) — `PosesComponent` with rig→world dynamic, camera→rig, lidar→rig, and radar→rig static per sensor. Optionally world→world_global static if a global CRS is available.
   - **Intrinsics** (`_write_intrinsics`) — `IntrinsicsComponent` with camera intrinsics (`_write_camera_intrinsics`) and LiDAR intrinsics (`_write_lidar_intrinsics`). The skeleton demonstrates all three NCore camera models: `OpenCVPinholeCameraModelParameters`, `OpenCVFisheyeCameraModelParameters`, and `FThetaCameraModelParameters` (one camera per model). LiDAR intrinsics use `RowOffsetStructuredSpinningLidarModelParameters` for structured spinning sensors.
   - **Masks** (`_write_masks`) — per-camera static masks via `MasksComponent` (e.g. sky or ego-vehicle regions). Register with empty masks (`{}`) if your dataset has none; NuRec expects this component to exist.
   - **Cameras** (`_write_cameras`) — one `CameraSensorComponent.Writer` per camera, `store_frame()` per image. The skeleton registers writers for `pinhole_camera`, `fisheye_camera`, and `ftheta_camera`.
   - **LiDAR** (`_write_lidars`) — `LidarSensorComponent` with direction, distance, intensity, per-ray `timestamp_us`, and **`model_element`** `[N, 2]` `uint16` (row, col) aligned with `store_lidar_intrinsics`. See [LiDAR (spinning models)](#lidar-spinning-models).
   - **Radar** (`_write_radars`) — `RadarSensorComponent` with direction, distance, and timestamps per sweep (not required for NuRec).
   - **Cuboids** (`_write_cuboids`) — `CuboidsComponent` with `CuboidTrackObservation` and `BBox3`.  
     The skeleton converter demonstrates all of the above with placeholder data. Prefer **`ncore.data.v4`** component writers and `ncore.data` model types for stable surface area; layout helpers such as `ComponentGroupAssignments` and `HalfClosedInterval` may come from **`ncore.impl`** (same as upstream NCore converters—**not** a semver-stable public API; pin `nvidia-ncore` and re-validate after upgrades). See [NCore shared utilities](#ncore-shared-utilities) and [data.v4 API](https://nvidia.github.io/ncore/apis/data.html).

4. **Conventions** (see [NCore conventions](https://nvidia.github.io/ncore/data/conventions.html))

   - Rig frame: +X forward, +Y left, +Z up. Origin at the middle of the rear axle on the nominal ground plane.
   - Camera frame: +Z optical axis, +X right, +Y down.
   - Timestamps in **microseconds** (`uint64`).
   - Poses use `T_a_b` naming: maps points from frame `a` to frame `b`.
   - First `T_rig_world` is conventionally identity (local world frame).

5. **Timestamp constraints (important)**  
   The NCore V4 writer enforces these at write time — violating them raises `AssertionError`:

   - **Sequence interval** – Derive the interval from the sensor that provides the rig trajectory (typically ego-pose or LiDAR timestamps). `HalfClosedInterval.from_start_end(start, end)` sets `stop = end + 1`, giving the half-closed range `[start, stop)`.
   - **Dynamic poses must span the interval** – The first dynamic pose timestamp must equal `interval.start` and the last must equal `interval.stop - 1` (the inclusive end). If other sensors (e.g. cameras) have timestamps slightly outside the pose range, clamp them when writing frames.
   - **Frame end timestamps must be unique** – Each call to `store_frame()` must use a unique value for `frame_timestamps_us[1]` within that component writer. Two frames with the same end timestamp trigger an error.
   - **Frame timestamps within interval** – Both values in `frame_timestamps_us` must be within `[interval.start, interval.stop)`.
   - **Per-point timestamps and motion compensation** – Per-ray `timestamp_us` values must fall within `[frame_timestamps_us[0], frame_timestamps_us[1]]` and should reflect actual capture time (e.g. column-based for spinning LiDARs). NuRec uses these for motion compensation — always supply real per-ray capture times.

   For global-shutter cameras, `[ts, ts]` works for `frame_timestamps_us` — just ensure each `ts` is unique across frames.

6. **Finalize**  
   Call `store_writer.finalize()`.

7. **Sequence meta**  
   If `store_sequence_meta` is True, open the written store with `SequenceComponentGroupsReader`, call `get_sequence_meta().to_dict()`, and write it to `{sequence_id}.json` in the sequence output directory (required for NuRec / ncore_vis). The [example converter](ncore_example/converter.py) does this after `finalize()`.

Recommended component registration order: Poses → Intrinsics → Masks → Cameras/Lidars → … → `finalize()` → sequence meta JSON.

### Step 4: Register a CLI subcommand

In [converter.py](ncore_example/converter.py)'s CLI section (the merged `cli` Click group at the bottom of the file), add a subcommand that builds your Config and calls your converter (the **`example`** subcommand there shows how `ExampleConverter` and `ExampleConverterConfig` defined in the same module are wired up):

1. Add a new subcommand (e.g. `myformat`).
2. Add subcommand options: `--store-type`, `--profile`, `--sequence-meta` / `--no-sequence-meta` (and any format-specific options).
3. Build your Config from the shared group options plus subcommand options, then call `YourConverter.convert(config)`.

Shared options (already on the group) include `--root-dir`, `--output-dir`, `--no-cameras`, `--camera-id`, `--no-lidars`, `--lidar-id`, `--no-radars`, `--radar-id`, `--verbose`, `--debug`, `--debug-port`.

### Sequence meta

Sequence meta JSON is **required** for NuRec and ncore_vis. Two ways to get it:

1. **During conversion** – With `store_sequence_meta=True` (default), the converter writes `{sequence_id}.json` after `finalize()` using `SequenceComponentGroupsReader.get_sequence_meta().to_dict()`. See the end of `convert_sequence` in [converter.py](ncore_example/converter.py).
2. **From existing V4 data** – Use NCore's `ncore_sequence_meta` tool (from a clone of the [NCore repo](https://github.com/NVIDIA/ncore)):  
   `bazel run //tools:ncore_sequence_meta -- --output-dir=... v4 --component-group=<path_to_store_or_meta>`

---

## Validation

After converting, validate your output with the platform validator (provided by NuRec Onboard as `//skills/nurec-onboard/ncore_validation:validate`):

```bash
bazel run //skills/nurec-onboard/ncore_validation:validate -- --component-group=<path_to_sequence_meta.json> [--verbose]
# (run from the NuRec Onboard checkout, not from this workdir -- validators are a platform service)
```

### Example output

> **Note:** The skeleton converter writes intentionally-incomplete data — all
> static sensor-to-rig poses are identity, the LiDAR direction stream is
> derived directly from the placeholder model, and the fisheye `max_angle`
> doesn't match its distortion polynomial. The warnings below are _expected_
> for the skeleton and are how you confirm the validator is working; a real
> converter that loads its dataset's calibration and per-frame data should not
> produce them.

<details>
<summary>Running <code>validate --verbose</code> on the skeleton converter output</summary>

```text
  sequence_id:                    example_sequence_001
  version:                        v4
  sequence_timestamp_interval_us: [0, 100001)

  Poses: default
  Cameras: fisheye_camera, ftheta_camera, pinhole_camera
  Intrinsics: default
  Masks: default
  LiDAR: lidar_top
  Radar: radar_front
  Cuboids: default

V4 Structure:
  Sequence:
    ✓ sequence_id non-empty
    ✓ version is v4
    ✓ sequence_timestamp_interval_us (start < stop)
  Poses:
    ✓ rig→world dynamic in interval
    ✓ dynamic pose timestamps strictly increasing

NuRec Readiness:
  Sequence ID:
    ✓ sequence_id safe as a store filename and aux-discovery prefix
  Cameras:
    ✓ at least one present
    ✓ every camera has intrinsics
    ✓ every camera has camera→rig pose
    ✓ frame timestamps in interval
  Masks:
    ✓ component present
  LiDAR:
    ✓ at least one sensor present
    ✓ every LiDAR has intrinsics
    ✓ every sensor has lidar→rig pose
    ✓ frame timestamps in interval
    ✓ per-ray model_element present and consistent
  Cuboids:
    ✓ observations readable

Data Quality:
  Poses:
    ✓ static sensor→rig translations < 20.0 m
    ! all sensor-to-rig translations are near-zero — confirm the calibration file was loaded ...
    ✓ static pose dtypes (float32 except world→world_global which is float64)
    ✓ dynamic pose dtypes (float32)
    ✓ poses are valid SE3 (last row = [0, 0, 0, 1])
    ✓ rig→world translations near origin (max |component| = 10 m, float32 precision ≈ 0.00 mm)
  LiDAR:
    ✓ frame timestamps ordered (start <= end)
    ✓ frame timestamps strictly increasing across frames
    ✓ frame end timestamps unique
    ✓ direction vectors are unit-norm
    ✓ distances non-negative and < 300.0 m
    ✓ intensity values in [0, 1]
    ✓ per-point timestamps within frame bounds
    ✓ model accuracy <= 0.5° (stored vs model directions)
    ✓ LiDAR model FOV self-consistency
    ! lidar_top stored direction matches model-derived direction ... (LiDAR-smearing failure mode)
    ! lidar_top row_elevations_rad look uniformly spaced ...
    ! lidar_top column_azimuths_rad look uniformly spaced ...
    ✓ spinning LiDAR best forward ray in rig (dot >= 0.85)
  Cameras:
    ✓ frame timestamps ordered (start <= end)
    ✓ frame timestamps strictly increasing across frames
    ✓ frame end timestamps unique
    ! All 3 cameras share identical frame timestamps ...
    ! All 3 cameras have ShutterType.GLOBAL ...
    ! OpenCV-fisheye max_angle deviation: fisheye_camera: stored max_angle=90.0 deg, calibration-derived=126.2 deg ...
    ! camera optical axis far from horizontal: fisheye_camera: optical axis tilted 90 deg from horizontal (pointing up; z_rig=+1.00) ...
    ! camera optical axis far from horizontal: pinhole_camera: optical axis tilted 90 deg from horizontal (pointing up; z_rig=+1.00) ...
    ! camera optical axis far from horizontal: ftheta_camera: optical axis tilted 90 deg from horizontal (pointing up; z_rig=+1.00) ...
  Cross-Sensor:
    ✓ camera positions within LiDAR point cloud bounds
    ✓ camera and LiDAR timestamp ranges overlap
    ✓ cuboid timestamps overlap with sensor range
    ✓ cuboid observation timestamps in sequence interval
    ✓ cuboid reference_frame timestamps in sequence interval
    ✓ cuboid reference timestamp pose gap <= 1.0 ms ...
    ! only 4 near-range cuboids (< 50.0 m); need >= 5 for overlap check
    ! Sparse pose trajectory ...

✓ Validation passed.
```

</details>

### Scope

- **`validate`** (the NuRec Onboard platform validator) performs **structural** checks, **NuRec-oriented** readiness (including LiDAR **`model_element`** vs spinning grid), and **data-quality** heuristics. It does **not** replace NuRec's human-facing calibration guidance (see NuRec documentation, _Ensure data quality_) or subsume every training-dataloader invariant.
- **Official interactive check:** NuRec documentation points to the **NCore Data Sanity Check** notebook (`data_sanity_check.ipynb`) shipped with NCore docs on NGC. Use that notebook **in addition to** this script when you need the full documented workflow.
- **Pinning:** Ray-bundle field behavior and reader APIs follow the **`nvidia-ncore` version** declared under [deps/pip](deps/pip). After upgrading NCore, re-run `bazel test //...` and both `validate` and `eval` on your sequences.

### Validation tiers

The validator runs three tiers of checks and exits 0 on success, 1 on failure:

1. **V4 Structure:**
   - `sequence_id` non-empty, version `"v4"`
   - Valid `sequence_timestamp_interval_us`
   - Rig→world dynamic poses with timestamps in interval
   - Dynamic pose timestamps strictly increasing
   - Pose coverage warning (first/last timestamps match interval boundaries)
2. **NuRec Readiness:**
   - `sequence_id` free of `.`, `/`, and whitespace — sanitize with `sanitize_sequence_id()`. NuRec finds auxiliary stores by truncating the data shard filename at its first `.`, so a dotted ID matches nothing and the reconstruction silently runs without aux data.
   - At least one camera (with intrinsics, camera→rig poses, frame timestamps in interval)
   - Masks component present (NuRec requires it)
   - At least one LiDAR (with spinning intrinsics, lidar→rig poses, frame timestamps in interval, per-ray `model_element` `[N,2]` `uint16` within grid bounds); **`model_element` is always required** here (NCore V4 allows it to be optional; this template targets NuRec)
   - Cuboids readable (when present)
3. **Data Quality** — Validates data correctness and quality:
   - **Poses:**
     - Dtype enforcement (float32 for sensor poses, float64 for world→world_global)
     - SE3 validity (last row `[0,0,0,1]`)
     - Static pose translation sanity
     - Translation magnitude check (fails on unreferenced GPS/UTM coordinates — float32 precision loss causes motion-compensation artifacts)
     - Rig forward alignment check and warning
     - Angular rate warning (detects corrupted entries)
   - **LiDAR:**
     - Frame timestamp strict ordering and end-ts uniqueness
     - Direction unit-norm (squared-norm with 1e-4 tolerance matching NCore), distance/intensity range, per-point timestamp bounds
     - Model accuracy (angular error <= 0.5°)
     - Model FOV self-consistency (instantiates the NCore spinning model with `angles_to_columns_map_init=True` — the same code path NuRec executes — catching `AssertionError` when beam angles fall outside the model's FOV rectangle)
     - Model_element all-zero warning (detects `np.zeros` placeholder)
     - Uniform row elevations warning (detects `np.linspace` fallback instead of real per-beam calibration)
     - Uniform column azimuths warning (detects synthetic `np.linspace` schedule instead of real per-frame azimuth data)
     - Spinning direction data mismatch warning (detects `spinning_direction` disagreeing with `column_azimuths_rad` monotonicity)
     - Direction collapsed to model warning (detects the LiDAR-smearing failure mode where stored direction = model-derived direction with `row_azimuth_offsets_rad = 0`)
     - Spinning LiDAR forward ray alignment check (catches native-to-rig yaw double-counting in `T_lidar_rig`)
     - Cardinal ray axis residual warning (samples four model azimuths at 90° intervals to catch wholesale residual yaw on 360° spinners that the single-column forward-ray check cannot detect)
     - Low intensity dynamic range warning, centroid drift warning
   - **Cameras:**
     - Frame timestamp strict ordering and end-ts uniqueness
     - Identical timestamp warning
     - Global shutter enum warning
     - Frame count consistency warning
     - Fisheye `max_angle` deviation warning (validates OpenCV-fisheye `max_angle` against calibration-derived value via `OpenCVFisheyeCameraModelParameters.compute_max_angle()`)
     - Optical-axis-tilt warning (catches a missing source-frame to NCore-camera-frame basis change that leaves the camera looking straight up or down; id-agnostic, applies to every camera regardless of naming)
     - Optical-axis-azimuth warning (catches whole-rig yaw bugs in `T_camera_rig` when the camera id encodes a mounting direction — `front` / `back` / `left` / `right` and the four diagonals — by comparing the id-implied azimuth to the actual optical-axis azimuth in the rig frame; silent for opaque ids like `cam_0`. Tilt + azimuth together cover both vertical-axis-swap and horizontal-yaw failure modes; for opaque ids and pixel-level confirmation use NCore's `ncore_project_pc_to_img` / `ncore_vis`.)
   - **Cross-sensor:**
     - Camera positions within padded LiDAR AABB
     - Timestamp overlap
     - LiDAR-camera sweep midpoint alignment (hard error when frame timestamps are centered instead of start-aligned)
     - Sequence envelope coverage warning (warns when any sensor envelope doesn't span the sequence interval — NuRec restricts its working window to where rig poses exist and filters per sensor, so partial coverage is handled gracefully; cuboid envelope gaps are also advisory since per-track filtering downstream handles tracks that come and go mid-sequence)
     - Cuboid-sensor timestamp overlap
     - Individual cuboid timestamp interval containment
     - Cuboid-LiDAR overlap (near-range cuboids must contain LiDAR points — catches `T_lidar_rig` rotation errors)
     - Cuboid OBB inlier ratio check (detects dim-axis swaps and rotation errors)
     - Cuboid ground clearance warning (detects floating cuboids via below-bbox LiDAR point ratio — catches bottom-center vs centroid z convention errors)
     - Cuboid rig proximity check (catches world-frame mismatch after pose re-referencing)
     - Cuboid pose gap check
     - Sparse pose trajectory warning
     - Sweep timestamp centering warning
     - Clock sync warning

### Auxiliary data (not covered here)

NuRec may consume **auxiliary** zarr stores (segmentation, depth, etc.) depending on scenario config. The NuRec Onboard validator only inspects **core V4** component groups. See NuRec documentation (_Generate Auxiliary Data_) when your pipeline requires aux.

### NCore tools

For additional verification, use tools from the [NCore repo](https://github.com/NVIDIA/ncore):

- **ncore_sequence_meta** – Confirms stores are readable and produces sequence meta JSON for NuRec / ncore_vis.
- **ncore_vis** – Interactive 3D viewer (cameras, lidar, poses, cuboids).
- **ncore_project_pc_to_img** – Projects point clouds into camera images; verifies extrinsics and trajectory.
- **ncore_export_ply** – Exports point clouds (e.g. world frame).

More: [nvidia.github.io/ncore](https://nvidia.github.io/ncore). Opening V4 data with `SequenceComponentGroupsReader` already validates store format and consistency.

---

## Metrics evaluation

After validation, measure quantitative dataset quality with the platform eval binary (provided by NuRec Onboard as `//skills/nurec-onboard/ncore_validation:eval`):

```bash
bazel run //skills/nurec-onboard/ncore_validation:eval -- --component-group=<path_to_sequence_meta.json> \
# (run from the NuRec Onboard checkout, not from this workdir -- eval is a platform service)

  [--output-json metrics.json] [--output-yaml metrics.yaml] [--output-csv metrics.csv] [--quiet]
```

Follows a structured per-sensor metrics pattern: this script computes per-sensor and per-sequence metrics and exports to JSON, YAML, or CSV. A human-readable summary table is printed to stdout by default.

### Metric categories

| Category         | Metrics                                                                                                                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Poses**        | `rate_hz`, `jitter_us`, `temporal_coverage_pct`, `total_displacement_m`, `max_angular_rate_deg_s`, `rig_forward_alignment`                                                             |
| **Camera**       | `count`, `frame_balance`, `mask_presence`, per-camera `frame_rate_hz`, `resolution_w`/`resolution_h`, `distortion_coeff_order`                                                         |
| **LiDAR**        | per-sensor `points_per_frame`, `distance_min_m`/`max_m`/`mean_m`, `intensity_completeness`, `model_angular_error_max_deg`/`mean_deg`, `element_coverage_pct`, `model_element_oob_frac` |
| **Cuboids**      | `track_count`, `observation_density`, `single_obs_ratio`, `class_diversity`, `lidar_points_per_cuboid`, `lidar_density_pts_per_m3`                                                     |
| **Cross-sensor** | `temporal_overlap_pct`, `clock_consistency_ratio`                                                                                                                                      |

Single-valued metrics (e.g. `count`, `element_coverage_pct`) report a single number. Per-frame metrics (e.g. `points_per_frame`, `model_angular_error_max_deg`) are aggregated to `avg`/`min`/`max`/`std`/`cnt`. Each metric includes a human-readable description in the JSON output.

### Example output

<details>
<summary>Running <code>eval</code> on the skeleton converter output</summary>

```text
Metrics for sequence: ncore_example

  camera:
    (sequence):
      count                         3.0000
      frame_balance                 1.0000
      mask_presence                 1.0000
    fisheye_camera:
      distortion_coeff_order        0.0000
      frame_rate_hz                 4.0000
      resolution_h                  1080.0000
      resolution_w                  1920.0000
    ftheta_camera:
      frame_rate_hz                 4.0000
      resolution_h                  1080.0000
      resolution_w                  1920.0000
    pinhole_camera:
      distortion_coeff_order        0.0000
      frame_rate_hz                 4.0000
      resolution_h                  1080.0000
      resolution_w                  1920.0000

  cross_sensor:
    (sequence):
      clock_consistency_ratio       1.0000
      temporal_overlap_pct          20.0000

  cuboids:
    (sequence):
      class_diversity               1.0000
      lidar_density_pts_per_m3      avg=    0.1852  min=    0.1852  max=    0.1852  std=  0.0000  n=3
      lidar_points_per_cuboid       avg=    5.0000  min=    5.0000  max=    5.0000  std=  0.0000  n=3
      observation_density           1.0000
      single_obs_ratio              1.0000
      track_count                   3.0000

  lidar:
    lidar_top:
      distance_max_m                10.0000
      distance_mean_m               7.5000
      distance_min_m                5.0000
      element_coverage_pct          0.6944
      intensity_completeness        1.0000
      model_angular_error_max_deg   0.0215
      model_angular_error_mean_deg  0.0105
      points_per_frame              10.0000

  poses:
    (sequence):
      jitter_us                     0.0000
      max_angular_rate_deg_s        0.0000
      rate_hz                       2.0000
      rig_forward_alignment         1.0000
      temporal_coverage_pct         99.9999
      total_displacement_m          10.0000
```

</details>

---

## Common pitfalls

Issues encountered across multiple NCore converters (Waymo, PAI). Check these early — most manifest as misaligned LiDAR-camera overlays in `ncore_vis`.

### Coordinate frame conventions

Datasets often define sensor frames that differ from [NCore's conventions](https://nvidia.github.io/ncore/data/conventions.html). If your dataset uses a different convention, you must apply a rotation to the static sensor→rig pose. For example, the Waymo converter applies a 3×3 rotation to camera extrinsics because Waymo cameras use +X as the principal axis:

```python
T_sensor_rig[:3, :3] = T_sensor_rig[:3, :3] @ np.array(
    [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32
)
```

Verify alignment in `ncore_vis` as soon as you have a working converter — flipped or rotated LiDAR points relative to camera images are the most common symptom of a frame convention mismatch.

### Rig trajectory source

The rig→world dynamic pose can come from different sources depending on the dataset:

- **Dedicated egomotion** (e.g. PAI provides a separate egomotion parquet).
- **Camera poses** (e.g. Waymo uses `image.pose`; some datasets derive rig poses from camera GPS + calibration).
- **LiDAR poses** (if available and accurate).

The sequence timestamp interval must be derived from the **same source** as the rig trajectory. NCore requires the first dynamic pose timestamp to equal `interval.start` and the last to equal `interval.stop - 1`. Clamp any other sensor timestamps that fall outside this range.

### LiDAR (spinning models)

**NuRec and the current NCore tooling require structured spinning LiDAR models** (`RowOffsetStructuredSpinningLidarModelParameters`). See [Lidar models](https://nvidia.github.io/ncore/data/sensor_models.html#lidar-models) in the NCore sensor models documentation for the supported `row-offset-spinning` parameterization, and the broader [NCore data formats](https://nvidia.github.io/ncore/data/formats.html) docs for V4 component layout, `store_lidar_intrinsics`, and per-ray `model_element` indexing.

`validate` checks **model accuracy** (Data Quality tier): the angular error between stored ray directions and model-predicted directions must be **<= 0.5 deg** across all frames. The `eval` script also reports `model_angular_error_max_deg` and `model_angular_error_mean_deg` per LiDAR sensor for quantitative tracking.

### NCore shared utilities

The NCore repo provides transformation helpers (`ncore.impl.common.transformations`) and sensor model APIs (`ncore.sensors`) used by `validate.py`, `eval.py`, and the upstream Waymo/PAI converters. These live under **`ncore.impl`** (implementation namespace, not semver-stable) — **pin `nvidia-ncore`** and regression-test after upgrades. See [NCore API docs](https://nvidia.github.io/ncore/apis/data.html) and [sensor models](https://nvidia.github.io/ncore/data/sensor_models.html) for details.

### Verification checklist

After converting your data, verify the output before delivering to NuRec:

1. **Run `validate.py`** — all checks pass (structure, NuRec readiness, data quality including **model accuracy <= 0.5 deg** and **model FOV self-consistency**).
2. **Run `eval.py`** — review quantitative metrics (pose rate, LiDAR density, camera coverage, cuboid stats). Export to JSON for comparison across sequences.
3. **Open in `ncore_vis`** — LiDAR points should align with camera images when overlaid.
4. **Step through frames** — LiDAR points should remain static in world space (not drift with ego).
5. **Check camera positions** — should be symmetric around the rig if the physical setup is symmetric.
6. **Check cuboid alignment** — bounding boxes should match the LiDAR point cloud.
7. **Project LiDAR to cameras** — use `ncore_project_pc_to_img` for a pixel-level check of extrinsics and trajectory.

---

## Layout

- [ncore_example/converter.py](ncore_example/converter.py) – Skeleton converter (`ExampleConverter`, `ExampleConverterConfig`) and Click CLI in one module. Customize the converter class here; the agent renames `ncore_example/` to `ncore_<dataset>/` per dataset (single-token `example` → `<dataset>` substitution).
- [ncore_example/dataset_probe.py](ncore_example/dataset_probe.py) – Pre-conversion dataset probe skeleton; fill in dataset-specific exploration logic.
- **Shard validation, NuRec-readiness checks, and metrics evaluation** – live in NuRec Onboard, not in this workdir. They are invoked from there with the materialized workdir's output as input.
- [pyproject.toml](pyproject.toml) – Ruff settings (line-length, isort) aligned with NCore.

## Tests

- Run all tests:  
  `bazel test //...`
- Individual test targets:
  - `bazel test //tests:ncore_example_test` — runs the example converter and probe end-to-end and checks the converter produces valid V4 output.
  - `bazel test //tests:utils_test` — unit-tests the converter-author helpers in [ncore_example/utils.py](ncore_example/utils.py).
  - `bazel test //deps/pip:requirements_test` — checks the pinned `requirements_3_13.txt` lockfile is in sync with `requirements_3_13.in`.

`ncore_example_test` also invokes the platform `validate`/`eval` binaries through the `NCORE_VALIDATE_BIN` / `NCORE_EVAL_BIN` env vars; run without them, that portion is skipped.

## Code style

For consistency with NCore:

- SPDX license headers in new files.
- Ruff, 120-char line length (configured in [pyproject.toml](pyproject.toml)).
- Format: `bazel run //:format`.

See [NCore CONTRIBUTING](https://github.com/NVIDIA/ncore/blob/v19.8.0/CONTRIBUTING.md).
