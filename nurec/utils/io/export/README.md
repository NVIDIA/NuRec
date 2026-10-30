<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Export Commands

This directory contains export utility commands for the NuRec library.

## `export-parsed-config`

This command is used to more easily export a parsed config from the existing config files or an USDZ artifact file. This can then be compared with the parsed config from a different nurec version to help check what changes were done between them, and then use this information to write the appropriate config upgrade function. The parsed configuration is printed to standard output (`stdout`) or written to a file.

### Usage

The command can be run via Bazel or Docker:

**Bazel:**

```bash
bazel run //:run -- export-parsed-config --config-name <config_name> [-o <output_file>] [--sort-keys/--no-sort-keys] [-- hydra_args...]
bazel run //:run -- export-parsed-config --input <artifact.usdz> [-o <output_file>] [--sort-keys/--no-sort-keys]
```

**Docker:**

```bash
docker run nurec_run export-parsed-config --config-name <config_name> [-o <output_file>] [--sort-keys/--no-sort-keys] [-- hydra_args...]
docker run nurec_run export-parsed-config --input <artifact.usdz> [-o <output_file>] [--sort-keys/--no-sort-keys]
```

**Arguments:**

- `-c, --config-name`: Path to the input config file that needs to be parsed. Configs are resolved from the repository `configs` directory by default; absolute config file paths are also accepted. The `.yaml` extension is optional.
- `-i, --input`: Path to the input artifact file (`.usdz`).
- `-o, --output`: (Optional) Path to the output file. When specified, a confirmation message is printed to standard output upon completion. If not provided, the parsed config is printed to standard output.
- `--sort-keys/--no-sort-keys`: Whether or not to sort the dictionary before exporting. It is recommended to sort (default) to make it comparable with other (sorted) configs.
- `--upgrade`: Enable upgrade mode to upgrade the config to the current version before exporting.
- `hydra_args`: (Optional) Any additional hydra overrides. Only applicable when using `--config-name`.

### Examples

**Using the CLI command:**

To parse a config file and save the output to `parsed_config.yaml`:

```bash
bazel run //:run -- export-parsed-config --config-name path/to/config_file --output parsed_config.yaml
```

To export the parsed config from an artifact and print it to the console:

```bash
bazel run //:run -- export-parsed-config --input /path/to/my_artifact.usdz
```

To parse a config with Hydra overrides and upgrade to current version:

```bash
bazel run //:run -- export-parsed-config --config-name my_config.yaml --upgrade -- dataset.batch_size=32
```

**Using Docker:**

```bash
docker run nurec_run export-parsed-config --config-name configs/my_config.yaml --output parsed_config.yaml
```

## `export-artifact-structure`

This command inspects the model checkpoint (`checkpoint.ckpt`) inside a `.usdz` artifact or a raw PyTorch checkpoint file and outputs the structure of all tensors as a hierarchical JSON object. For each tensor, the JSON output includes its shape, data type (`dtype`) and device.

This is useful for creating model upgrade functions, as it allows you to compare the checkpoint structure between two different NuRec versions and identify what has changed (e.g., renamed layers, modified tensor shapes).

### Usage

The command can be run via Bazel or Docker:

**Bazel:**

```bash
bazel run //:run -- export-artifact-structure --input <input_file> [-o <output.json>]
```

**Docker:**

```bash
docker run nurec_run export-artifact-structure --input <input_file> [-o <output.json>]
```

**Arguments:**

- `-i, --input`: (Required) Path to the input file. Can be a `.usdz` artifact or a PyTorch checkpoint file (`.ckpt`).
- `-o, --output`: (Optional) Path to the output JSON file. When specified, a confirmation message is printed to standard output upon completion. If not provided, the JSON is printed to standard output.

### Examples

**Using the CLI command:**

To inspect an artifact and print to console:

```bash
bazel run //:run -- export-artifact-structure --input /path/to/my_artifact.usdz
```

To inspect a checkpoint and save the output to a file:

```bash
bazel run //:run -- export-artifact-structure --input /path/to/checkpoint.ckpt --output structure.json
```

To inspect an artifact and save for comparison:

```bash
bazel run //:run -- export-artifact-structure --input /path/to/my_artifact.usdz --output my_artifact_structure.json
```

**Using Docker:**

```bash
docker run nurec_run export-artifact-structure --input /path/to/my_artifact.usdz --output structure.json
```

## `export-external-assets`

This command merges external assets from the Asset Harvester output into an existing NuRec artifact (`.usdz` file). The command also generates an `edit-assets.json` configuration file for use with `render-grpc`.

### Usage

The command can be run via Bazel or Docker:

**Bazel:**

```bash
bazel run //:run -- export-external-assets --artifact-path <artifact.usdz> --external-assets-dir <asset_harvester_output> --output-artifact-path <output.usdz> --output-edit-file <edit-assets.json>
```

**Docker:**

```bash
docker run nurec_run export-external-assets --artifact-path <artifact.usdz> --external-assets-dir <asset_harvester_output> --output-artifact-path <output.usdz> --output-edit-file <edit-assets.json>
```

**Arguments:**

- `--artifact-path`: (Required) Path to the input `.usdz` artifact file.
- `--external-assets-dir`: (Required) Path to the Asset Harvester output directory containing the external assets and `metadata.yaml`.
- `--output-artifact-path`: (Required) Path for the output `.usdz` artifact file with external assets included.
- `--output-edit-file`: (Required) Path for the `edit-assets.json` output file for use with `render-grpc`.

### Examples

**Using the CLI command:**

To merge external assets into an artifact:

```bash
bazel run //:run -- export-external-assets \
    --artifact-path /path/to/original.usdz \
    --external-assets-dir /path/to/asset_harvester_output \
    --output-artifact-path /path/to/output_with_assets.usdz \
    --output-edit-file /path/to/edit-assets.json
```

**Using Docker:**

```bash
docker run nurec_run export-external-assets \
    --artifact-path /path/to/original.usdz \
    --external-assets-dir /path/to/asset_harvester_output \
    --output-artifact-path /path/to/output_with_assets.usdz \
    --output-edit-file /path/to/edit-assets.json
```

The command will:

1. Load the PLY paths and asset metadata from the Asset Harvester `metadata.yaml` file.
2. Create a new artifact with the external assets under `external_assets/{track_id}/{track_id}.ply`.
3. Generate an `edit-assets.json` template that lists the packaged asset metadata.

## `export-gaussian-usd-asset`

This command exports Gaussian splatting model checkpoints to USD format. It supports three USD schema types:

- **geompoints**: Uses `UsdGeomPoints` primitives with custom primvars for Gaussian attributes
- **lightfield**: Uses the UsdVol ParticleField schema (`ParticleField3DGaussianSplat` or `ParticleField` with surflet API). Requires **usd-core>=26.3** for the UsdVol schema API. Supports split half-precision (geometry vs features).
- **nrendvol** (legacy, deprecated): Writes the model as a NuRec/NRend volume (`volume.usda` + `volume.nurec`) inside the same USD asset package. Retained only so NuRec-Isaac can keep consuming assets through the NRend RTX rendering path; it is not exercised by NuRec's own rendering, which is GSplat-only. Do not pick it for new work — it will be removed once NuRec-Isaac has migrated off the NRend RTX path.

The export can produce individual USD files (`.usda`, `.usdc`, `.usd`) or a packaged USDZ archive containing all assets.

### Features

- **Schema Selection**: Choose between `geompoints` (legacy) or `lightfield` (modern) USD schemas
- **Half-Precision Export**: Reduce file size using float16 attributes (LightField schema only)
- **Rigid Track Animation**: Export tracked objects with animated transforms
- **Deformable Gaussians**: Export models with animated positions, rotations, and scales
- **Temporal Appearance**: Export animated spherical harmonics / albedo coefficients
- **Environment Maps**: Automatically exports `SkyEnvMapBackground` as USD DomeLight with HDR texture
- **Rig Trajectories**: Optionally include camera/rig trajectory data
- **Gaussian Subsampling**: Reduce exported Gaussian count by percentage

### Usage

**Bazel:**

```bash
bazel run //:run -- export-gaussian-usd-asset \
    --config-name <config_name> \
    [--output-dir <output_directory>] \
    [--usd-format <usda|usdc|usd|usdz>] \
    [--usd-schema <geompoints|lightfield|nrendvol>] \
    [--percentage-gaussians <0-100>] \
    [--half-precision] \
    [--force-sh-0] \
    [--apply-activation] \
    [--flip-axis <xyz>] \
    [--do-not-cast-shadows] \
    [--skip-gaussian-deformation] \
    [--resample-animation/--no-resample-animation] \
    [--export-rig-trajectories/--no-export-rig-trajectories] \
    [-- hydra_args...]
```

**Arguments:**

- `--config-name`: (Required) Hydra config to load - must contain a dataset specification and checkpoint path. Configs are resolved from the repository `configs` directory by default; absolute config file paths are also accepted.
- `--output-dir`: Output directory path (defaults to `<checkpoint_dir>/../usd_asset`)
- `--usd-format`: USD output format: `usda` (text), `usdc` (binary), `usd`, or `usdz` (packaged archive, default)
- `--usd-schema`: USD schema type: `geompoints`, `lightfield` (default), or `nrendvol` (legacy, deprecated - see above)
- `--percentage-gaussians`: Percentage of Gaussians to export, 0-100 (default: 100)
- `--half-precision`: Use half-precision (float16) attributes for reduced file size (LightField schema only)
- `--force-sh-0`: Force SH degree to 0 (skip f_rest coefficients, export only DC color)
- `--apply-activation`: Export post-activation parameter values (applies sigmoid/exp/normalize)
- `--flip-axis`: Axes to flip in 'xyz' form (e.g., `--flip-axis y` to flip Y axis)
- `--do-not-cast-shadows`: Author `doNotCastShadows` primvar on Gaussian prims
- `--skip-gaussian-deformation`: Disable Gaussian deformation animation export
- `--resample-animation/--no-resample-animation`: Resample track timestamps to rig timestamps (default: enabled)
- `--export-rig-trajectories/--no-export-rig-trajectories`: Include rig trajectory data (default: enabled)

### Examples

**Export to USDZ with LightField schema:**

```bash
bazel run //:run -- export-gaussian-usd-asset \
    --config-name my_scene.yaml \
    --usd-schema lightfield \
    --output-dir /output/path
```

### PPISP integration

The exporter authors PPISP (Physically Plausible Image Signal
Processing) effects on the resulting USD asset. Two integration
modes are available, plus a controller path:

- `--ppisp-integration-mode {spg-runtime,sh-optimized}` (default
  `spg-runtime`) -- runtime SPG shaders vs. PPISP folded into the
  Gaussian SH coefficients.
- `--scene-radiance-scale FLOAT` -- asset-level multiplicative
  brightness scale baked into SH.
- `--ppisp-responsivity FLOAT` -- runtime achromatic HDR
  multiplier authored on every PPISP shader's
  `inputs:responsivity` USD attribute (SPG-runtime only).
- `--ppisp-reference-camera-id INT`,
  `--ppisp-reference-frame-id INT` -- pick which PPISP
  configuration is folded (in `sh-optimized`) or pinned (in
  `spg-runtime`).
- `--enable-ppisp-controller-export` /
  `--disable-ppisp-controller-export` -- controls whether the
  per-camera controller + automatic-parameter PPISP shaders are
  authored instead of the static SPG path. When omitted,
  `spg-runtime` exports use the controller whenever the loaded PPISP
  module was trained with controllers. Controller weights are embedded
  in generated per-camera CUDA sidecars; no separate weight sidecar is
  packaged.
- `--sh-optimization-num-iterations INT` -- override for the
  SH-match iteration count (only meaningful in `sh-optimized`).

See [`ppisp_export/README.md`](ppisp_export/README.md) for the rationale,
the authored USD surface (including Omniverse / RTX-specific
attributes), traps (camera shim, mode-specific flag scopes), and
troubleshooting.

## Ground Mesh Topology

Road initialization defaults to CUDA grid ground-mesh topology. Grid meshing is
faster than the legacy scipy/Qhull Delaunay path and keeps triangles local to
occupied BEV cells, avoiding long convex-hull bridges across unobserved
regions. Enclosed grid holes are filled before triangle emission so exported
road meshes remain continuous for simulation-oriented USDZ consumers.

Restore the legacy topology in the road-initialization config:

```bash
model.layers.road.initialization.mesh_topology=delaunay
```

Choose Delaunay only for workflows that prefer fully filled convex-hull
geometry over the grid path's local, observation-bounded topology.

Customer-facing export commands:

- `export-camera-ground-mesh` is pinned to grid. Flags are `--input`,
  `--output`, `--road-class-id` (repeatable, default 0), and
  `--postprocess/--no-postprocess`.
- `export-ground-mesh` still defaults to Delaunay. `--grid-hole-fill-min-neighbors`
  (default 6, range 0–8; `0` disables filling) applies when reconstruction uses
  grid topology. A degenerate mesh (fewer than three vertices, or no triangles)
  is an error rather than an empty PLY.
- instant-nurec USDZ ground-mesh export still pins Delaunay.

See [Customize NuRec Configuration](https://docs.nvidia.com/nurec/nurec/configs.html).
Hosted NuRec links track the latest published docs and may differ from this
checkout.

# Other Export Commands

This directory also contains many other export commands for various data types:

- `depth.py` - Export depth maps
- `ego_mask.py` - Export ego masks
- `gaussian_plys.py` - Export Gaussian PLY files
- `gaussian_statistics.py` - Export Gaussian statistics
- `gaussian_usd_asset.py` - Export Gaussian models to USD format (see above)
- `ground_mesh.py` - Export ground meshes
- `mesh.py` - Export meshes
- `ncore_diagnostic.py` - Export NCore diagnostics
- `ncore_tracks.py` - Export NCore tracks
- `point_cloud.py` - Export point clouds
- `render.py` - Render images along training or custom trajectories with optional bounding box overlay
- `render_grpc.py` - Render via gRPC
- `rig_trajectories.py` - Export rig trajectories
- `sequence_tracks.py` - Export sequence tracks
- `usdz_artifact.py` - Export USDZ artifacts

For usage details of these commands, run:

```bash
bazel run //:run -- <command-name> --help
```
