<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Omniverse NuRec Developer Hub

**Real-world data to interactive, photorealistic simulation.**

NVIDIA Omniverse™ NuRec is a set of APIs and libraries that leverage Gaussian-based methods to reconstruct and render interactive simulations from real-world data.

[![Python (dev container): 3.13](https://img.shields.io/badge/Python%20%28dev%20container%29-3.13-blue)][source-setup]
[![Code license: Apache 2.0](https://img.shields.io/badge/Code%20license-Apache%202.0-green)](LICENSE.md)
[![contributions: welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)
[![docs: NuRec](https://img.shields.io/badge/docs-NuRec-76b900)][docs]

[![Hugging Face: Models](https://img.shields.io/badge/Hugging%20Face-Models-yellow)][nvidia-models]
[![NVIDIA: Forum](https://img.shields.io/badge/NVIDIA-Forum-76b900)][forum]

## What's New

**NuRec 26.10**

- **Modular SDK source.** Inspect and customize GPU-accelerated building blocks, reconstruction models, training systems, and rendering services. Start with the [source overview](#repository-overview), [sensor library](nurec/libs/sensors/README.md), and [loss library](nurec/libs/losses/README.md).
- **Container-based development and execution.** Build a development container from the GHCR development dependencies image, and use separate runtime images for the reconstruction application and tools. See [build setup](#build-develop-and-run-from-source) and the [AV benchmark workflow](workflows/av_benchmark/README.md).
- **LiDAR v2:** Reconstruct a scene from measured rays and render observations for spinning and solid-state sensors with different scan patterns, sensor geometries, and timing, including simultaneous multi-LiDAR rendering at new vehicle poses. See the [LiDAR implementation](nurec/utils/unstructured_lidar.py) and [sensor API](nurec/grpc/protos/sensorsim.proto).
- **Radar reprojection.** Update recorded radar returns and Doppler estimates for simulated vehicle and actor poses, supporting closed-loop sensor-fusion evaluation within the coverage of recorded viewpoints. See the [repository][radar-repo] and [build and usage guide][radar-usage].
- **A reproducible AV benchmark workflow.** Run a defined reconstruction-to-evaluation chain with task timings, image and object metrics, and scene export. See the [workflow](workflows/av_benchmark/README.md).
- **A Physical AI dataset refresh.** Updated reconstructions and benchmark results for Physical AI scenes; the [dataset card][dataset-av-nurec] is the source for available versions.

For release-specific changes, container versions, and known issues, see the [NuRec 26.10 release notes][release-notes].

## What is Omniverse NuRec?

NVIDIA Omniverse™ NuRec is a set of APIs and libraries that leverage Gaussian-based methods to reconstruct and render interactive simulations from real-world data. It includes libraries and containers for reconstruction, a rendering API, and tools for scene initialization and rendering refinement.

NuRec uses the open-source [gsplat][gsplat-repo] library for efficient Gaussian splatting and high-fidelity scene rendering. It supports camera features critical to physical AI simulation, including rolling shutter and fisheye lenses.

## Why Omniverse NuRec?

|                                                     |                                                                                                                                                                                                                  |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Narrow the real-to-sim gap**                      | Reconstruct environments from real-world observations, then refine artifacts, lighting mismatches, and temporal flicker to improve the consistency of simulated sensor data.                                     |
| **Scale scene creation and simulation preparation** | Use GPU-accelerated reconstruction and rendering, multi-GPU training, and reusable scenes to scale preparation and evaluation across sensor configurations.                                                      |
| **Get more value from each capture**                | Turn captures into reusable environments and object assets for interactive simulation, synthetic data generation, and repeated evaluation within scene coverage.                                                 |
| **Adopt the capabilities your pipeline needs**      | Integrate individual reconstruction, rendering, and enhancement APIs and libraries into an existing stack, or use NuRec's libraries and containers as a complete workflow.                                       |
| **Connect sensor observations to simulation**       | Use NCore, OpenUSD, and rendering APIs to connect capture to supported simulators. AV workflows can combine camera and LiDAR rendering with radar reprojection for closed-loop perception and fusion evaluation. |

## Get Started

Choose the guide that matches your application. Check the [hardware][hardware] and [software][software] requirements before running NuRec.

| Your goal                                                             | Start here                                             |
| --------------------------------------------------------------------- | ------------------------------------------------------ |
| Use or reconstruct driving scenes for AV simulation                   | [For AV Developers][av-guide]                          |
| Reconstruct robotics environments or use existing scenes in Isaac Sim | [For Robotics Developers][robotics-guide]              |
| Connect NuRec rendering to your own simulator                         | [For Simulation Platform Developers][simulation-guide] |

**Try an existing scene first:** use the [pre-reconstructed AV dataset][dataset-av-nurec] and follow [Render the Physical AI Dataset][render-pai]. This path starts at rendering; it does not require you to reconstruct the scene first.

**Bring your own data:** follow [Prepare Data][prepare-data], [Generate Auxiliary Data][aux-data], and [Reconstruct an AV Scene][reconstruction]. For robotics inputs, use the robotics guide above.

**Render a USDZ artifact:** the [render command][render-command] uses a trained artifact without the original dataset or a gRPC service. Render camera images by default, or use `--lidar` for native LiDAR point clouds. Run each modality separately with its own output directory.

## Which repository should I use?

Choose a component by the task you want to accomplish. Installation, model details, and supported inputs are maintained at the linked destinations.

| Component          | Use it to                                                                                                  | Resources                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| NCore              | Convert and represent sensor recordings for NuRec reconstruction                                           | [GitHub][ncore-repo] · [Documentation][prepare-data]                                             |
| Omniverse NuRec    | Reconstruct and render interactive scenes from real-world data, or integrate individual APIs and libraries | [GitHub][nurec-repo] (This Repository)· [Documentation][docs]                                    |
| Radar Reprojection | Reproject recorded radar detections for simulated ego and actor poses                                      | [GitHub][radar-repo] · [Build and usage][radar-usage]                                            |
| gsplat             | Integrate Gaussian splatting rendering primitives into your own pipeline                                   | [GitHub][gsplat-repo] · [Documentation][gsplat-docs]                                             |
| InstantNuRec       | Predict Gaussian scenes from images for rapid inspection or initialization of NuRec reconstruction         | [GitHub][instant-repo] · [Documentation][instant-guide] · [Hugging Face][instant-model]          |
| Fixer              | Remove artifacts from neural reconstruction outputs to improve novel-view fidelity                         | [GitHub][fixer-repo] · [Hugging Face][fixer-model]                                               |
| Harmonizer         | Correct rendering artifacts, lighting mismatches, and temporal flicker                                     | [GitHub][harmonizer-repo] · [Documentation][harmonizer-guide] · [Hugging Face][harmonizer-model] |
| Asset Harvester    | Extract reusable Gaussian actors and objects for scene edits and synthetic data generation                 | [GitHub][harvester-repo] · [Documentation][harvester-guide] · [Hugging Face][harvester-model]    |
| NuRec Agent Skills | Guide a coding agent through NuRec tasks                                                                   | [Skills index][agent-skills-guide]                                                               |

## Workflows

### Data Ingestion

#### NCore

NCore provides a common representation for sensor recordings, calibration, and pose across NuRec workflows. Convert a recording once, then use it as input to reconstruction and related data-processing tools.

- **What it provides:** a shared data representation and conversion guides for bringing your own sensor recordings into NuRec.
- **What you can build:** a repeatable data-ingestion path for fleet recordings or public driving datasets.

**Resources:** [Repository][ncore-repo] · [Data and conversion guides][ncore-docs]

### Reconstruction

#### Omniverse NuRec

Use NuRec's APIs, libraries, and containers to reconstruct and render interactive simulations from real-world data, or integrate individual components into an existing pipeline.

- **What it provides:** modular reconstruction libraries and containers, camera and LiDAR rendering APIs, tools for scene initialization and rendering refinement, OpenUSD scene artifacts, and simulator-facing interfaces.
- **What you can build:** reusable environments and synthetic sensor data for physical AI applications, including robotics and autonomous driving.

**Resources:** [Source and SDK][nurec-repo] · [Build setup](#build-develop-and-run-from-source) · [AV benchmark workflow](workflows/av_benchmark/README.md) · [AV guides][av-guide] · [Robotics guides][robotics-guide]

#### InstantNuRec

InstantNuRec is a feed-forward neural model that predicts 3D Gaussian scenes from images in one pass. It runs on a single GPU and can initialize NuRec reconstruction or provide standalone predictions for rapid scene inspection before refinement and rendering.

- **What you can build:** rapid previews from captured images and an initial scene representation for subsequent reconstruction.

**Resources:** [Repository and usage][instant-repo] · [Model][instant-model]

### Scene Enhancement and Editing

#### Fixer

Fixer removes artifacts from neural reconstruction outputs, including flicker, blur, and visual errors in novel-view renders of 3D Gaussian Splat scenes. It cleans rendered outputs before they reach simulation or synthetic data generation.

- **What you can build:** a render-cleanup pass for trajectories that depart from the original capture path or expose under-observed parts of a reconstruction.

**Resources:** [Repository][fixer-repo] · [Model][fixer-model] · [Technical blog][fixer-blog]

#### Harmonizer

Harmonizer is a diffusion model that corrects artifacts, lighting mismatches, and temporal flicker in neural scene renderings. Applied after NuRec reconstruction and rendering, it improves visual consistency for training and simulation. Temporal conditioning stabilizes sequences, and object harmonization aligns inserted assets with the surrounding scene.

- **What you can build:** a render-refinement pass for novel viewpoints, edited scenes, and inserted assets, with consistent lighting and appearance across frames.

- **Built on NVIDIA Cosmos**

  The Harmonizer post-processing feature uses [Cosmos-Predict2-0.6B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image) and the [Harmonizer](https://huggingface.co/nvidia/Harmonizer) checkpoints, which are licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). The model weights are not part of this repository or of the published images: the server downloads them from Hugging Face on first use, and you must accept the Cosmos-Predict2 access terms on Hugging Face and provide an `HF_TOKEN` first. See [Use Harmonizer for Post-Processing](https://docs.nvidia.com/nurec/nurec/harmonizer.html).

**Resources:** [Repository][harmonizer-repo] · [Model][harmonizer-model] · [NuRec integration guide][harmonizer-guide]

#### Asset Harvester

Asset Harvester is a model pipeline that extracts reusable Gaussian-based 3D actors and objects from driving recordings and reconstructed scenes, including limited or obstructed views. Given a reconstruction and an object ID, it produces a reusable asset for scene population. Reuse harvested assets across traffic configurations and scene edits, with Harmonizer aligning inserted assets to the surrounding environment.

- **What you can build:** asset libraries drawn from real fleet observations and diverse simulation scenes populated without manually creating every object.

**Resources:** [Repository][harvester-repo] · [Model][harvester-model] · [Live demo][harvester-demo] · [Benchmark][object-benchmark] · [NuRec integration guide][harvester-guide]

These components support different entry points. Scene enhancement and asset extraction are optional additions to a workflow; they are not mandatory stages for every reconstruction or rendering task.

### Rendering and Simulator Integration

- **AlpaSim:** start with the [AlpaSim repository][alpasim-repo] for closed-loop AV simulation.
- **CARLA:** follow the [CARLA NuRec integration guide][carla-guide].
- **Isaac Sim:** follow [Neural Volume Rendering][isaac-guide].
- **Radar Reprojection:** use the CPU library or gRPC service alongside NuRec camera and LiDAR rendering. Follow the [build and usage guide][radar-usage] for setup and integration.
- **Custom simulator:** follow the [gRPC server guide][grpc-guide] and [API reference][api-reference].

For rendering implementations and research, see [gsplat][gsplat-repo] and [3DGRUT][3dgrut-repo]. Use the NuRec integration guides above for the supported product workflow.

## Repository Overview

### NuRec SDK

NuRec can be used as a complete reconstruction and rendering workflow or as building blocks within another pipeline. The source tree separates [accelerated libraries](nurec/libs/), [models](nurec/models/), [training systems](nurec/systems/), data loading, and [application entry points](apps/). For example, the sensor library exposes differentiable projection kernels and Python model wrappers; the loss library provides fused computation and configurable loss modules.

### gsplat

- **gsplat is an independent open-source project.** gsplat is a CUDA-accelerated Gaussian splatting library with Python bindings, hosted by the Nerfstudio project and developed by contributors from UC Berkeley, NVIDIA, and other institutions. It is not an NVIDIA product.
- **NuRec renders through gsplat.** NuRec uses a single gsplat-based rendering path for cameras and LiDAR, built on gsplat's 3DGUT rasterizer, and builds against a pinned gsplat revision.
- **NVIDIA contributes NuRec capabilities upstream.** Features developed for NuRec, such as 3DGUT, LiDAR rasterization, NCore V4 capture support, and the HiGS-based inference rendering path, are contributed to gsplat, where any gsplat user can adopt them.
- **Pick the level you need.** Use gsplat directly if you already have a Gaussian splatting training pipeline and want faster or more capable rendering. Use NuRec if you want the full pipeline: multi-sensor data handling, dynamic actors, USDZ export, enhancement, and serving.

### Source, containers, and models

NuRec assets are distributed through six channels. Use the one that matches what you need.

| What                           | Where                                                                                       | How to get it                                    | Terms                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| NuRec source                   | GitHub: [NVIDIA/NuRec][nurec-repo]                                                          | `git clone https://github.com/NVIDIA/NuRec.git`  | Apache 2.0                                           |
| Development dependencies image | GitHub Container Registry: `ghcr.io/nvidia/nurec/nurec_dev_deps`                            | `docker pull`; tag 26.10                         | See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| Runtime images                 | NVIDIA NGC: [Omniverse NuRec container][ngc-container] · [NuRec Tools][ngc-tools-container] | `docker pull`; use the tag for your release      | As listed on the NGC catalog pages                   |
| Models                         | Hugging Face: [nvidia][nvidia-models]                                                       | Download from each model card                    | NVIDIA Open Model License                            |
| Datasets                       | Hugging Face: [Physical AI collection][physical-ai-collection]                              | Accept each dataset's terms on Hugging Face      | Per dataset; see [Datasets](#datasets)               |
| Python packages                | PyPI                                                                                        | `pip install nvidia-ncore`, `pip install gsplat` | Apache 2.0                                           |

**Choosing between GHCR and NGC.** Use the GHCR development dependencies image to build a development container for this repository. Runtime containers are distributed through NVIDIA NGC. Follow the [release notes][release-notes] for the image names and tags appropriate to your release.

## Build, develop, and run from source

### Set up and build

Prepare a GPU-enabled development container and build the NuRec application and tools. Follow the [source setup guide][source-setup] for software requirements, cloning, container setup, and build instructions.

### Develop and test

Before submitting a change, follow the [contributor tests and checks](CONTRIBUTING.md#tests-and-checks) for formatting, builds, and full or targeted tests.

### Configure and run

Choose and customize a configuration for your data and reconstruction task. Follow the [configuration guide][configuration] for presets, overrides, and time-range settings.

For source-checkout training and checkpoint validation, follow the [reconstruction guide][reconstruction] and [validation guide][validation]. To resume training, see [Resuming a prior run](skills/nurec/references/cli-reference.md#resuming-a-prior-run).

### Render a trained scene

Render camera images or native LiDAR point clouds from a USDZ artifact without the original recording or a live gRPC service. Follow the [source rendering guide][render-command] for commands, camera and LiDAR selection, resolution defaults, and separate invocations for each modality.

## Agent Skills

NuRec agent skills guide coding agents through dataset discovery, data onboarding, reconstruction, rendering, and AV benchmarking. The skills are included in this repository under [`skills/`](skills/). See the [NuRec Agent Skills index][agent-skills-guide] for available skills and usage guidance.

## Datasets

Pick data at the stage where you want to enter the workflow. Each dataset card contains its current contents, access requirements, and license.

| Dataset                                        | Use it for                                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| [Physical AI Autonomous Vehicles][dataset-av]  | Start from raw driving recordings; use the [AV devkit][av-devkit] for data access                                                    |
| [Physical AI AV NCore][dataset-av-ncore]       | Start reconstruction from data already converted to NCore                                                                            |
| [Physical AI AV NuRec][dataset-av-nurec]       | Use reconstructed Gaussian driving scenes in AlpaSim for policy testing and reinforcement learning without reconstructing each scene |
| [Physical AI Robotics NuRec][dataset-robotics] | Use reconstructed robotics environments                                                                                              |
| [Physical AI NuRec PPISP][dataset-ppisp]       | Explore photometric variation in reconstruction                                                                                      |
| [NuRec AV Object Benchmark][object-benchmark]  | Evaluate object reconstruction from driving observations                                                                             |
| [Harmonizer Dataset][harmonizer-dataset]       | Train models to correct rendering artifacts and harmonize lighting and inserted objects                                              |

The AV datasets require acceptance of their access terms. Follow the dataset card and the relevant NuRec guide for download and preparation.

## Tech Blogs

Explore NVIDIA technical articles for worked examples and tutorials. Use the [official documentation][docs] for current setup and API details.

- [Scale AV Perception Across Vehicle Platforms with NVIDIA Omniverse NuRec](https://developer.nvidia.com/blog/scale-av-perception-across-vehicle-platforms-with-nvidia-omniverse-nurec/) — Render reconstructed scenes through a different vehicle's camera rig.
- [Optimizing a Neural Reconstruction Pipeline Using NVIDIA Nsight Developer Tools](https://developer.nvidia.com/blog/optimizing-a-neural-reconstruction-pipeline-using-nvidia-nsight-developer-tools/) — Profile and optimize the NuRec reconstruction pipeline.
- [How to Enhance 3D Gaussian Reconstruction Quality for Simulation][fixer-blog] — Explore artifact cleanup with Fixer.
- [Reconstruct a Scene in NVIDIA Isaac Sim Using Only a Smartphone](https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/) — Follow a capture-to-simulation robotics example.
- [Revolutionizing Neural Reconstruction and Rendering in gsplat with 3DGUT](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) — Learn the rendering and reconstruction foundations.

## Documentation

Use the [official NuRec documentation][docs] for detailed procedures. These links take you directly to common tasks.

| Topic                              | Official guide                                                             |
| ---------------------------------- | -------------------------------------------------------------------------- |
| Prepare reconstruction inputs      | [Prepare Data][prepare-data] · [Generate Auxiliary Data][aux-data]         |
| Reconstruct and assess an AV scene | [Reconstruction][reconstruction] · [Validation][validation]                |
| Generate new camera views          | [Novel Views][novel-views] · [Render Command][render-command]              |
| gsplat                             | [gsplat documentation][gsplat-docs]                                        |
| Refine renders and insert assets   | [Harmonizer][harmonizer-guide] · [Asset Harvester Output][harvester-guide] |
| Integrate a renderer               | [gRPC Server][grpc-guide] · [API Reference][api-reference]                 |
| Scale a reconstruction workflow    | [Scaling Neural Reconstruction][scaling-guide]                             |

For sensor schemas, conventions, and converters, use the [NCore documentation][ncore-docs]. For release-specific behavior, use the [NuRec release notes][release-notes].

## Support

- **Usage questions and NuRec container help:** [NVIDIA Developer Forum — NuRec][forum].
- **NuRec source bugs, documentation issues, and feature requests:** use [NuRec Issues][issues]. For source questions, use the Question / General Discussion issue form.

## Contributing

For NuRec code and documentation changes, follow [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests are reviewed on GitHub, imported into NVIDIA's source repository for review and testing, and published in a subsequent source update. GitHub pull requests are then closed with a reference to the landed commit rather than merged directly. Discuss new features, public API or configuration changes, and substantial refactors in an issue first. For companion repositories, follow each repository's contribution policy.

## Security

To report a security vulnerability, please follow the process described in [SECURITY.md](SECURITY.md).

## License

This project is licensed under the Apache License 2.0. See [LICENSE.md](LICENSE.md) for details. Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Containers, model weights, companion repositories, and datasets have their own terms; consult the linked container page, repository, or model and dataset card.

**Vendored PyTorch3D code**

[`nurec/libs/pytorch3d_knn/`](nurec/libs/pytorch3d_knn/) contains k-nearest-neighbor kernels derived from [PyTorch3D](https://github.com/facebookresearch/pytorch3d), copyright Meta Platforms, Inc. and affiliates. The files derived from PyTorch3D are under the BSD-3-Clause license. Its text is in [`nurec/libs/pytorch3d_knn/LICENSE`](nurec/libs/pytorch3d_knn/LICENSE) and is repeated in the header of each of those files. The other source files in that directory carry NVIDIA Apache-2.0 headers.

[docs]: https://docs.nvidia.com/nurec/
[hardware]: https://docs.nvidia.com/nurec/basics/hardware.html
[software]: https://docs.nvidia.com/nurec/basics/software.html
[source-setup]: https://docs.nvidia.com/nurec/basics/software.html#build-and-run-from-source
[agent-skills-guide]: https://docs.nvidia.com/nurec/nurec/agent-skills.html
[release-notes]: https://docs.nvidia.com/nurec/release-notes/index.html
[av-guide]: https://docs.nvidia.com/nurec/av/index.html
[robotics-guide]: https://docs.nvidia.com/nurec/robotics/index.html
[simulation-guide]: https://docs.nvidia.com/nurec/simulation/index.html
[prepare-data]: https://docs.nvidia.com/nurec/basics/get-data.html
[aux-data]: https://docs.nvidia.com/nurec/nurec/nurec-aux-data.html
[reconstruction]: https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html
[validation]: https://docs.nvidia.com/nurec/nurec/run-validation.html
[configuration]: https://docs.nvidia.com/nurec/nurec/configs.html
[render-pai]: https://docs.nvidia.com/nurec/nurec/physical-ai-data.html
[render-command]: https://docs.nvidia.com/nurec/nurec/render-command.html
[novel-views]: https://docs.nvidia.com/nurec/nurec/novel-view-synthesis.html
[harmonizer-guide]: https://docs.nvidia.com/nurec/nurec/harmonizer.html
[harvester-guide]: https://docs.nvidia.com/nurec/nurec/use-ah-assets.html
[grpc-guide]: https://docs.nvidia.com/nurec/api/grpc_api_guide.html
[api-reference]: https://docs.nvidia.com/nurec/api/index.html
[scaling-guide]: https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html
[nurec-repo]: https://github.com/NVIDIA/NuRec
[ncore-repo]: https://github.com/NVIDIA/ncore
[ncore-docs]: https://nvidia.github.io/ncore/
[radar-repo]: https://github.com/NVIDIA/radar-reprojection
[radar-usage]: https://github.com/NVIDIA/radar-reprojection/blob/main/docs/USAGE.md
[gsplat-repo]: https://github.com/nerfstudio-project/gsplat
[gsplat-docs]: https://docs.gsplat.studio/
[instant-repo]: https://github.com/NVIDIA/instant-nurec
[instant-model]: https://huggingface.co/nvidia/instant-nurec
[instant-guide]: https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html#launch-reconstruction-model-training
[fixer-repo]: https://github.com/nv-tlabs/Fixer
[fixer-model]: https://huggingface.co/nvidia/Fixer
[fixer-blog]: https://developer.nvidia.com/blog/how-to-enhance-3d-gaussian-reconstruction-quality-for-simulation/
[harmonizer-repo]: https://github.com/NVIDIA/harmonizer
[harmonizer-model]: https://huggingface.co/nvidia/Harmonizer
[harmonizer-dataset]: https://huggingface.co/datasets/nvidia/Harmonizer-Dataset
[harvester-repo]: https://github.com/NVIDIA/asset-harvester
[harvester-model]: https://huggingface.co/nvidia/asset-harvester
[harvester-demo]: https://huggingface.co/spaces/nvidia/asset-harvester
[object-benchmark]: https://huggingface.co/datasets/nvidia/NuRec-AV-Object-Benchmark
[alpasim-repo]: https://github.com/NVlabs/alpasim
[carla-guide]: https://carla.readthedocs.io/en/latest/nvidia_nurec/
[isaac-guide]: https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_nurec.html
[3dgrut-repo]: https://github.com/nv-tlabs/3dgrut
[nvidia-models]: https://huggingface.co/nvidia
[ngc-container]: https://catalog.ngc.nvidia.com/orgs/nvidia/nre/containers/nurec-ga
[ngc-tools-container]: https://catalog.ngc.nvidia.com/orgs/nvidia/nre/containers/nurec-tools-ga
[physical-ai-collection]: https://huggingface.co/collections/nvidia/physical-ai
[dataset-av]: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
[dataset-av-ncore]: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore
[dataset-av-nurec]: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
[dataset-robotics]: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec
[dataset-ppisp]: https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP
[av-devkit]: https://github.com/NVlabs/physical_ai_av
[forum]: https://forums.developer.nvidia.com/c/omniverse/platform/nurec/752
[issues]: https://github.com/NVIDIA/NuRec/issues
