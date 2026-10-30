<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Description: <br>

Use to onboard a raw autonomous-driving dataset to NVIDIA Omniverse NuRec end to end: audit the raw data, write an NCore V4 converter from the bundled template, validate and evaluate its output, then run a NuRec reconstruction, looping back to fix the converter when validation or reconstruction diagnoses an NCore data issue. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner

NVIDIA <br>

### License/Terms of Use: <br>

Apache-2.0 <br>

## Use Case: <br>

Developers and engineers with autonomous-vehicle sensor data in a format NCore does not yet support use this skill to produce a reusable NCore V4 converter for that dataset and a first NuRec reconstruction from it, without hand-writing the converter. <br>

### Deployment Geography for Use: <br>

Global <br>

## Known Risks and Mitigations: <br>

Risk: The agent runs shell and Docker commands autonomously, and Docker commands run on the host with access to the configured mounts. <br>
Mitigation: Both supported agents run with fail-closed command sandboxes and per-launch network allowlists; review the SKILL files and Docker templates before running GPU/container steps (see the Security section of the [README](../README.md#security)). <br>

Risk: The generated converter can misread the raw data (for example sensor frames, timestamps or calibration), producing NCore data that looks valid but reconstructs poorly. <br>
Mitigation: Every conversion passes the bundled NCore validation and eval gates before reconstruction, inputs without a reviewed dataset profile are audited with cited evidence first, and reconstruction diagnoses route NCore data issues back to the converter. <br>

Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>

- [NVIDIA Omniverse NuRec Documentation](https://docs.nvidia.com/nurec/) <br>
- [NCore Library](https://github.com/NVIDIA/ncore) <br>
- [Setup and Usage](../README.md) <br>
- [Workflow Specification](SKILL.md) <br>
- [Example Inputs](../examples/) <br>
- [Converter Workspace Template](../ncore_template/README.md) <br>

## Skill Output: <br>

**Output Type(s):** [Shell commands, Source code, Files] <br>
**Output Format:** [Bazel workspace with Python converter source, NCore V4 data, YAML] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces a standalone NCore V4 converter workspace, converted NCore V4 data, a custom NuRec config, a NuRec reconstruction (USDZ plus PSNR/SSIM/LPIPS metrics and predicted-view videos), and a YAML workflow summary] <br>

## Evaluation Tasks: <br>

Not yet evaluated with NVSkills-Eval. Each step's output contract is enforced by the validators under [`validators/`](../validators/), which are covered by NuRec Onboard's Bazel test suite together with the launcher and agent-configuration tests. <br>

## Evaluation Metrics Used: <br>

None reported yet. <br>

## Skill Version(s): <br>

0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
