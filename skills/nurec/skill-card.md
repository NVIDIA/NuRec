<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Description: <br>

Use to drive NVIDIA Omniverse NuRec / Neural Reconstruction Engine from released source in a development image built on the matching GitHub-hosted development-dependencies image to train 3DGUT Gaussian reconstructions from NCore clips, generate aux data, render frames or LiDAR sweeps, export PLY/depth/mesh/USDZ, edit actors, and evaluate metrics. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner

NVIDIA <br>

### License/Terms of Use: <br>

Apache-2.0 <br>

## Use Case: <br>

Developers and engineers working with autonomous vehicle sensor data use this skill to train neural 3D reconstructions, render novel camera and LiDAR views, export scene artifacts, and edit actors with source-built NVIDIA Omniverse NuRec. <br>

### Deployment Geography for Use: <br>

Global <br>

## Known Risks and Mitigations: <br>

Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>

- [NVIDIA Omniverse NuRec Product Page](https://www.nvidia.com/en-us/omniverse/) <br>
- [PhysicalAI Autonomous Vehicles NuRec Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) <br>
- [Harmonizer Model](https://huggingface.co/nvidia/Harmonizer) <br>
- [NuRec Documentation](https://docs.nvidia.com/nurec/) <br>
- [NuRec Skills GitHub Repository](https://github.com/NVIDIA/NuRec/tree/release/26.10/skills) <br>
- [Installation Guide](references/install.md) <br>
- [Cookbook](references/cookbook.md) <br>
- [CLI Reference](references/cli-reference.md) <br>
- [Workflows](references/workflows.md) <br>
- [gRPC API](references/grpc-api.md) <br>

## Skill Output: <br>

**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces USDZ, PLY, MP4, PNG, YAML, and JSON artifacts via container execution] <br>

## Evaluation Tasks: <br>

Historical NVSkills-Eval profile `external`: Tier 1 static validation (9 checks), Tier 2 deduplication (2 checks). These results predate the source-workflow migration; no fresh live agent-evaluation pass is claimed. <br>

## Evaluation Metrics Used: <br>

Reported benchmark dimensions: <br>

- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

## Skill Version(s): <br>

0.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
