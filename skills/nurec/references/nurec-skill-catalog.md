<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Skill routing

Register the skills available under `skills/` using the public README's
agent-skill instructions.

- `nurec`: source-built reconstruction, scene rendering/export, gRPC, and the
  shipped Physical AI AV benchmark workflow.
- `nurec-onboard`, when present: raw dataset onboarding, including writing and
  validating an NCore converter before reconstruction.
- `physical-ai-datasets`: dataset selection, access, bounded downloads, and a
  handoff to a concrete consumer.
- `physical-ai-av-benchmark`, when present in the checkout: the dedicated
  end-to-end Physical AI AV benchmark skill.
  Prefer its instructions for that workflow; otherwise use the source-based
  benchmark recipe in this skill.

For raw Physical AI AV conversion, use the shipped `//apps:nurec_tools` commands
or end-to-end benchmark. For other raw formats, consult the public NCore project
and verify a compatible converter exists. Do not assume a separate `ncore`,
`harmonizer`, or `asset-harvester` skill is installed. Use this checkout's
product documentation for supported commands; explain missing capabilities rather
than fetching a guessed skill path.
