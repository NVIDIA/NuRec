<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `nurec/libs/losses`

## Consuming a gsplat loss

The fused dispatches are `torch.nn.Module`s. Own them as submodules of
`ModuleLosses` and call them directly. They cannot be called from inside a
`torch.autograd.Function.forward`, which runs under `no_grad` — the graph it
builds is discarded and gradients vanish with no error.

Read `gsplat/losses/design.md` for the calling contract before writing a call
site. It defines factor semantics (positive scales, zero writes zeros,
negative disables the member), when a disabled member's tensors may be
omitted, which reductions belong to a dispatch rather than to the caller, and
where row masks come from.
