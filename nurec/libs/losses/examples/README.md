<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NuRec Examples

This directory contains example scripts demonstrating various NuRec features and modules.

## Available Examples

### Loss Functions (`example_losses.py`)

Demonstrates various loss functions in the NuRec losses module:

- **Python-based losses**: Simple loss functions like `relu_sum` and `bce_clipped`
- **Slang-based Losses**: Advanced GPU-accelerated losses using the Slang shader language via `SlangLosses`
- **CUDA-based losses**: Advanced CUDA-accelerated losses used in the training pipeline

## Running an Example

The examples are included in the NuRec image.
Run an example script using the `run-script` command, e.g.:

```bash
docker run --rm -it --gpus all <nurec-image> -- run-script nurec/libs/losses/examples/example_losses.py
```

Replace `<nurec-image>` with your NuRec Docker image name.

### Expected Output

```
Using device: cuda
Python example:
  ReLU sum loss: tensor(1.2000, grad_fn=<ToCopyBackward0>)
  BCE clipped loss: tensor([0.7981, 0.4032, 0.2014], grad_fn=<ToCopyBackward0>)
  Time to run: 0.079 seconds

Slang example:
  Computed 2 loss(es):
    rgb_l1_mean: 0.335279
    background_mse_mean: 0.459050
  Time to run: 0.085 seconds

CUDA example: RoadGaussiansLoss
  road_gaussians_abs_mean: 26.164042
  Time to run: 0.054 seconds
```
