<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

-->

# Public Harmonizer weights

The public model repository is [nvidia/Harmonizer](https://huggingface.co/nvidia/Harmonizer).
For non-temporal NuRec enhancement, use `harmonizer_nontemporal.pt` with its
576 × 1024 resolution. Do not substitute the temporal `.pkl` checkpoint into
this recipe; it uses a different loader and additional base-model configuration.
Read the model card's terms before use. Raw reconstruction/rendering does not
require this optional enhancement step.

This source release pins the public model in
`configs/harmonizer/public_harmonizer_nontemporal.yaml`. Prefer that config for
Hydra-composed training recipes:

```bash
nurec_dev bazelisk run //:run -- main \
  --config-name="$CONFIG_PATH" "dataset.path=$DATASET_JSON" "out_dir=$OUTPUT_DIR" \
  harmonizer=public_harmonizer_nontemporal harmonizer.training.enabled=true
```

Inspect the resolved configuration before launching training. The benchmark's
standalone resolved YAML is not a Hydra config-group recipe: leave its disabled
Harmonizer settings alone unless intentionally changing the benchmark, and use
explicit `harmonizer.model_url`, `harmonizer.model_filename`, and
`harmonizer.model_resolution` overrides when adapting a resolved config.

For gRPC, explicitly select the matching public checkpoint:

```bash
export HARMONIZER_URL=https://huggingface.co/nvidia/Harmonizer/resolve/f2bae9823e07d281c9dce6581d06118d675e08c3/harmonizer_nontemporal.pt
nurec_dev bazelisk run //:run -- serve-grpc \
  --artifact-glob="$USDZ_PATH" --host=127.0.0.1 --port="${NUREC_GRPC_PORT:-8080}" \
  --enable-harmonizer --harmonizer-url="$HARMONIZER_URL" \
  --harmonizer-model-filename=harmonizer_nontemporal.pt --harmonizer-resolution 576 1024
```

A client must also request enhancement with the supported rendering flag or RPC
field; inspect `render-grpc --help` or `RGBRenderRequest.apply_harmonizer`.
The server binds before loading the optional model: use its health/readiness
service and inspect load errors before an enhanced render. Preserve raw frames
for comparison. If access is denied, check the actual model's requirements and
pass `HF_TOKEN` through the environment only when required.

For temporal enhancement or standalone image/video processing, follow the public
[Harmonizer code](https://github.com/NVIDIA/Harmonizer) and
[NuRec Harmonizer guide](https://docs.nvidia.com/nurec/nurec/harmonizer.html); do not assume a separate skill is installed.
Hosted NuRec links track the latest published docs and may differ from this
checkout.
