#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${NUREC_SRC:?set NUREC_SRC to the released checkout matching the server}"
proto_source="$NUREC_SRC/nurec/grpc/protos"
proto_dest="${NUREC_RENDER_CLIENT_PROTO_DIR:-$HOME/.cache/nurec-render-client/proto}"
python_bin="${NUREC_CLIENT_PYTHON:-python3}"
for name in common sensorsim ipc; do
    if [[ ! -f "$proto_source/$name.proto" ]]; then
        echo "Missing source schema: $proto_source/$name.proto" >&2
        exit 1
    fi
done
"$python_bin" -c 'import grpc_tools.protoc' || {
    echo "Install grpcio-tools in your Python 3.13 client environment; see the client README." >&2
    exit 1
}
marker=.nurec-render-client-generated
if [[ -e "$proto_dest/nurec" && ! -f "$proto_dest/nurec/$marker" ]]; then
    echo "Refusing to overwrite an existing nurec package; choose another NUREC_RENDER_CLIENT_PROTO_DIR." >&2
    exit 1
fi
mkdir -p "$proto_dest"
# Generate into a temporary directory; never copy a partially generated module.
proto_stage=$(mktemp -d "$proto_dest/.generate.XXXXXXXX")
trap 'rm -rf -- "$proto_stage"' EXIT
"$python_bin" -m grpc_tools.protoc \
    --proto_path="$NUREC_SRC" --python_out="$proto_stage" --grpc_python_out="$proto_stage" \
    "$proto_source/common.proto" "$proto_source/sensorsim.proto" "$proto_source/ipc.proto"
touch "$proto_stage/nurec/__init__.py" "$proto_stage/nurec/grpc/__init__.py" \
    "$proto_stage/nurec/grpc/protos/__init__.py" "$proto_stage/nurec/$marker"
"$python_bin" - "$proto_stage" <<'PYCODE'
import sys
sys.path.insert(0, sys.argv[1])
from nurec.grpc.protos import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc, ipc_pb2
assert hasattr(common_pb2, "Pose")
assert hasattr(sensorsim_pb2, "RGBRenderRequest")
PYCODE
# Replace only this helper's generated package, after compilation and import checks pass.
rm -rf -- "$proto_dest/nurec"
mv "$proto_stage/nurec" "$proto_dest/nurec"
echo "Generated client stubs from $NUREC_SRC into $proto_dest."
