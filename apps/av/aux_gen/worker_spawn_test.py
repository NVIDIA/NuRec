# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise aux loader payloads through Python's multiprocessing spawn path."""

from pathlib import Path

from python.runfiles import runfiles
from upath import UPath

from apps.av.aux_gen.ncore_aux_data import (
    LoaderWorkerPayload,
    ManifestLoaderWorkerPayload,
)
from apps.av.aux_gen.worker_spawn_test_support import open_payload_in_spawned_process
from nurec.ncore_backends import DatasetRequest, resolve_dataset


RUNFILES = runfiles.Create()
SEQUENCE = "clipgt-9048443e-c482-4228-8326-5b3dff3be711"


def _runfile(repository: str, filename: str) -> Path:
    path = Path(RUNFILES.Rlocation(f"{repository}/{filename}"))
    if not path.is_file():
        raise AssertionError(f"Test data not found at {path}")
    return path


def _manifest_payload() -> LoaderWorkerPayload:
    manifest_path = str(_runfile("test_data_ncore_v4", f"{SEQUENCE}.json"))
    resolved = resolve_dataset(DatasetRequest(metadata_path=UPath(manifest_path)))
    return ManifestLoaderWorkerPayload(resolved_dataset=resolved)


def main() -> None:
    backend_name, sequence_id = open_payload_in_spawned_process(_manifest_payload())
    if backend_name != "ncore-v4" or sequence_id != SEQUENCE:
        raise AssertionError(f"Expected {('ncore-v4', SEQUENCE)}, got {(backend_name, sequence_id)}")


if __name__ == "__main__":
    main()
