# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

from pathlib import Path

from python.runfiles import runfiles  # type: ignore[import-not-found]


def _executable() -> Path:
    """The PoissonRecon executable built from the `@poisson_recon` repository."""
    path = runfiles.Create().Rlocation("poisson_recon/poisson_recon")
    if not path or not Path(path).exists():
        raise FileNotFoundError("poisson_recon executable not found in the runfiles")
    return Path(path)


def reconstruct_surface(
    input_file: str, output_file: str, width: float = 0.1, density: bool = True, samples_per_node: float = 1.0
) -> None:
    """Runs Poisson surface reconstruction on an oriented point cloud PLY, writing the mesh to `output_file`."""
    command = [
        str(_executable()),
        "--in",
        input_file,
        "--out",
        output_file,
        "--width",
        str(width),
        "--samplesPerNode",
        str(samples_per_node),
        "--colors",
    ]
    if density:
        command.append("--density")
    subprocess.run(command, check=True)
