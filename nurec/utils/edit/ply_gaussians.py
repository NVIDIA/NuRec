# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static-prop input for the asset-edits bake: a 3D Gaussian Splatting ``.ply``.

A SimReady USD asset is the bake's native input (its variantSets are the controls).
A ``.ply`` is the format the rest of NuRec already speaks for ONE static splat cloud —
AssetBank tracks, asset-harvester exports, the ``edit_assets`` RPC — and this module
lets the bake accept one wherever it accepts a controls-free SimReady asset. The file
is read with the same :class:`PLYGaussianLoader` those paths use and presented as a
:class:`SimReadyAsset` holding one static part, no variantSets and no controls, so
placement, row append, world-bind synthesis and the manifest record are untouched.

Conventions follow ``edit_assets``, so a ``.ply`` means the same thing to the bake as
to the live RPC:

* The file is **Y-up** (the 3DGS / AssetBank convention) and is rotated into the Z-up
  asset frame the placement math expects — the fixed rotation
  ``RigidGaussianModel.insert_asset`` applies to every PLY. Spherical harmonics are
  copied as stored: NuRec evaluates SH in the world frame and never rotates them, and
  the SimReady->PLY converter leaves them alone on the way out for the same reason.
* Coordinates are taken as **metres**; the placement's ``scale`` decides what that
  means. ``fit_length`` (the default on track binds) fits the asset's bbox to the
  cuboid, so the unit-normalised PLYs the asset bank ships need nothing extra there;
  a world bind takes ``bind.dims`` + ``scale: fit_length``, or a numeric ``scale``.
* Attributes are pre-activation like the USD payload (log scales, logit opacity);
  ``sh_rest`` is ``(N, K-1, 3)`` coefficient-major, and the row builder pads or
  truncates it to the target layer's SH width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from nurec.models.gaussians.utils import PLYGaussianLoader
from nurec.utils.edit.layer_append import matrix_to_quat_wxyz, quat_mul_wxyz
from nurec.utils.edit.simready_gaussians import SimReadyAsset, SimReadyGaussians, SimReadyPart


# Y-up file frame -> Z-up asset frame (+90 deg about x): the rotation
# ``RigidGaussianModel.insert_asset`` applies to every PLY the live RPC inserts.
ZUP_FROM_YUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class PlyAssetError(ValueError):
    """Raised when a ``.ply`` cannot serve as a static asset-edit prop."""


def _f64(tensor: torch.Tensor) -> np.ndarray:
    """A float64 numpy copy of a CPU tensor (the USD importer's payload dtype)."""
    return tensor.detach().cpu().numpy().astype(np.float64)


def load_ply_asset(
    ply_path: Union[str, Path],
    control_sets: Optional[list[str]] = None,
    root_prim_path: Optional[str] = None,
) -> SimReadyAsset:
    """Load a 3DGS ``.ply`` as a controls-free :class:`SimReadyAsset` with one static part.

    Same signature as :func:`load_simready_asset` so the baker can pick the loader by
    the asset's suffix; ``control_sets`` and ``root_prim_path`` exist only to be
    refused (a PLY has no variantSets and no prims). The part is named after the file.
    """
    path = Path(ply_path)
    if control_sets:
        raise PlyAssetError(
            f"{path.name}: a .ply is a static splat cloud with no variantSets, so controls "
            f"{list(control_sets)} cannot bind to it (controllable edits need a SimReady USD asset)"
        )
    if root_prim_path is not None:
        raise PlyAssetError(f"{path.name}: a .ply has no prims; prim_path {root_prim_path!r} does not apply")
    try:
        loaded = PLYGaussianLoader(path, quaternion_format="wxyz", device="cpu")
    except KeyError as exc:
        raise PlyAssetError(f"{path.name}: not a 3DGS PLY (missing vertex property {exc.args[0]!r})") from exc

    n = int(loaded.positions.shape[0])
    if n == 0:
        raise PlyAssetError(f"{path.name}: PLY holds no gaussians")

    positions = _f64(loaded.positions) @ ZUP_FROM_YUP.T
    rotations = _f64(loaded.rotations)
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    if not (norms > 1e-12).all():
        # A zero quaternion is corrupt authoring; passing it through would silently
        # collapse those splats' orientations (same rule as the USD importer).
        raise PlyAssetError(f"{path.name}: {int((norms <= 1e-12).sum())} zero-norm quaternion(s)")
    q_up = np.broadcast_to(matrix_to_quat_wxyz(ZUP_FROM_YUP), (n, 4))
    rotations = quat_mul_wxyz(q_up, rotations / norms)
    scales_log = _f64(loaded.scales)
    densities_logit = _f64(loaded.densities).reshape(n, 1)
    sh_dc = _f64(loaded.features_albedo)
    if loaded.features_specular is None:
        sh_rest = np.zeros((n, 0, 3))
    else:
        # The loader hands back (N, (K-1)*3) coefficient-major; the payload keeps (N, K-1, 3).
        sh_rest = _f64(loaded.features_specular).reshape(n, -1, 3)
    sh_count = sh_rest.shape[1] + 1
    sh_degree = int(round(np.sqrt(sh_count))) - 1
    if (sh_degree + 1) ** 2 != sh_count:
        raise PlyAssetError(
            f"{path.name}: {sh_rest.shape[1] * 3} f_rest_* properties is not a whole SH degree "
            "(expected 0, 9, 24 or 45)"
        )
    for name, arr in (
        ("positions", positions),
        ("rotations", rotations),
        ("scales", scales_log),
        ("opacity", densities_logit),
        ("f_dc", sh_dc),
        ("f_rest", sh_rest),
    ):
        if not np.isfinite(arr).all():
            raise PlyAssetError(f"{path.name}: non-finite values in {name}")

    gaussians = SimReadyGaussians(
        positions=positions,
        rotations=rotations,
        scales_log=scales_log,
        densities_logit=densities_logit,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        up_axis="Z",
        meters_per_unit=1.0,
        sh_degree=sh_degree,
    )
    return SimReadyAsset(
        asset_path=str(path),
        root_prim_path="",  # a PLY has no prims
        static_parts=[SimReadyPart(prim_path=path.name, gaussians=gaussians)],
        controls={},
        variant_sets={},
        up_axis="Z",
        meters_per_unit=1.0,
    )
