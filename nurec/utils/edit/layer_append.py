# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gaussian-layer append utilities for the asset-edits bake.

The tensor-level half of the bake, kept free of any schema/manifest knowledge:
placement math (euler/quaternion helpers), row construction for imported SimReady
clusters, and :class:`LayerAppender`, which owns the row-index arithmetic when
appending to a layer's parameter tensors. :class:`~nurec.utils.edit.asset_edits.AssetEditsBaker`
drives these; the manifest records the indices they return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from nurec.datasets.tracks import CuboidTracks
from nurec.models.gaussians.gaussians_model import RigidGaussianModel

# One Hamilton product for the whole feature (re-exported here for the bake's callers).
from nurec.utils.edit.simready_gaussians import quat_mul_wxyz  # noqa: F401


# Sentinel logit written into the (pre-sigmoid) ``densities`` parameter for overlay
# rows. Any large negative number squashes through sigmoid to ~0 (hidden by default;
# the render runtime overwrites the post-activation value per frame).
HIDDEN_DENSITY_LOGIT = -10.0

# Per-Gaussian tensors we extend when appending an overlay cluster.
LAYER_PARAM_NAMES = [
    "positions",
    "rotations",
    "scales",
    "densities",
    "extra_signal",
    "camera_extra_signal",
    "lidar_extra_signal",
    "features_albedo",
    "features_specular",
    "gaussian_cuboid_ids",
]


# Row-aligned sidecar buffers the bake can fill for IMPORTED rows (copies take the
# source's values). Anything else a layer declares can only be copied, never synthesized.
_SIDECAR_FILLERS = {
    # A layer's MCMC densifier clamps positions to their initial value; a new row's anchor is
    # its own position.
    "mcmc_position_anchor": lambda rows: rows["positions"].clone(),
}


def layer_row_tensor_names(layer_node: torch.nn.Module) -> list[str]:
    """Every row-aligned per-Gaussian tensor of a layer.

    The standard parameters plus the sidecar buffers the layer class declares through
    ``get_additional_buffers`` (for example ``mcmc_position_anchor``). Every
    append and every checkpoint splice must cover exactly this set, or the layer fails its own
    field validation.
    """
    names = list(LAYER_PARAM_NAMES)
    getter = getattr(layer_node, "get_additional_buffers", None)
    if callable(getter):
        for name, tensor in getter().items():
            if name not in names and isinstance(tensor, torch.Tensor):
                names.append(str(name))
    return names


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def euler_xyz_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Rotation matrix ``R = Rz @ Ry @ Rx`` (degrees) acting on column vectors (``p' = R @ p``)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion ``(w, x, y, z)``."""
    m = np.asarray(rot, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def make_simready_rows(
    *,
    spec: "LayerTensorSpec",
    positions_np: np.ndarray,
    rotations_np: np.ndarray,
    scales_log_np: np.ndarray,
    densities_logit_np: np.ndarray,
    sh_dc_np: np.ndarray,
    sh_rest_np: np.ndarray,
    cuboid_track_index: int,
    hidden: bool = True,
) -> dict[str, torch.Tensor]:
    """Build per-Gaussian rows for one imported SimReady cluster (real captured appearance).

    The rows carry the asset's own pre-activation scales/rotations and SH, so the cluster
    renders with its authored appearance. It is baked *hidden* by default
    (``hidden=True`` -> the density sentinel) and the runtime toggles its opacity from the
    schedule selects which baked variant clusters are visible for each frame.
    """
    n = int(positions_np.shape[0])
    device, dtype = spec.device, spec.dtype
    positions = torch.from_numpy(np.ascontiguousarray(positions_np)).to(device=device, dtype=dtype)
    rotations = torch.from_numpy(np.ascontiguousarray(rotations_np)).to(device=device, dtype=dtype)
    scales = torch.from_numpy(np.ascontiguousarray(scales_log_np)).to(device=device, dtype=dtype)
    if hidden:
        densities = torch.full((n, 1), HIDDEN_DENSITY_LOGIT, device=device, dtype=dtype)
    else:
        densities = torch.from_numpy(np.ascontiguousarray(densities_logit_np.reshape(n, 1))).to(
            device=device, dtype=dtype
        )

    sh_dc = torch.from_numpy(np.ascontiguousarray(sh_dc_np)).to(device=device, dtype=dtype)  # (n, 3)
    dc_cols = min(3, spec.albedo_sh_dim)
    if spec.fourier_features_dim > 1:
        features_albedo = torch.zeros((n, spec.fourier_features_dim, spec.albedo_sh_dim), device=device, dtype=dtype)
        features_albedo[:, 0, :dc_cols] = sh_dc[:, :dc_cols]
    else:
        features_albedo = torch.zeros((n, spec.albedo_sh_dim), device=device, dtype=dtype)
        features_albedo[:, :dc_cols] = sh_dc[:, :dc_cols]

    features_specular = torch.zeros((n, spec.specular_sh_dim), device=device, dtype=dtype)
    if spec.specular_sh_dim > 0 and sh_rest_np.size:
        rest_flat = np.ascontiguousarray(sh_rest_np.reshape(n, -1))  # (n, 15*3) coeff-major
        k = min(spec.specular_sh_dim, rest_flat.shape[1])
        features_specular[:, :k] = torch.from_numpy(rest_flat[:, :k]).to(device=device, dtype=dtype)

    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "densities": densities,
        "extra_signal": torch.zeros((n, spec.extra_signal_dim), device=device, dtype=dtype),
        "camera_extra_signal": torch.zeros((n, spec.camera_extra_signal_dim), device=device, dtype=dtype),
        "lidar_extra_signal": torch.zeros((n, spec.lidar_extra_signal_dim_total), device=device, dtype=dtype),
        "features_albedo": features_albedo,
        "features_specular": features_specular,
        "gaussian_cuboid_ids": torch.full((n,), cuboid_track_index, device=device, dtype=torch.int32),
    }


def replace_layer_param(node: torch.nn.Module, name: str, *new_tensors: torch.Tensor) -> None:
    """Concatenate ``new_tensors`` onto the existing parameter / buffer ``name`` in ONE cat."""
    current = getattr(node, name)
    if not isinstance(current, torch.Tensor):
        raise TypeError(f"{name}: expected tensor-like, got {type(current).__name__}")
    cat = torch.cat([current.data, *(t.to(current.dtype) for t in new_tensors)], dim=0)
    if isinstance(current, torch.nn.Parameter):
        setattr(node, name, torch.nn.Parameter(cat.contiguous()))
    else:
        setattr(node, name, torch.nn.Buffer(cat.contiguous()))


@dataclass(frozen=True)
class LayerTensorSpec:
    """Row shapes/dtype/device for tensors appended to one Gaussian layer."""

    fourier_features_dim: int
    albedo_sh_dim: int
    specular_sh_dim: int
    extra_signal_dim: int
    camera_extra_signal_dim: int
    lidar_extra_signal_dim_total: int
    scale_activation: str
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def from_layer(cls, layer_node: RigidGaussianModel, fourier_features_dim: int) -> "LayerTensorSpec":
        """Read the spec off a live layer node (dims, activation, device, dtype)."""
        # Appended rows carry LOG-space scales, which only decode correctly through an
        # exp scale activation. Fail the bake loudly rather than mis-size every splat.
        activation = str(layer_node.config.scale_activation)
        if activation != "exp":
            raise ValueError(
                f"layer scale_activation is {activation!r}, but the asset-edits bake writes "
                "log-space scales that assume 'exp'; this layer cannot take baked overlays."
            )
        return cls(
            fourier_features_dim=fourier_features_dim,
            albedo_sh_dim=layer_node.get_albedo_sh_dim(),
            specular_sh_dim=layer_node.get_specular_sh_dim(),
            extra_signal_dim=int(layer_node.config.particle.extra_signal_dim),
            camera_extra_signal_dim=int(layer_node.config.particle.camera_extra_signal_dim),
            lidar_extra_signal_dim_total=int(layer_node.config.particle.lidar_extra_signal_dim)
            * int(layer_node.get_lidar_extra_signal_sph_dim()),
            scale_activation=str(layer_node.config.scale_activation),
            device=layer_node.positions.device,
            dtype=layer_node.positions.dtype,
        )


@dataclass(frozen=True)
class CapturedSnapshot:
    """One layer's captured (trained) state, snapshotted BEFORE any edit touches it.

    ``replace`` / ``exclude`` must hide ORIGINAL rows only, never appended ones (appended
    rows are tagged to the same track), and a captured clone must copy the TRAINED
    density logits even when an earlier edit of the same bake has already hidden its
    source track — so both the per-row cuboid ids and the density logits are kept.
    """

    cuboid_ids: np.ndarray  # (N0,) layer-local track index per captured row
    densities_logit: np.ndarray  # (N0,) pre-activation density logit per captured row

    @classmethod
    def from_layer(cls, layer_node: RigidGaussianModel) -> "CapturedSnapshot":
        """Snapshot the layer's per-Gaussian cuboid ids and density logits as they are RIGHT NOW."""
        # COPIES, not views: on a CPU layer ``.numpy()`` aliases the parameter storage, and the
        # bake's in-place hides would rewrite the snapshot underneath later edits.
        return cls(
            cuboid_ids=np.array(layer_node.gaussian_cuboid_ids.detach().cpu().numpy(), copy=True).reshape(-1),
            densities_logit=np.array(layer_node.densities.detach().cpu().numpy(), dtype=np.float32, copy=True).reshape(
                -1
            ),
        )

    def rows_of_track(self, track_idx: int) -> np.ndarray:
        """The captured rows riding ``track_idx`` (ascending row indices)."""
        return np.nonzero(self.cuboid_ids == track_idx)[0]


class LayerAppender:
    """Appends imported or copied Gaussian rows to one layer node.

    Baked indices are layer-local and contiguous from ``initial_n`` (the layer size before the
    first append), which is what the manifest records and the runtime replays.

    Appends are staged: each batch is validated immediately but concatenated onto the layer
    only by :meth:`flush`, one ``torch.cat`` per tensor for the whole bake. Call ``flush()``
    before reading the layer's tensors.
    """

    def __init__(self, layer_node: RigidGaussianModel, spec: LayerTensorSpec) -> None:
        """Bind to a layer node; records its pre-append size as ``initial_n``."""
        self.node = layer_node
        self.spec = spec
        self.initial_n = int(layer_node.get_num_gaussians())
        self.appended = 0
        self.row_names = layer_row_tensor_names(layer_node)
        self._pending: dict[str, list[torch.Tensor]] = {}

    def _append_rows(self, rows: dict[str, torch.Tensor]) -> list[int]:
        """Stage one row batch for every layer tensor; returns the new row indices.

        Indices are pure arithmetic from ``initial_n`` plus rows staged so far, so
        they are identical to eager per-part concatenation. Unknown / non-tensor
        targets still fail HERE, exactly as the eager path did.
        """
        # Imported rows carry the standard tensors only; fill the layer's sidecar buffers.
        for name in self.row_names:
            if name in rows or name in LAYER_PARAM_NAMES:
                continue
            filler = _SIDECAR_FILLERS.get(name)
            if filler is None:
                raise ValueError(
                    f"layer has a per-Gaussian sidecar buffer {name!r} the bake cannot synthesize for imported "
                    "rows; only copies of captured rows can be appended to this layer"
                )
            rows[name] = filler(rows)
        # A batch must cover EXACTLY the layer tensor set with one row count: a
        # missing or extra key, or a mismatched leading dim, would desynchronize
        # the per-tensor row arithmetic silently at flush.
        expected = set(self.row_names)
        if set(rows) != expected:
            missing = sorted(expected - set(rows))
            extra = sorted(set(rows) - expected)
            raise ValueError(f"row batch keys do not match the layer's tensors: missing {missing}, unexpected {extra}")
        n = int(rows["positions"].shape[0])
        for name, tensor in rows.items():
            if int(tensor.shape[0]) != n:
                raise ValueError(f"{name}: row batch has {int(tensor.shape[0])} rows, expected {n} (positions)")
            current = getattr(self.node, name)
            if not isinstance(current, torch.Tensor):
                raise TypeError(f"{name}: expected tensor-like, got {type(current).__name__}")
            self._pending.setdefault(name, []).append(tensor)
        start = self.initial_n + self.appended
        self.appended += n
        return list(range(start, start + n))

    def flush(self) -> None:
        """Concatenate every staged row batch onto the layer — ONE cat per tensor."""
        for name, batches in self._pending.items():
            replace_layer_param(self.node, name, *batches)
        self._pending.clear()

    def widen_frame_table(self, width: int) -> None:
        """Pad the layer's per-track-frame appearance table (``features_albedo``, ``[G, N, 3]``) to ``width`` columns.

        Every row, captured or staged, repeats its last column, as the trained table pads a track
        with fewer frames than the longest; a row reads only its own track's frames, so nothing
        renders differently. Needed when a clone has more keyframes than the table has columns.
        """
        current = self.node.features_albedo
        if not isinstance(current, torch.Tensor) or current.ndim != 3:
            raise ValueError(f"features_albedo is not a [G, N, 3] frame table: {getattr(current, 'shape', None)}")
        n = int(current.shape[1])
        if width <= n:
            return

        def _pad(table: torch.Tensor) -> torch.Tensor:
            return torch.cat([table, table[:, -1:, :].expand(-1, width - n, -1)], dim=1).contiguous()

        padded = _pad(current.data)
        kind = torch.nn.Parameter if isinstance(current, torch.nn.Parameter) else torch.nn.Buffer
        setattr(self.node, "features_albedo", kind(padded))
        if "features_albedo" in self._pending:
            self._pending["features_albedo"] = [_pad(batch) for batch in self._pending["features_albedo"]]

    def append_simready(
        self,
        track_idx: int,
        *,
        positions_local: np.ndarray,
        rotations: np.ndarray,
        scales_log: np.ndarray,
        densities_logit: np.ndarray,
        sh_dc: np.ndarray,
        sh_rest: np.ndarray,
        hidden: bool = True,
    ) -> list[int]:
        """Append one already-transformed imported SimReady cluster; returns its rows."""
        rows = make_simready_rows(
            spec=self.spec,
            positions_np=positions_local.astype(np.float32),
            rotations_np=rotations.astype(np.float32),
            scales_log_np=scales_log.astype(np.float32),
            densities_logit_np=densities_logit.astype(np.float32),
            sh_dc_np=sh_dc.astype(np.float32),
            sh_rest_np=sh_rest.astype(np.float32),
            cuboid_track_index=track_idx,
            hidden=hidden,
        )
        return self._append_rows(rows)

    def append_copied_rows(
        self,
        source_rows: np.ndarray,
        track_idx: int,
        *,
        densities_logit: Optional[np.ndarray] = None,
        frame_columns: Optional[np.ndarray] = None,
    ) -> list[int]:
        """Stage copies of captured rows (a cloned actor), retagged to ``track_idx``.

        Every per-Gaussian tensor is copied as is (positions in the source's cuboid-local frame,
        rotations, scales, SH including temporal coefficients, extra signals, the layer's sidecar
        buffers) except the cuboid id
        and, when the caller passes the trained logits, the densities (the source may already be
        hidden by an earlier edit). ``source_rows`` must address original rows; staged appends
        cannot be copied before ``flush``.

        ``frame_columns`` re-indexes a per-track-frame appearance table (``features_albedo`` of
        shape ``[G, N, 3]``): column ``k`` of a copied row takes the source row's column
        ``frame_columns[k]``. One entry per column; more entries than the table has columns
        widen it first (:meth:`widen_frame_table`).
        """
        src = np.asarray(source_rows, dtype=np.int64).reshape(-1)
        if src.size and int(src.max()) >= self.initial_n:
            raise ValueError(
                f"copied rows must be captured rows (< {self.initial_n}); got a row index of {int(src.max())}"
            )
        idx = torch.as_tensor(src, device=self.spec.device)
        n = int(idx.numel())
        rows: dict[str, torch.Tensor] = {}
        for name in self.row_names:
            current = getattr(self.node, name)
            if not isinstance(current, torch.Tensor):
                raise TypeError(f"{name}: expected tensor-like, got {type(current).__name__}")
            rows[name] = current.data.index_select(0, idx).clone()
        if frame_columns is not None:
            table = rows["features_albedo"]
            columns = torch.as_tensor(np.asarray(frame_columns, dtype=np.int64).reshape(-1), device=table.device)
            if table.ndim == 3:
                self.widen_frame_table(int(columns.numel()))
            width = int(self.node.features_albedo.shape[1]) if table.ndim == 3 else -1
            if int(columns.numel()) != width:
                raise ValueError(
                    f"frame_columns has {int(columns.numel())} entries for a features_albedo table of shape "
                    f"{tuple(table.shape)}; expected one per column of a [G, N, 3] table ({width})"
                )
            rows["features_albedo"] = table.index_select(1, columns)
        rows["gaussian_cuboid_ids"] = torch.full((n,), track_idx, device=self.spec.device, dtype=torch.int32)
        if densities_logit is not None:
            logits = np.ascontiguousarray(np.asarray(densities_logit, dtype=np.float32).reshape(-1))
            if logits.shape[0] != n:
                raise ValueError(f"densities_logit has {logits.shape[0]} rows, expected {n} (one per copied row)")
            rows["densities"] = (
                torch.from_numpy(logits)
                .to(device=self.spec.device, dtype=self.spec.dtype)
                .reshape(rows["densities"].shape)
            )
        return self._append_rows(rows)


def world_track_poses(ct: CuboidTracks, world_to_nurec) -> np.ndarray:
    """``(n_tracks, 4, 4)`` world-frame track poses (cuboid-local -> world)."""
    n_tracks = len(ct.tracks_id)
    ct_world = CuboidTracks.Ops.transform_with_frame_conversion(ct, world_to_nurec.inverse(), None)
    pose_world = np.zeros((n_tracks, 4, 4), dtype=np.float64)
    for i in range(n_tracks):
        sub = CuboidTracks.Ops.subset_from_indices(ct_world, [i])
        pose_world[i] = sub.tracks_poses.matrix()[0].detach().cpu().numpy()
    return pose_world


__all__ = [
    "HIDDEN_DENSITY_LOGIT",
    "LAYER_PARAM_NAMES",
    "CapturedSnapshot",
    "LayerAppender",
    "LayerTensorSpec",
    "euler_xyz_to_matrix",
    "layer_row_tensor_names",
    "make_simready_rows",
    "matrix_to_quat_wxyz",
    "quat_mul_wxyz",
    "replace_layer_param",
    "world_track_poses",
]
