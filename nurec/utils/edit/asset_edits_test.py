# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compile-asset-edits helpers: geometry, row building, scheduling.

Standalone and CPU-only: every helper under test takes explicit ``device``/``dtype``
or plain numpy arrays; no scene data, no GPU, no USD assets. The end-to-end bake is
covered separately (AssetEditsBaker tests with fake layer nodes; artifact-level checks
happen in the author's A/B re-bake, since a real trained USDZ cannot ship with tests).
One exception: ``WorldBindBakeTest`` exercises real ``CuboidTracks`` subset ops whose
kernels are CUDA-only, so it skips on GPU-less hosts.
"""

import math
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import numpy as np
import torch

from omegaconf import OmegaConf

from nurec.datasets.tracks import CuboidTracks
from nurec.models.composite import LayerTrackIds
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.models.gaussians.utils import write_ply_3dgs
from nurec.utils.edit.asset_edits import (
    AssetEditsBakeError,
    AssetEditsBaker,
    _element_token,
    _elements_identical,
    _match_element,
)
from nurec.utils.edit.asset_edits_schema import AssetEditsDoc, AssetEditsError
from nurec.utils.edit.layer_append import (
    HIDDEN_DENSITY_LOGIT,
    LAYER_PARAM_NAMES,
    LayerAppender,
    LayerTensorSpec,
    euler_xyz_to_matrix,
    make_simready_rows,
    matrix_to_quat_wxyz,
    quat_mul_wxyz,
    replace_layer_param,
)
from nurec.utils.edit.simready_gaussians import (
    SimReadyAsset,
    SimReadyGaussians,
    SimReadyPart,
    SimReadyVariantControl,
)
from nurec.utils.edit.variant_animation import VariantAnimation
from nurec.utils.types import FrameConversion, TrackFlags


_CPU = torch.device("cpu")
_F32 = torch.float32


def _rot_z(deg: float) -> np.ndarray:
    """A 3x3 rotation matrix of ``deg`` degrees about the Z axis."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q`` via q ⊗ (0, v) ⊗ q*."""
    qv = np.concatenate([[0.0], v])
    q_conj = q * np.array([1.0, -1.0, -1.0, -1.0])
    return quat_mul_wxyz(quat_mul_wxyz(q, qv), q_conj)[1:]


class RotationHelpersTest(unittest.TestCase):
    """The rotation helpers agree with each other and with direct matrix algebra."""

    def test_euler_convention_is_rz_ry_rx(self) -> None:
        """euler_xyz_to_matrix composes as Rz @ Ry @ Rx, so 90 deg about Z sends +x to +y."""
        np.testing.assert_allclose(euler_xyz_to_matrix(0.0, 0.0, 90.0) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
        rx, ry, rz = 10.0, 20.0, 30.0
        expected = euler_xyz_to_matrix(0, 0, rz) @ euler_xyz_to_matrix(0, ry, 0) @ euler_xyz_to_matrix(rx, 0, 0)
        np.testing.assert_allclose(euler_xyz_to_matrix(rx, ry, rz), expected, atol=1e-12)

    def test_matrix_quat_roundtrip_hits_all_branches(self) -> None:
        """matrix_to_quat_wxyz returns a unit quaternion that rotates like the source matrix in all four branches."""
        # 180-degree rotations select the m00/m11/m22-dominant branches; the generic
        # rotation takes the trace>0 branch.
        cases = [
            euler_xyz_to_matrix(180.0, 0.0, 0.0),
            euler_xyz_to_matrix(0.0, 180.0, 0.0),
            euler_xyz_to_matrix(0.0, 0.0, 180.0),
            euler_xyz_to_matrix(10.0, 20.0, 30.0),
        ]
        v = np.array([0.3, -0.5, 0.8])
        for m in cases:
            q = matrix_to_quat_wxyz(m)
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=9)
            np.testing.assert_allclose(_quat_rotate(q, v), m @ v, atol=1e-9)

    def test_quat_mul_matches_matrix_product(self) -> None:
        """quat_mul_wxyz composes rotations in the same order as the matrix product m1 @ m2."""
        m1 = euler_xyz_to_matrix(10.0, 20.0, 30.0)
        m2 = euler_xyz_to_matrix(-40.0, 5.0, 60.0)
        q = quat_mul_wxyz(matrix_to_quat_wxyz(m1), matrix_to_quat_wxyz(m2))
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_quat_rotate(q, v), m1 @ m2 @ v, atol=1e-9)


class SimreadyRowsTest(unittest.TestCase):
    """make_simready_rows lays out appended-row tensors: visibility, SH packing and column placement."""

    def _rows(self, hidden: bool, albedo_sh_dim: int = 12, specular_sh_dim: int = 45) -> dict[str, torch.Tensor]:
        """Rows for three gaussians built from deterministic inputs, with configurable SH widths."""
        n = 3
        spec = LayerTensorSpec(
            fourier_features_dim=1,
            albedo_sh_dim=albedo_sh_dim,
            specular_sh_dim=specular_sh_dim,
            extra_signal_dim=2,
            camera_extra_signal_dim=1,
            lidar_extra_signal_dim_total=3,
            scale_activation="exp",
            device=_CPU,
            dtype=_F32,
        )
        return make_simready_rows(
            spec=spec,
            positions_np=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
            rotations_np=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
            scales_log_np=np.full((n, 3), -2.0, dtype=np.float32),
            densities_logit_np=np.array([-1.0, 0.0, 2.0], dtype=np.float32),
            sh_dc_np=np.full((n, 3), 0.25, dtype=np.float32),
            sh_rest_np=np.arange(n * 15 * 3, dtype=np.float32).reshape(n, 15, 3),
            cuboid_track_index=5,
            hidden=hidden,
        )

    def test_hidden_writes_sentinel_logits(self) -> None:
        """hidden=True overwrites every density logit with the HIDDEN_DENSITY_LOGIT sentinel."""
        rows = self._rows(hidden=True)
        self.assertTrue(torch.all(rows["densities"] == HIDDEN_DENSITY_LOGIT))

    def test_visible_passes_real_logits_through(self) -> None:
        """hidden=False keeps the caller's density logits unchanged."""
        rows = self._rows(hidden=False)
        np.testing.assert_allclose(rows["densities"].numpy().reshape(-1), [-1.0, 0.0, 2.0])

    def test_sh_rest_flattened_coeff_major_and_truncated(self) -> None:
        """sh_rest flattens coefficient-major into features_specular; a smaller width keeps the leading columns."""
        rows = self._rows(hidden=True, specular_sh_dim=45)
        expected = np.arange(3 * 15 * 3, dtype=np.float32).reshape(3, -1)
        np.testing.assert_allclose(rows["features_specular"].numpy(), expected)
        truncated = self._rows(hidden=True, specular_sh_dim=10)["features_specular"]
        np.testing.assert_allclose(truncated.numpy(), expected[:, :10])

    def test_dc_written_to_first_columns_only(self) -> None:
        """The DC band fills the first three albedo columns; the remaining columns stay zero."""
        albedo = self._rows(hidden=True)["features_albedo"]
        self.assertTrue(torch.all(albedo[:, :3] == 0.25))
        self.assertTrue(torch.all(albedo[:, 3:] == 0.0))


class ReplaceLayerParamTest(unittest.TestCase):
    """replace_layer_param swaps in grown tensors while preserving how they are registered."""

    def test_parameter_and_buffer_both_grow(self) -> None:
        """Appending grows rows along dim 0 while keeping Parameter vs Buffer registration and dtype."""
        node = torch.nn.Module()
        node.positions = torch.nn.Parameter(torch.zeros((2, 3)))
        node.gaussian_cuboid_ids = torch.nn.Buffer(torch.zeros((2,), dtype=torch.int32))
        replace_layer_param(node, "positions", torch.ones((3, 3)))
        replace_layer_param(node, "gaussian_cuboid_ids", torch.ones((3,), dtype=torch.int32))
        self.assertIsInstance(node.positions, torch.nn.Parameter)
        self.assertEqual(node.positions.shape, (5, 3))
        self.assertNotIsInstance(node.gaussian_cuboid_ids, torch.nn.Parameter)
        self.assertEqual(node.gaussian_cuboid_ids.shape, (5,))
        self.assertEqual(node.gaussian_cuboid_ids.dtype, torch.int32)

    def test_non_tensor_attribute_raises(self) -> None:
        """Targeting a non-tensor attribute raises TypeError instead of silently replacing it."""
        node = torch.nn.Module()
        node.name = "layer"  # type: ignore[assignment]
        with self.assertRaises(TypeError):
            replace_layer_param(node, "name", torch.zeros((1,)))


class MatchElementTest(unittest.TestCase):
    """_match_element resolves prim paths against root-relative element keys."""

    def test_exact_and_subtree_matches(self) -> None:
        """A key matches its exact prim and any prim inside that subtree."""
        keys = ["AssemblyL", "AssemblyR/part_0"]
        self.assertEqual(_match_element(keys, "/Car/AssemblyL", "/Car"), "AssemblyL")
        self.assertEqual(_match_element(keys, "/Car/AssemblyL/part_2", "/Car"), "AssemblyL")
        self.assertEqual(_match_element(keys, "/Car/AssemblyR/part_0", "/Car"), "AssemblyR/part_0")

    def test_no_match_and_foreign_root(self) -> None:
        """Unlisted prims yield None, and prims outside the root prefix never relativise into a match."""
        self.assertIsNone(_match_element(["AssemblyL"], "/Car/Body", "/Car"))
        # Paths outside the root prefix are compared as-is (no relativisation).
        self.assertIsNone(_match_element(["AssemblyL"], "/Other/AssemblyL", "/Car"))

    def test_nested_key_wins_over_its_ancestor(self) -> None:
        """When an asset animates an assembly AND a panel inside it, both are keys.

        The part belongs to the nested element; attributing it to the ancestor would fold
        its rows into the parent's cluster while the manifest still advertises the nested
        aspect, leaving that aspect with no geometry.
        """
        keys = sorted(["AssemblyL", "AssemblyL/part_2"])
        self.assertEqual(_match_element(keys, "/Car/AssemblyL/part_2/Splats", "/Car"), "AssemblyL/part_2")
        # A sibling under the same ancestor still resolves to the ancestor.
        self.assertEqual(_match_element(keys, "/Car/AssemblyL/other/Splats", "/Car"), "AssemblyL")


def _payload(seed: int, opacity_shift: float = 0.0) -> SimReadyGaussians:
    """A deterministic random 4-gaussian payload; ``opacity_shift`` offsets its density logits."""
    rng = np.random.default_rng(seed)
    n = 4
    return SimReadyGaussians(
        positions=rng.normal(size=(n, 3)),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        scales_log=rng.normal(size=(n, 3)),
        densities_logit=rng.normal(size=n) + opacity_shift,
        sh_dc=rng.normal(size=(n, 3)),
        sh_rest=rng.normal(size=(n, 15, 3)),
        sh_degree=3,
        up_axis="Z",
        meters_per_unit=1.0,
    )


class ElementsIdenticalTest(unittest.TestCase):
    """_elements_identical spots any change in per-state part membership or gaussian data."""

    _STATES = ["off", "sequence"]

    def test_identical_payloads_across_states(self) -> None:
        """States whose parts carry identical gaussians at the same prim path compare equal."""
        part_a = SimpleNamespace(prim_path="/Car/Detail", gaussians=_payload(1))
        part_b = SimpleNamespace(prim_path="/Car/Detail", gaussians=_payload(1))
        self.assertTrue(_elements_identical({"off": [part_a], "sequence": [part_b]}, self._STATES))

    def test_opacity_only_change_is_a_difference(self) -> None:
        """A shift in density logits alone makes states compare unequal."""
        part_a = SimpleNamespace(prim_path="/Car/Detail", gaussians=_payload(1))
        part_b = SimpleNamespace(prim_path="/Car/Detail", gaussians=_payload(1, opacity_shift=0.5))
        self.assertFalse(_elements_identical({"off": [part_a], "sequence": [part_b]}, self._STATES))

    def test_membership_change_is_a_difference(self) -> None:
        """The same gaussians under a different prim path make states compare unequal."""
        part_a = SimpleNamespace(prim_path="/Car/Detail", gaussians=_payload(1))
        part_b = SimpleNamespace(prim_path="/Car/OtherPart", gaussians=_payload(1))
        self.assertFalse(_elements_identical({"off": [part_a], "sequence": [part_b]}, self._STATES))

    def test_element_missing_from_a_state_is_not_shared(self) -> None:
        """An element that contributes in only SOME states must stay state-scoped.

        With one state it is vacuously "identical in every state it appears in", but a
        state-agnostic cluster has no way to express "hidden in the state where this
        element does not exist" — so collapsing it would move correctness out of the
        baked geometry and into the sender.
        """
        part = SimpleNamespace(prim_path="/Car/Detail", gaussians=_payload(1))
        self.assertFalse(_elements_identical({"sequence": [part]}, self._STATES))


class _FakeCuboidTracks:
    """Duck-typed CuboidTracks: only the attributes the baker reads directly."""

    def __init__(self, labels: list[str], dims: tuple[float, float, float] = (8.0, 3.0, 3.0)) -> None:
        """One track per label, string ids counting up from "100", all sharing the same cuboid dims."""
        self.tracks_id = [str(100 + i) for i in range(len(labels))]
        self.tracks_label_class = labels
        self.cuboids_dims = torch.tensor([list(dims)] * len(labels), dtype=torch.float64)


class _FakeLayerNode(torch.nn.Module):
    """A layer node with real CPU Parameters/Buffers for every appended tensor.

    Real registered Parameters/Buffers matter: ``replace_layer_param`` dispatches on
    ``torch.nn.Parameter`` vs ``torch.nn.Buffer`` when concatenating appended rows.
    """

    def __init__(self, n0: int, ct: _FakeCuboidTracks, cuboid_ids: list[int] | None = None) -> None:
        """Register tensors for ``n0`` gaussians, optionally assigning each row a cuboid track id."""
        super().__init__()
        self.positions = torch.nn.Parameter(torch.linspace(-1.0, 1.0, n0 * 3).reshape(n0, 3))
        rotations = torch.zeros(n0, 4)
        rotations[:, 0] = 1.0
        self.rotations = torch.nn.Parameter(rotations)
        self.scales = torch.nn.Parameter(torch.full((n0, 3), -2.0))
        self.densities = torch.nn.Parameter(torch.zeros(n0, 1))
        self.extra_signal = torch.nn.Parameter(torch.zeros(n0, 2))
        self.camera_extra_signal = torch.nn.Parameter(torch.zeros(n0, 1))
        self.lidar_extra_signal = torch.nn.Parameter(torch.zeros(n0, 3))
        # Real layers keep only the DC band in features_albedo (higher-order SH lives
        # in features_specular).
        self.features_albedo = torch.nn.Parameter(torch.zeros(n0, 3))
        self.features_specular = torch.nn.Parameter(torch.zeros(n0, 45))
        self.gaussian_cuboid_ids = torch.nn.Buffer(
            torch.tensor(cuboid_ids if cuboid_ids is not None else [0] * n0, dtype=torch.int32)
        )
        self.cuboid_tracks = ct
        self.config = SimpleNamespace(
            particle=SimpleNamespace(extra_signal_dim=2, camera_extra_signal_dim=1, lidar_extra_signal_dim=1),
            scale_activation="exp",
        )

    def get_num_gaussians(self) -> int:
        """The live row count, reflecting any appended rows."""
        return int(self.positions.shape[0])

    def get_albedo_sh_dim(self) -> int:
        """A DC-only albedo width of 3, matching real layers."""
        return 3

    def get_specular_sh_dim(self) -> int:
        """The full degree-3 specular width of 45 columns."""
        return 45

    def get_lidar_extra_signal_sph_dim(self) -> int:
        """A lidar extra-signal width of 3, matching the registered parameter."""
        return 3

    def _validate_fields(self) -> None:
        """No-op stand-in for the real layer's post-append consistency check."""
        pass


def _fake_composite(layer_id: str, node: _FakeLayerNode) -> GaussiansComposite:
    """A duck-typed GaussiansComposite holding a single fake layer node."""
    fake = SimpleNamespace(
        gaussians_nodes={layer_id: node},
        config=SimpleNamespace(layers={layer_id: SimpleNamespace(fourier_features_dim=1)}),
    )
    return cast(GaussiansComposite, fake)


def _identity_poses(n_tracks: int) -> np.ndarray:
    """Identity 4x4 world poses, one per track."""
    return np.tile(np.eye(4), (n_tracks, 1, 1))


class LayerAppenderFlushTest(unittest.TestCase):
    """Staged appends + ONE deferred flush reproduce the eager per-part cat path exactly.

    The manifest's on_indices come from the appender's pure row arithmetic
    (``initial_n`` + rows appended so far), so the staged path must yield identical
    row ranges AND byte-identical tensors to concatenating each part as it arrives.
    """

    _N0 = 6
    _PARTS = ((7, True), (8, False), (9, True))  # (payload seed, hidden)

    def _append_three_parts(self, flush_per_part: bool) -> tuple[_FakeLayerNode, list[list[int]]]:
        """Append three deterministic parts; flushing per part IS the old sequential-cat path."""
        node = _FakeLayerNode(self._N0, _FakeCuboidTracks(labels=["car"]))
        layer = cast(RigidGaussianModel, node)
        appender = LayerAppender(layer, LayerTensorSpec.from_layer(layer, fourier_features_dim=1))
        indices: list[list[int]] = []
        for seed, hidden in self._PARTS:
            g = _payload(seed)
            indices.append(
                appender.append_simready(
                    0,
                    positions_local=g.positions,
                    rotations=g.rotations,
                    scales_log=g.scales_log,
                    densities_logit=g.densities_logit,
                    sh_dc=g.sh_dc,
                    sh_rest=g.sh_rest,
                    hidden=hidden,
                )
            )
            if flush_per_part:
                appender.flush()
        appender.flush()
        return node, indices

    def test_single_flush_matches_sequential_cats_and_row_ranges(self) -> None:
        """One deferred flush yields the same tensors and row indices as per-part cats."""
        node_eager, idx_eager = self._append_three_parts(flush_per_part=True)
        node_staged, idx_staged = self._append_three_parts(flush_per_part=False)
        expected = [list(range(self._N0 + 4 * i, self._N0 + 4 * (i + 1))) for i in range(3)]
        self.assertEqual(idx_eager, expected)
        self.assertEqual(idx_staged, expected)
        for name in LAYER_PARAM_NAMES:
            eager, staged = getattr(node_eager, name), getattr(node_staged, name)
            self.assertTrue(torch.equal(eager.data, staged.data), name)
            self.assertEqual(isinstance(eager, torch.nn.Parameter), isinstance(staged, torch.nn.Parameter), name)

    def test_bad_batches_still_fail_at_append_time(self) -> None:
        """Staging validates immediately, before any flush: an off-contract key set raises,
        a mismatched row count raises, and a non-tensor target raises."""
        node = _FakeLayerNode(self._N0, _FakeCuboidTracks(labels=["car"]))
        layer = cast(RigidGaussianModel, node)
        appender = LayerAppender(layer, LayerTensorSpec.from_layer(layer, fourier_features_dim=1))
        with self.assertRaisesRegex(ValueError, "LAYER_PARAM_NAMES"):
            appender._append_rows({"config": torch.zeros((1, 3))})
        good = {name: torch.zeros((2, 3)) for name in LAYER_PARAM_NAMES}
        bad_count = dict(good)
        bad_count["densities"] = torch.zeros((3, 1))
        with self.assertRaisesRegex(ValueError, "densities"):
            appender._append_rows(bad_count)
        del node._parameters["positions"]  # re-register the target as a plain non-tensor attr
        node.positions = "not-a-tensor"  # type: ignore[assignment]
        with self.assertRaises(TypeError):
            appender._append_rows(good)


class AssetEditsBakerTest(unittest.TestCase):
    """Full composed-asset bakes via an injected asset loader."""

    @staticmethod
    def _asset() -> SimReadyAsset:
        """A two-part static asset plus a panelState variant whose off/on states carry distinct gaussians."""
        body = SimReadyPart(prim_path="/Car/Body", gaussians=_payload(2))
        wheel = SimReadyPart(prim_path="/Car/Wheel_FL", gaussians=_payload(3))
        panels_off = SimReadyPart(prim_path="/Car/Panels", gaussians=_payload(4))
        panels_on = SimReadyPart(prim_path="/Car/Panels", gaussians=_payload(5))
        control = SimReadyVariantControl(
            variant_set="panelState",
            states=["off", "on"],
            default_state="off",
            parts_by_state={"off": [panels_off], "on": [panels_on]},
        )
        return SimReadyAsset(
            asset_path="car.usd",
            root_prim_path="/Car",
            static_parts=[body, wheel],
            controls={"panelState": control},
            variant_sets={"panelState": ["off", "on"]},
            up_axis="Z",
            meters_per_unit=1.0,
        )

    def _bake(self, n0: int = 6) -> tuple[AssetEditsBaker, _FakeLayerNode, SimReadyAsset]:
        """Bake one replace-original edit bound to track 0; returns the baker, mutated node and source asset."""
        ct = _FakeCuboidTracks(labels=["car"], dims=(8.0, 3.0, 3.0))
        # Half the captured rows ride track 0 (the bound track), half another track.
        node = _FakeLayerNode(n0, ct, cuboid_ids=[0] * (n0 // 2) + [1] * (n0 - n0 // 2))
        asset = self._asset()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "car.usd").touch()
            doc = AssetEditsDoc.from_dict(
                {
                    "edits": [
                        {
                            "id": "truck_1",
                            "bind": {"nre_track_id": 0},
                            "asset": "car.usd",
                            "replace_original": True,
                            "controls": {"panelState": {"default": "off", "controllable": True}},
                        }
                    ]
                }
            )
            baker = AssetEditsBaker(
                _fake_composite("dynamic_rigids", node),
                None,
                Path(tmp),
                asset_loader=lambda path, control_sets=None, root_prim_path=None: asset,
            )
            with mock.patch("nurec.utils.edit.asset_edits.world_track_poses", return_value=_identity_poses(1)):
                baker.bake({"dynamic_rigids": doc.edits})
        return baker, node, asset

    def test_parts_bake_with_default_visible_and_others_hidden(self) -> None:
        """Parts append after the originals; the default state keeps real logits, other states get the sentinel."""
        baker, node, asset = self._bake(n0=6)
        # Part order: static (body, wheel), then control states off, on — 4 rows each.
        self.assertEqual(node.get_num_gaussians(), 6 + 16)
        body_logits = np.float32(asset.static_parts[0].gaussians.densities_logit.reshape(-1))
        np.testing.assert_allclose(node.densities.data[6:10].numpy().reshape(-1), body_logits, rtol=1e-6)
        off_logits = np.float32(asset.controls["panelState"].parts_by_state["off"][0].gaussians.densities_logit)
        np.testing.assert_allclose(
            node.densities.data[14:18].numpy().reshape(-1), off_logits.reshape(-1), rtol=1e-6
        )  # default state visible
        self.assertTrue(torch.all(node.densities.data[18:22] == HIDDEN_DENSITY_LOGIT))  # non-default hidden

    def test_one_cluster_per_state_with_real_alphas(self) -> None:
        """Each state yields one cluster pointing at its appended rows, with sigmoid(logit) on-opacities."""
        baker, _, asset = self._bake()
        self.assertEqual([b.aspect for b in baker.baked_clusters], ["panelstate:off", "panelstate:on"])
        on_cluster = baker.baked_clusters[1]
        self.assertEqual(on_cluster.on_indices, list(range(18, 22)))
        logits = asset.controls["panelState"].parts_by_state["on"][0].gaussians.densities_logit.reshape(-1)
        np.testing.assert_allclose(on_cluster.on_opacities, 1.0 / (1.0 + np.exp(-logits)), rtol=1e-6)

    def test_control_animation_table_emitted(self) -> None:
        """One animation entry is emitted with the control name, states, default and the bound track's string id."""
        baker, _, _ = self._bake()
        (anim,) = baker.control_animations
        self.assertEqual((anim.control, anim.states, anim.default_state), ("panelstate", ["off", "on"], "off"))
        self.assertEqual(anim.track_id_str, "100")

    def test_replace_original_hides_only_bound_track_rows(self) -> None:
        """replace_original sentinels only original rows on the bound track; other tracks stay untouched."""
        _, node, _ = self._bake(n0=6)
        self.assertTrue(torch.all(node.densities.data[0:3] == HIDDEN_DENSITY_LOGIT))  # track-0 originals
        self.assertTrue(torch.all(node.densities.data[3:6] == 0.0))  # other track untouched

    def test_asset_record_fit_length_scale(self) -> None:
        """fit_length derives scale from the DEFAULT-VISIBLE parts' X extent vs the cuboid
        length (hidden state copies are excluded so they cannot skew the shared fit)."""
        baker, _, asset = self._bake()
        (record,) = baker.asset_records
        visible_pos = np.concatenate(
            [asset.static_parts[0].gaussians.positions, asset.static_parts[1].gaussians.positions]
            + [p.gaussians.positions for p in asset.controls["panelState"].parts_by_state["off"]]
        )
        expected_scale = 8.0 / float(visible_pos[:, 0].max() - visible_pos[:, 0].min())
        self.assertAlmostEqual(record["placement"]["scale"], expected_scale, places=9)
        self.assertEqual(record["original_layer_size"], 6)
        self.assertEqual(len(record["parts"]), 4)

    def test_unknown_control_fails_against_asset(self) -> None:
        """Requesting a control that is not a variantSet on the asset aborts the bake."""
        ct = _FakeCuboidTracks(labels=["car"])
        node = _FakeLayerNode(4, ct)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "car.usd").touch()
            doc = AssetEditsDoc.from_dict(
                {
                    "edits": [
                        {
                            "id": "truck_1",
                            "bind": {"nre_track_id": 0},
                            "asset": "car.usd",
                            "controls": {"detailState": {"controllable": True}},
                        }
                    ]
                }
            )
            baker = AssetEditsBaker(
                _fake_composite("dynamic_rigids", node),
                None,
                Path(tmp),
                asset_loader=lambda path, control_sets=None, root_prim_path=None: self._asset(),
            )
            with mock.patch("nurec.utils.edit.asset_edits.world_track_poses", return_value=_identity_poses(1)):
                with self.assertRaisesRegex(AssetEditsBakeError, "not a variantSet"):
                    baker.bake({"dynamic_rigids": doc.edits})


def _write_prop_ply(path: Path) -> int:
    """A Y-up 3DGS PLY box: 1 m wide (x), 2 m tall (y, base at 0), 0.5 m deep (z). Returns its splat count."""
    pos = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (0.0, 2.0) for z in (-0.25, 0.25)])
    write_ply_3dgs(
        path,
        positions=torch.tensor(pos, dtype=torch.float32),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(pos)),
        scales=torch.full((len(pos), 3), -2.0),
        densities=torch.zeros(len(pos)),
        features_albedo=torch.ones(len(pos), 3),
    )
    return len(pos)


class PlyPropBakeTest(unittest.TestCase):
    """A ``.ply`` edit reaches the real PLY loader through the suffix dispatch (no injected loader),
    bakes as a plain static part, and is placed in the Z-up frame the file is rotated into."""

    def _bake(self, edit: dict, n0: int = 4) -> tuple[AssetEditsBaker, _FakeLayerNode, int]:
        """Bake one edit against a single 3 x 1 x 0.5 m cuboid track; returns (baker, node, splat count)."""
        ct = _FakeCuboidTracks(labels=["cone"], dims=(3.0, 1.0, 0.5))
        node = _FakeLayerNode(n0, ct, cuboid_ids=[0] * n0)
        with tempfile.TemporaryDirectory() as tmp:
            n_ply = _write_prop_ply(Path(tmp) / "cone.ply")
            doc = AssetEditsDoc.from_dict({"edits": [edit]})
            baker = AssetEditsBaker(_fake_composite("dynamic_rigids", node), None, Path(tmp))
            with mock.patch("nurec.utils.edit.asset_edits.world_track_poses", return_value=_identity_poses(1)):
                baker.bake({"dynamic_rigids": doc.edits})
        return baker, node, n_ply

    def test_static_ply_bakes_visible_rows_and_no_clusters(self) -> None:
        """Every splat lands as a visible row on the bound track; nothing is addressable per frame."""
        baker, node, n_ply = self._bake({"id": "cone_1", "bind": {"nre_track_id": 0}, "asset": "cone.ply"})
        self.assertEqual(node.get_num_gaussians(), 4 + n_ply)
        self.assertTrue((node.densities[4:] > HIDDEN_DENSITY_LOGIT).all())
        self.assertTrue((node.gaussian_cuboid_ids[4:] == 0).all())
        self.assertEqual(baker.baked_clusters, [])
        self.assertEqual(baker.control_animations, [])
        rec = baker.asset_records[0]
        self.assertTrue(rec["asset"].endswith("cone.ply"))
        self.assertEqual((rec["root_prim"], rec["up_axis"], rec["controls"]), ("", "Z", {}))
        self.assertEqual([(p["prim_path"], p["role"]) for p in rec["parts"]], [("cone.ply", "static")])

    def test_fit_length_and_height_use_the_zup_frame(self) -> None:
        """fit_length scales the file's 1 m x-extent onto the 3 m cuboid length, and the file's 2 m
        y-extent stands up as height: the rows span 6 m along z, not 1.5 m."""
        baker, node, _ = self._bake({"id": "cone_1", "bind": {"nre_track_id": 0}, "asset": "cone.ply"})
        self.assertAlmostEqual(baker.asset_records[0]["placement"]["scale"], 3.0, places=6)
        appended = node.positions[4:].detach()
        extent = (appended.max(dim=0).values - appended.min(dim=0).values).tolist()
        np.testing.assert_allclose(extent, [3.0, 1.5, 6.0], atol=1e-5)

    def test_controls_on_a_ply_never_reach_the_bake(self) -> None:
        """The schema refuses controls on a .ply before any file is opened."""
        with self.assertRaisesRegex(AssetEditsError, "takes no controls"):
            AssetEditsDoc.from_dict(
                {
                    "edits": [
                        {
                            "id": "cone_1",
                            "bind": {"nre_track_id": 0},
                            "asset": "cone.ply",
                            "controls": {"appearance": {"default": "off", "controllable": True}},
                        }
                    ]
                }
            )


def _box_payload(dx: float = 0.5, dy: float = 2.0, dz: float = 0.6) -> SimReadyGaussians:
    """Eight gaussians on the corners of a box centered in x/y with its base at z=0."""
    positions = np.array([[x, y, z] for x in (-dx / 2, dx / 2) for y in (-dy / 2, dy / 2) for z in (0.0, dz)])
    n = len(positions)
    return SimReadyGaussians(
        positions=positions,
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        scales_log=np.zeros((n, 3)),
        densities_logit=np.zeros(n),
        sh_dc=np.zeros((n, 3)),
        sh_rest=np.zeros((n, 15, 3)),
        sh_degree=3,
        up_axis="Z",
        meters_per_unit=1.0,
    )


def _prop_asset(prim_path: str = "/Body", **box_kwargs: float) -> SimReadyAsset:
    """A bare controls-free prop (a single static splat part), like a traffic barrier."""
    part = SimReadyPart(prim_path=prim_path, gaussians=_box_payload(**box_kwargs))
    return SimReadyAsset(
        asset_path="Body.usd",
        root_prim_path=prim_path,
        static_parts=[part],
        controls={},
        variant_sets={},
        up_axis="Z",
        meters_per_unit=1.0,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CuboidTracks subset ops dispatch to CUDA-only packed_ops kernels")
class WorldBindBakeTest(unittest.TestCase):
    """World binds bake against REAL CuboidTracks + LayerTrackIds (CUDA packed ops;
    the layer node's per-gaussian tensors stay on CPU like the rest of this file).

    The pre-pass is exactly what these tests pin down: synthetic-track synthesis,
    world->NuRec conversion, layer re-partition, and the index-agreement guarantee the
    persisted artifact depends on.
    """

    _LAYER = "dynamic_rigids"
    _WINDOW = (10_000_000, 20_000_000)
    _PAD = 2_000_000  # keep in sync with asset_edits._WORLD_TRACK_PAD_US

    @staticmethod
    def _make_tracks(entries: list[tuple[str, str, TrackFlags]]) -> CuboidTracks:
        """One 2-keyframe identity-pose track per (id, label, flags) entry."""
        n = len(entries)
        poses = [np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)) for _ in range(n)]
        ts = [np.array([10_000_000, 20_000_000], dtype=np.int64)] * n
        return CuboidTracks.Factory.from_numpy(
            tracks_id=[e[0] for e in entries],
            tracks_poses=poses,
            tracks_timestamps_us=ts,
            tracks_label_class=[e[1] for e in entries],
            tracks_flags=[e[2] for e in entries],
            cuboids_dims=[np.array([4.0, 2.0, 1.5])] * n,
            device=torch.device("cuda"),
        )

    def _composite(self, node: _FakeLayerNode) -> GaussiansComposite:
        """A composite whose global track list holds one layer track and one foreign track."""
        global_ct = self._make_tracks(
            [("car", "automobile", TrackFlags.DYNAMIC), ("tree", "vegetation", TrackFlags.NONE)]
        )
        config = OmegaConf.create({"label_classes": ["automobile", "protruding_object"], "is_dynamic": True})
        ids = LayerTrackIds(config=config)
        ids.initialize_from_tracks(global_ct)
        node.cuboid_tracks = cast(_FakeCuboidTracks, ids.get_layer_tracks(global_ct))
        fake = SimpleNamespace(
            cuboid_tracks=global_ct,
            gaussians_nodes={self._LAYER: node},
            obj_track_ids={self._LAYER: ids},
            config=SimpleNamespace(layers={self._LAYER: SimpleNamespace(fourier_features_dim=1)}),
        )
        return cast(GaussiansComposite, fake)

    def _bake(self, edits: list[dict], n0: int = 4, world_to_nurec: FrameConversion | None = None) -> tuple:
        """Bake ``edits`` against a 1-layer-track composite; returns (baker, node, composite)."""
        node = _FakeLayerNode(n0, cast(_FakeCuboidTracks, None), cuboid_ids=[0] * n0)
        composite = self._composite(node)
        if world_to_nurec is None:
            world_to_nurec = FrameConversion.from_origin_scale_axis(np.zeros(3), 1.0, [0, 1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Body.usd").touch()
            (Path(tmp) / "car.usd").touch()
            doc = AssetEditsDoc.from_dict({"edits": edits})
            baker = AssetEditsBaker(
                composite,
                world_to_nurec,
                Path(tmp),
                asset_loader=lambda path, control_sets=None, root_prim_path=None: _prop_asset(),
                clip_window_us=self._WINDOW,
            )
            baker.bake({self._LAYER: doc.edits})
        return baker, node, composite

    @staticmethod
    def _world_edit(edit_id: str = "barrier_1", **bind_over: object) -> dict:
        bind: dict = {"type": "world", "position": [100.0, -3.5, 0.0], "yaw_deg": 90.0}
        bind.update(bind_over)
        return {"id": edit_id, "bind": bind, "asset": "Body.usd"}

    def test_synthetic_track_appends_at_layer_end(self) -> None:
        """The synthetic track lands at the end of both the global and the layer track lists."""
        baker, node, composite = self._bake([self._world_edit()])
        self.assertEqual(composite.cuboid_tracks.tracks_id, ["car", "tree", "prop:barrier_1"])
        self.assertEqual(node.cuboid_tracks.tracks_id, ["car", "prop:barrier_1"])
        (synth,) = baker.synthetic_tracks
        self.assertEqual((synth.nurec_track_id, synth.global_index), (1, 2))
        self.assertEqual(synth.track_id_str, "prop:barrier_1")
        self.assertEqual(synth.timestamps_us, (self._WINDOW[0] - self._PAD, self._WINDOW[1] + self._PAD))

    def test_rows_tagged_with_synthetic_track_index(self) -> None:
        """Appended prop rows carry the synthetic track's layer-local index as cuboid id."""
        _, node, _ = self._bake([self._world_edit()], n0=4)
        self.assertEqual(node.get_num_gaussians(), 4 + 8)
        self.assertTrue(torch.all(node.gaussian_cuboid_ids[4:] == 1))

    def test_dims_derived_from_placed_bbox_and_anchor_base_lifts_pose(self) -> None:
        """Derived dims equal the placement-rotated asset bbox and `anchor: base` puts
        the cuboid center half a height above the authored position."""
        baker, _, _ = self._bake([self._world_edit()])
        (synth,) = baker.synthetic_tracks
        np.testing.assert_allclose(synth.dims_world, [0.5, 2.0, 0.6], atol=1e-6)
        # bind yaw does not change dims (it rotates the whole cuboid in world).
        np.testing.assert_allclose(synth.pose_tquat_world[0][:3], [100.0, -3.5, 0.3], atol=1e-6)

    def test_ply_prop_dims_come_from_the_zup_bbox(self) -> None:
        """A .ply world bind (real loader, no injection) derives its cuboid from the file rotated to
        Z-up — 2 m tall, not 2 m deep — and anchor: base lifts the pose by half that height."""
        node = _FakeLayerNode(4, cast(_FakeCuboidTracks, None), cuboid_ids=[0] * 4)
        composite = self._composite(node)
        with tempfile.TemporaryDirectory() as tmp:
            n_ply = _write_prop_ply(Path(tmp) / "cone.ply")
            doc = AssetEditsDoc.from_dict(
                {
                    "edits": [
                        {"id": "cone_1", "bind": {"type": "world", "position": [100.0, -3.5, 0.0]}, "asset": "cone.ply"}
                    ]
                }
            )
            baker = AssetEditsBaker(
                composite,
                FrameConversion.from_origin_scale_axis(np.zeros(3), 1.0, [0, 1, 2]),
                Path(tmp),
                clip_window_us=self._WINDOW,
            )
            baker.bake({self._LAYER: doc.edits})
        synth = baker.synthetic_tracks[0]
        np.testing.assert_allclose(synth.dims_world, [1.0, 0.5, 2.0], atol=1e-6)
        np.testing.assert_allclose(synth.pose_tquat_world[0][:3], [100.0, -3.5, 1.0], atol=1e-6)
        self.assertEqual(node.get_num_gaussians(), 4 + n_ply)
        self.assertEqual(baker.baked_clusters, [])

    def test_placement_rotation_rotates_derived_dims(self) -> None:
        """A placement yaw of 90 swaps the derived x/y dims (the asset turns inside its cuboid)."""
        edit = self._world_edit()
        edit["placement"] = {"yaw_deg": 90.0}
        baker, _, _ = self._bake([edit])
        (synth,) = baker.synthetic_tracks
        np.testing.assert_allclose(synth.dims_world, [2.0, 0.5, 0.6], atol=1e-6)

    def test_world_to_nurec_conversion_applied(self) -> None:
        """The persisted NuRec pose is the world pose through the frame conversion."""
        conv = FrameConversion.from_origin_scale_axis(np.array([10.0, 5.0, 0.0]), 1.0, [0, 1, 2])
        baker, _, composite = self._bake([self._world_edit()], world_to_nurec=conv)
        (synth,) = baker.synthetic_tracks
        np.testing.assert_allclose(synth.pose_tquat_world[0][:3], [100.0, -3.5, 0.3], atol=1e-5)
        np.testing.assert_allclose(synth.pose_tquat_nurec[0][:3], [90.0, -8.5, 0.3], atol=1e-5)
        # The stored global tracks are NuRec-frame.
        prop_idx = composite.cuboid_tracks.tracks_id.index("prop:barrier_1")
        sub = CuboidTracks.Ops.subset_from_indices(composite.cuboid_tracks, [prop_idx])
        np.testing.assert_allclose(
            sub.tracks_poses.matrix()[0][:3, 3].detach().cpu().numpy(), [90.0, -8.5, 0.3], atol=1e-5
        )

    def test_prop_rows_rest_on_cuboid_floor(self) -> None:
        """align: bbox centers the prop in its own cuboid with the base on the floor."""
        _, node, _ = self._bake([self._world_edit()], n0=4)
        appended = node.positions.data[4:].detach().numpy()
        np.testing.assert_allclose(appended[:, 2].min(), -0.3, atol=1e-6)  # dz/2 below center
        np.testing.assert_allclose(appended[:, 2].max(), 0.3, atol=1e-6)
        np.testing.assert_allclose(appended[:, :2].mean(axis=0), [0.0, 0.0], atol=1e-6)

    def test_mixed_document_assigns_stable_indices(self) -> None:
        """nurec_track edits keep their indices; multiple world edits get old_n, old_n+1, ..."""
        edits = [
            {"id": "truck_1", "bind": {"nre_track_id": 0}, "asset": "car.usd", "replace_original": True},
            self._world_edit("barrier_1"),
            self._world_edit("barrier_2", position=[104.0, -3.5, 0.0]),
        ]
        baker, node, composite = self._bake(edits, n0=4)
        self.assertEqual(node.cuboid_tracks.tracks_id, ["car", "prop:barrier_1", "prop:barrier_2"])
        by_id = {r["edit_id"]: r for r in baker.asset_records}
        self.assertEqual(by_id["truck_1"]["nurec_track_id"], 0)
        self.assertEqual(by_id["barrier_1"]["nurec_track_id"], 1)
        self.assertEqual(by_id["barrier_2"]["nurec_track_id"], 2)
        self.assertEqual(by_id["barrier_2"]["bind"]["track_id"], "prop:barrier_2")
        self.assertEqual(by_id["truck_1"]["bind"], {"type": "nurec_track", "nurec_track_id": 0})
        # Rows: 4 original (track 0) + 8 per edit, tagged 0 / 1 / 2 in document order.
        ids = node.gaussian_cuboid_ids.numpy()
        self.assertEqual(list(ids[4:12]), [0] * 8)
        self.assertEqual(list(ids[12:20]), [1] * 8)
        self.assertEqual(list(ids[20:28]), [2] * 8)

    def test_track_id_collision_with_artifact_fails(self) -> None:
        """A world bind claiming an existing artifact track id aborts the bake."""
        with self.assertRaisesRegex(AssetEditsBakeError, "collides with an existing track"):
            self._bake([self._world_edit(track_id="car")])

    def test_unroutable_label_class_fails(self) -> None:
        """A label_class no layer's filter accepts fails loud with the capture summary."""
        with self.assertRaisesRegex(AssetEditsBakeError, "captured by"):
            self._bake([self._world_edit(label_class="unknown_thing")])

    def test_missing_clip_window_fails(self) -> None:
        """World binds without the artifact clip window abort instead of guessing timestamps."""
        node = _FakeLayerNode(4, cast(_FakeCuboidTracks, None), cuboid_ids=[0] * 4)
        composite = self._composite(node)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Body.usd").touch()
            doc = AssetEditsDoc.from_dict({"edits": [self._world_edit()]})
            baker = AssetEditsBaker(
                composite,
                FrameConversion.from_origin_scale_axis(np.zeros(3), 1.0, [0, 1, 2]),
                Path(tmp),
                asset_loader=lambda path, control_sets=None, root_prim_path=None: _prop_asset(),
            )
            with self.assertRaisesRegex(AssetEditsBakeError, "clip_window_us"):
                baker.bake({self._LAYER: doc.edits})


class ElementWindowBakeTest(unittest.TestCase):
    """The element-window bake branch of ``_collect_parts`` and its manifest contract.

    Animated controls group variant parts by ELEMENT (the compiler-extracted
    ``VariantAnimation`` windows/members tables). The baker dedupes elements whose
    payload is identical in every state — baked ONCE with role "element" under the
    state-agnostic aspect ``"<control>:shared:<key>"`` — while state-unique payloads keep
    one state-scoped copy per state. These tests pin that producer behavior: grouping,
    baked-in default visibility, and the manifest's row accounting against the fake
    layer's grown tensors.
    """

    _N0 = 5

    @staticmethod
    def _asset(parts_by_state: dict[str, list[SimReadyPart]]) -> SimReadyAsset:
        """A controls-only asset whose surfaceState variant swaps ``parts_by_state``."""
        control = SimReadyVariantControl(
            variant_set="surfaceState",
            states=["off", "sequence"],
            default_state="off",
            parts_by_state=parts_by_state,
        )
        return SimReadyAsset(
            asset_path="car.usd",
            root_prim_path="/Car",
            static_parts=[],
            controls={"surfaceState": control},
            variant_sets={"surfaceState": ["off", "sequence"]},
            up_axis="Z",
            meters_per_unit=1.0,
        )

    def _bake(
        self,
        asset: SimReadyAsset,
        element_windows: dict | None,
        element_members: dict | None = None,
    ) -> tuple[AssetEditsBaker, _FakeLayerNode, AssetEditsDoc]:
        """Bake one controllable surfaceState edit with compiler-extracted element tables."""
        node = _FakeLayerNode(self._N0, _FakeCuboidTracks(labels=["car"]), cuboid_ids=[0] * self._N0)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "car.usd").touch()
            doc = AssetEditsDoc.from_dict(
                {
                    "edits": [
                        {
                            "id": "truck_1",
                            "bind": {"nre_track_id": 0},
                            "asset": "car.usd",
                            "controls": {"surfaceState": {"default": "off", "controllable": True}},
                        }
                    ]
                }
            )
            # The element tables are extracted from the ASSET by the compiler, never
            # authored: they reach the baker as a VariantAnimation, exactly as
            # _read_variant_animations supplies them.
            animation = VariantAnimation(
                variant_set="surfaceState",
                cycle_s=1.0,
                windows_by_state=element_windows or {},
                members_by_state=element_members or {},
            )
            baker = AssetEditsBaker(
                _fake_composite("dynamic_rigids", node),
                None,
                Path(tmp),
                asset_loader=lambda path, control_sets=None, root_prim_path=None: asset,
                animations={("truck_1", "surfacestate"): animation},
            )
            with mock.patch("nurec.utils.edit.asset_edits.world_track_poses", return_value=_identity_poses(1)):
                baker.bake({"dynamic_rigids": doc.edits})
        return baker, node, doc

    def _mixed_fixture(self) -> tuple[AssetEditsBaker, _FakeLayerNode, AssetEditsDoc]:
        """Detail's payload is IDENTICAL in both states; Trim's DIFFERS per state."""
        asset = self._asset(
            {
                "off": [
                    SimReadyPart(prim_path="/Car/Detail", gaussians=_payload(11)),
                    SimReadyPart(prim_path="/Car/Trim", gaussians=_payload(12)),
                ],
                "sequence": [
                    SimReadyPart(prim_path="/Car/Detail", gaussians=_payload(11)),
                    SimReadyPart(prim_path="/Car/Trim", gaussians=_payload(13)),
                ],
            }
        )
        windows = {
            "off": {"Detail": [[0.0, 1.0]], "Trim": [[0.0, 0.5]]},
            "sequence": {"Detail": [[0.0, 0.5]], "Trim": [[0.5, 1.0]]},
        }
        return self._bake(asset, element_windows=windows)

    def test_identical_element_bakes_once_with_shared_aspect(self) -> None:
        """A payload-identical element bakes ONCE (role "element"): recorded on the baker
        and addressed by the state-agnostic :shared: aspect."""
        baker, node, doc = self._mixed_fixture()
        self.assertEqual(baker._shared_elements[("truck_1", "surfacestate")], ["Detail"])
        # 4 gaussians per payload: Detail ONCE + one Trim copy per state = 12, not 16.
        self.assertEqual(node.get_num_gaussians(), self._N0 + 12)
        self.assertEqual(
            [b.aspect for b in baker.baked_clusters],
            [
                "surfacestate:shared:Detail",
                "surfacestate:off",
                "surfacestate:off:Trim",
                "surfacestate:sequence",
                "surfacestate:sequence:Trim",
            ],
        )
        shared = baker.baked_clusters[0]
        self.assertEqual(shared.cluster_id, "truck_1_surfacestate_shared_Detail")
        self.assertEqual(shared.on_indices, list(range(self._N0, self._N0 + 4)))
        self.assertTrue(shared.shape["shared"])
        self.assertIsNone(shared.shape["state"])
        self.assertEqual(shared.shape["element"], "Detail")
        # Part roles in the asset record: one shared element bake, two state copies.
        roles = [(p["role"], p["element"], p["state"]) for p in baker.asset_records[0]["parts"]]
        self.assertEqual(
            roles,
            [("element", "Detail", None), ("variant", "Trim", "off"), ("variant", "Trim", "sequence")],
        )
        # The animation table advertises the shared element to external animators.
        (anim,) = baker.control_animations
        self.assertEqual(anim.shared_elements, ["Detail"])
        # The published SimReady->wire dictionary matches the baked clusters exactly.
        self.assertEqual(anim.state_aspects, {"off": "surfacestate:off", "sequence": "surfacestate:sequence"})
        self.assertEqual(
            anim.element_aspects,
            {
                "Detail": ["surfacestate:shared:Detail"],
                "Trim": ["surfacestate:off:Trim", "surfacestate:sequence:Trim"],
            },
        )
        vocabulary = set(anim.state_aspects.values()) | {
            a for aspects in anim.element_aspects.values() for a in aspects
        }
        self.assertEqual(vocabulary, {b.aspect for b in baker.baked_clusters})

    def test_differing_element_bakes_state_scoped_copies(self) -> None:
        """A payload-differing element keeps one copy per state under state-scoped
        aspects; the default state's copy bakes visible, the other state's hidden."""
        baker, node, _ = self._mixed_fixture()
        by_aspect = {b.aspect: b for b in baker.baked_clusters}
        off_rows = by_aspect["surfacestate:off:Trim"].on_indices
        sequence_rows = by_aspect["surfacestate:sequence:Trim"].on_indices
        self.assertEqual(off_rows, list(range(self._N0 + 4, self._N0 + 8)))
        self.assertEqual(sequence_rows, list(range(self._N0 + 8, self._N0 + 12)))
        # The shared Detail (default window covers 0) and the default-state Trim
        # copy keep their real logits; the non-default copy gets the hidden sentinel.
        detail_logits = np.float32(_payload(11).densities_logit.reshape(-1))
        np.testing.assert_allclose(
            node.densities.data[self._N0 : self._N0 + 4].numpy().reshape(-1), detail_logits, rtol=1e-6
        )
        off_logits = np.float32(_payload(12).densities_logit.reshape(-1))
        np.testing.assert_allclose(node.densities.data[off_rows].numpy().reshape(-1), off_logits, rtol=1e-6)
        self.assertTrue(torch.all(node.densities.data[sequence_rows] == HIDDEN_DENSITY_LOGIT))
        # The hidden copy's cluster still records the REAL post-activation alphas.
        sequence_logits = _payload(13).densities_logit.reshape(-1)
        np.testing.assert_allclose(
            by_aspect["surfacestate:sequence:Trim"].on_opacities, 1.0 / (1.0 + np.exp(-sequence_logits)), rtol=1e-6
        )

    def test_shared_element_default_visibility_from_window_coverage(self) -> None:
        """A shared element bakes hidden unless its DEFAULT-state window covers cycle
        position 0; a window-less default-state member bakes visible."""

        def _part_parts() -> list[SimReadyPart]:
            return [
                SimReadyPart(prim_path="/Car/ElementA", gaussians=_payload(31)),
                SimReadyPart(prim_path="/Car/ElementB", gaussians=_payload(32)),
                SimReadyPart(prim_path="/Car/ElementC", gaussians=_payload(33)),
            ]

        windows = {
            "off": {"ElementA": [[0.25, 0.5]], "ElementC": [[0.0, 0.25]]},
            "sequence": {"ElementA": [[0.0, 0.5]], "ElementB": [[0.5, 1.0]], "ElementC": [[0.75, 1.0]]},
        }
        members = {"off": ["ElementB"], "sequence": []}
        baker, node, doc = self._bake(
            self._asset({"off": _part_parts(), "sequence": _part_parts()}),
            element_windows=windows,
            element_members=members,
        )
        self.assertEqual(baker._shared_elements[("truck_1", "surfacestate")], ["ElementA", "ElementB", "ElementC"])
        rows = {b.shape["element"]: b.on_indices for b in baker.baked_clusters if b.shape.get("shared")}
        # ElementA's default-state window [0.25, 0.5) misses cycle position 0 -> hidden.
        self.assertTrue(torch.all(node.densities.data[rows["ElementA"]] == HIDDEN_DENSITY_LOGIT))
        # ElementB is a window-less default-state MEMBER -> statically visible.
        np.testing.assert_allclose(
            node.densities.data[rows["ElementB"]].numpy().reshape(-1),
            np.float32(_payload(32).densities_logit.reshape(-1)),
            rtol=1e-6,
        )
        # ElementC's default-state window [0.0, 0.25) covers 0 -> visible.
        np.testing.assert_allclose(
            node.densities.data[rows["ElementC"]].numpy().reshape(-1),
            np.float32(_payload(33).densities_logit.reshape(-1)),
            rtol=1e-6,
        )

    def test_zero_row_clusters_and_manifest_row_accounting(self) -> None:
        """Plain state clusters emit even with ZERO rows, and across the manifest the
        cluster on_indices exactly partition the rows appended to the grown tensors."""
        baker, node, _ = self._mixed_fixture()
        manifest = baker.manifest()
        self.assertEqual(len(manifest.assets), 1)
        self.assertEqual(len(manifest.controls), 1)
        # Every off/cycle part matched an element, so the plain state clusters (the
        # states' static remainders) emit with no rows — but they still emit.
        empty = [b for b in manifest.clusters if not b.on_indices]
        self.assertEqual(sorted(b.aspect for b in empty), ["surfacestate:off", "surfacestate:sequence"])
        for b in empty:
            self.assertEqual(b.on_opacities, [])
        for b in manifest.clusters:
            self.assertEqual(len(b.on_opacities), len(b.on_indices))
            self.assertEqual(b.shape["gaussians"], len(b.on_indices))
            self.assertEqual((b.layer_id, b.nurec_track_id), ("dynamic_rigids", 0))
        # No static parts here, so cluster rows are pairwise disjoint and their union
        # covers EXACTLY the rows the flush actually grew the layer tensors by.
        all_rows = [i for b in manifest.clusters for i in b.on_indices]
        self.assertEqual(len(all_rows), len(set(all_rows)))
        self.assertEqual(sorted(all_rows), list(range(self._N0, node.get_num_gaussians())))
        self.assertEqual(node.positions.shape[0], self._N0 + 12)
        self.assertTrue(torch.all(node.gaussian_cuboid_ids[self._N0 :] == 0))


class TrackPlacementBakeTest(unittest.TestCase):
    """nurec_track placement POSITIONS: rotate -> uniform fit_length scale to the cuboid
    LENGTH -> bbox align (centered in x/y, base resting on the cuboid floor), computed
    in the bound track's cuboid-local frame with hand-checkable numbers."""

    _N0 = 4
    _DIMS = (8.0, 3.0, 3.0)  # identity pose: local x=LENGTH, y=WIDTH, z=HEIGHT(up)
    # _box_payload's corner layout: dx=0.5, dy=2.0, dz=0.6, base at z=0.
    _CORNERS = np.array([[x, y, z] for x in (-0.25, 0.25) for y in (-1.0, 1.0) for z in (0.0, 0.6)])

    def _bake(self, placement: dict | None = None) -> tuple[AssetEditsBaker, _FakeLayerNode]:
        """Bake one static box prop bound to track 0 of an identity-pose cuboid."""
        node = _FakeLayerNode(self._N0, _FakeCuboidTracks(labels=["car"], dims=self._DIMS), cuboid_ids=[0] * self._N0)
        edit: dict = {"id": "prop_1", "bind": {"nre_track_id": 0}, "asset": "Body.usd"}
        if placement is not None:
            edit["placement"] = placement
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Body.usd").touch()
            doc = AssetEditsDoc.from_dict({"edits": [edit]})
            baker = AssetEditsBaker(
                _fake_composite("dynamic_rigids", node),
                None,
                Path(tmp),
                asset_loader=lambda path, control_sets=None, root_prim_path=None: _prop_asset(),
            )
            with mock.patch("nurec.utils.edit.asset_edits.world_track_poses", return_value=_identity_poses(1)):
                baker.bake({"dynamic_rigids": doc.edits})
        return baker, node

    def test_fit_length_scale_and_bbox_align_positions(self) -> None:
        """Default placement: scale = cuboid length / bbox length, x/y bbox-centered,
        base resting on the cuboid floor — and ONLY the length axis is fitted."""
        baker, node = self._bake()
        appended = node.positions.data[self._N0 :].numpy()
        # scale = 8.0 (cuboid length) / 0.5 (asset x extent) = 16; the base (z=0)
        # lands on the floor at -dims_z/2 = -1.5. The y extent blows out to 32 m in a
        # 3 m-wide cuboid: fit_length fits the LENGTH axis only, uniformly.
        expected = self._CORNERS * 16.0 + np.array([0.0, 0.0, -1.5])
        np.testing.assert_allclose(appended, expected, atol=1e-5)
        # The uniform scale also grows every per-gaussian log scale by log(scale).
        np.testing.assert_allclose(node.scales.data[self._N0 :].numpy(), math.log(16.0), rtol=1e-6)
        (record,) = baker.asset_records
        self.assertEqual(record["placement"]["scale"], 16.0)
        self.assertEqual(record["placement"]["scale_mode"], "fit_length")
        np.testing.assert_allclose(record["placement"]["shift_local"], [0.0, 0.0, -1.5], atol=1e-12)

    def test_placement_rotation_applies_before_fit_and_align(self) -> None:
        """A 90-deg yaw turns the assembly FIRST, so fit_length fits the ROTATED bbox
        (rotated x extent 2.0 -> scale 4) and part quats compose with the placement."""
        baker, node = self._bake(placement={"yaw_deg": 90.0})
        appended = node.positions.data[self._N0 :].numpy()
        # Rotate: (x, y, z) -> (-y, x, z); fit: 8.0 / 2.0 = 4; align: base to -1.5.
        rotated = np.stack([-self._CORNERS[:, 1], self._CORNERS[:, 0], self._CORNERS[:, 2]], axis=1)
        expected = rotated * 4.0 + np.array([0.0, 0.0, -1.5])
        np.testing.assert_allclose(appended, expected, atol=1e-5)
        self.assertAlmostEqual(baker.asset_records[0]["placement"]["scale"], 4.0, places=9)
        # Identity part quats become the placement quat itself: (cos 45, 0, 0, sin 45).
        half = math.sqrt(0.5)
        np.testing.assert_allclose(
            node.rotations.data[self._N0 :].numpy(), np.tile([half, 0.0, 0.0, half], (8, 1)), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()


class SameAssetTwoVehiclesTest(unittest.TestCase):
    """Two vehicles using the SAME asset — identical control, states and element names —
    bake cleanly: cluster ids are edit-id-prefixed (edit ids are unique per document),
    so the duplicate-cluster_id guard and the install gate both pass, while the two
    tracks expose IDENTICAL aspect strings (the wire addresses aspects per track)."""

    def test_bake_and_install_gate_pass_with_identical_variant_names(self) -> None:
        ct = _FakeCuboidTracks(labels=["car", "car"], dims=(8.0, 3.0, 3.0))
        node = _FakeLayerNode(6, ct, cuboid_ids=[0, 0, 0, 1, 1, 1])
        asset = AssetEditsBakerTest._asset()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "car.usd").touch()
            doc = AssetEditsDoc.from_dict(
                {
                    "edits": [
                        {
                            "id": "truck_1",
                            "bind": {"nre_track_id": 0},
                            "asset": "car.usd",
                            "controls": {"panelState": {"default": "off", "controllable": True}},
                        },
                        {
                            "id": "truck_2",
                            "bind": {"nre_track_id": 1},
                            "asset": "car.usd",
                            "controls": {"panelState": {"default": "off", "controllable": True}},
                        },
                    ]
                }
            )
            baker = AssetEditsBaker(
                _fake_composite("dynamic_rigids", node),
                None,
                Path(tmp),
                asset_loader=lambda path, control_sets=None, root_prim_path=None: asset,
            )
            with mock.patch("nurec.utils.edit.asset_edits.world_track_poses", return_value=_identity_poses(2)):
                baker.bake({"dynamic_rigids": doc.edits})

        manifest = baker.manifest()  # duplicate-cluster_id guard must not fire
        ids = [c.cluster_id for c in manifest.clusters]
        self.assertEqual(len(ids), len(set(ids)), f"cluster ids not unique: {ids}")
        aspects_by_track: dict[int, set[str]] = {}
        for c in manifest.clusters:
            aspects_by_track.setdefault(c.nurec_track_id, set()).add(c.aspect)
        self.assertEqual(set(aspects_by_track), {0, 1})
        self.assertEqual(aspects_by_track[0], aspects_by_track[1])  # same aspect vocabulary
        manifest.validate_against_layers({"dynamic_rigids": node})  # install gate passes


class ElementTokenTest(unittest.TestCase):
    """cluster_id element tokens are injective: path/underscore variants cannot collide."""

    def test_slash_and_underscore_keys_stay_distinct(self) -> None:
        self.assertNotEqual(_element_token("Detail/surface"), _element_token("Detail_surface"))
        self.assertNotEqual(_element_token("A/b_c"), _element_token("A_b/c"))

    def test_manifest_rejects_duplicate_cluster_ids(self) -> None:
        """A residual collision fails the BAKE rather than shipping an unloadable manifest."""
        baker = AssetEditsBaker.__new__(AssetEditsBaker)
        cluster = mock.MagicMock(cluster_id="dup")
        baker.baked_clusters = [cluster, cluster]
        baker.asset_records = []
        baker.control_animations = []
        with self.assertRaisesRegex(AssetEditsBakeError, "duplicate cluster_id"):
            baker.manifest()
