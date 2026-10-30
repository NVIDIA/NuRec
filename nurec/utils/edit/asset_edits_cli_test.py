# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compile-asset-edits CLI persistence helpers.

Standalone and CPU-only: the checkpoint splice is pure state-dict surgery over
hand-built tensors, and the serialized track append is plain dict manipulation
mirroring the on-disk ``datasource_summary.json`` / ``sequence_tracks.json`` shape.
"""

import json
import tempfile
import unittest
import zipfile

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import click
import torch

from click.testing import CliRunner

from nurec.asset_edits.runtime import EDIT_MANIFEST_FILENAME
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.utils.edit.asset_edits import SyntheticTrack
from nurec.utils.edit.asset_edits_cli import (
    _append_tracks_to_serialized_cuboid_tracks,
    _splice_synthetic_tracks,
    _validate_asset_edit_target_layers,
    compile_asset_edits,
)


_LAYER = "dynamic_rigids"
_PREFIX = f"model.gaussians_nodes.{_LAYER}."


class _TopologyEditTarget(RigidGaussianModel):
    """Minimal rigid layer exposing the shared topology-edit preflight."""

    def __init__(self, *, reject: bool) -> None:
        """Create a target that either accepts or rejects topology edits."""

        torch.nn.Module.__init__(self)
        self.reject = reject

    def validate_external_gaussian_topology_edit(self, operation: str) -> None:
        """Optionally reject the requested edit like a row-aligned model."""

        if self.reject:
            raise NotImplementedError(f"{operation} is unsupported")


class AssetEditTargetValidationTest(unittest.TestCase):
    """Asset baking rejects row-aligned models before invoking the baker."""

    def test_regular_rigid_layer_is_accepted(self) -> None:
        """Ordinary rigid layers remain valid asset-edit targets."""

        composite = cast(
            Any,
            SimpleNamespace(gaussians_nodes={"dynamic_rigids": _TopologyEditTarget(reject=False)}),
        )

        _validate_asset_edit_target_layers(composite, ["dynamic_rigids"])

    def test_unsupported_rigid_layer_has_a_cli_error(self) -> None:
        """Unsupported topology edits surface as a user-facing CLI error."""

        composite = cast(
            Any,
            SimpleNamespace(gaussians_nodes={"dynamic_deformables": _TopologyEditTarget(reject=True)}),
        )

        with self.assertRaisesRegex(click.ClickException, "unsupported layer 'dynamic_deformables'"):
            _validate_asset_edit_target_layers(composite, ["dynamic_deformables"])


def _synth(edit_id: str, nurec_track_id: int, global_index: int) -> SyntheticTrack:
    """A minimal synthetic-track record with distinguishable pose/dims values."""
    return SyntheticTrack(
        edit_id=edit_id,
        layer_id=_LAYER,
        track_id_str=f"prop:{edit_id}",
        nurec_track_id=nurec_track_id,
        global_index=global_index,
        timestamps_us=(8_000_000, 22_000_000),
        pose_tquat_nurec=[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]] * 2,
        pose_tquat_world=[[11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 1.0]] * 2,
        dims_nurec=[0.5, 2.0, 0.6],
        dims_world=[0.5, 2.0, 0.6],
        label_class="protruding_object",
    )


def _state_dict(n_tracks: int = 3, n_poses: int = 10) -> dict:
    """A layer state_dict with every track-sized entry the splice handles."""
    return {
        _PREFIX + "tracks_calib.tracks_delta_q": torch.randn(n_poses, 4),
        _PREFIX + "tracks_calib.tracks_delta_t": torch.randn(n_poses, 3),
        _PREFIX + "time_embed.timestamps_us_ranges": torch.arange(n_tracks * 2, dtype=torch.int64).reshape(n_tracks, 2),
        _PREFIX + "time_embed.u": torch.randn(n_tracks, 2, 5),
        _PREFIX + "_extra_state": {"n_tracks": n_tracks},
        "model._extra_state": {
            "obj_track_ids": {
                _LAYER: [f"orig_{i}@some:datasource" for i in range(n_tracks)],
                "dynamic_deformables": ["ped_0@some:datasource"],
            }
        },
        # Per-gaussian tensors must never trip the leftover scan even when their
        # leading dim happens to be irrelevant here.
        _PREFIX + "positions": torch.zeros(100, 3),
        # Foreign-layer keys are out of scope entirely.
        "model.gaussians_nodes.background.some_buffer": torch.zeros(n_tracks, 2),
    }


class SpliceSyntheticTracksTest(unittest.TestCase):
    """_splice_synthetic_tracks extends every track-sized checkpoint entry positionally."""

    def test_extends_all_known_entries(self) -> None:
        """Delta rows, timestamp ranges, step embedding and n_tracks all grow by K."""
        sd = _state_dict(n_tracks=3, n_poses=10)
        synths = [_synth("barrier_1", 3, 5), _synth("barrier_2", 4, 6)]
        _splice_synthetic_tracks(sd, _LAYER, synths, old_n_tracks=3, old_n_poses=10)

        dq = sd[_PREFIX + "tracks_calib.tracks_delta_q"]
        self.assertEqual(dq.shape, (14, 4))
        self.assertTrue(torch.equal(dq[10:], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(4, 1)))
        dt = sd[_PREFIX + "tracks_calib.tracks_delta_t"]
        self.assertEqual(dt.shape, (14, 3))
        self.assertTrue(torch.all(dt[10:] == 0.0))

        ranges = sd[_PREFIX + "time_embed.timestamps_us_ranges"]
        self.assertEqual(ranges.shape, (5, 2))
        self.assertEqual(ranges.dtype, torch.int64)
        self.assertTrue(torch.equal(ranges[3:], torch.tensor([[8_000_000, 22_000_000]] * 2)))

        u = sd[_PREFIX + "time_embed.u"]
        self.assertEqual(u.shape, (5, 2, 5))
        expected_block = torch.linspace(0.0, 1.0, 5 * 2 + 2)[1:-1].reshape(1, 5, 2).transpose(1, 2)
        self.assertTrue(torch.allclose(u[3], expected_block[0]))
        self.assertTrue(torch.allclose(u[4], expected_block[0]))

        self.assertEqual(sd[_PREFIX + "_extra_state"]["n_tracks"], 5)
        # The composite's persisted per-layer track-id list grows in step ...
        obj_ids = sd["model._extra_state"]["obj_track_ids"]
        self.assertEqual(len(obj_ids[_LAYER]), 5)
        self.assertEqual(obj_ids[_LAYER][3:], ["prop:barrier_1", "prop:barrier_2"])
        # ... and other layers' lists stay untouched.
        self.assertEqual(obj_ids["dynamic_deformables"], ["ped_0@some:datasource"])
        # Untouched keys stay untouched.
        self.assertEqual(sd[_PREFIX + "positions"].shape, (100, 3))
        self.assertEqual(sd["model.gaussians_nodes.background.some_buffer"].shape, (3, 2))

    def test_missing_optional_entries_are_skipped(self) -> None:
        """Layers without calib or time-embed state splice only what exists."""
        sd = {_PREFIX + "_extra_state": {"n_tracks": 3}}
        _splice_synthetic_tracks(sd, _LAYER, [_synth("b", 3, 5)], old_n_tracks=3, old_n_poses=10)
        self.assertEqual(sd[_PREFIX + "_extra_state"]["n_tracks"], 4)

    def test_unhandled_track_sized_tensor_raises(self) -> None:
        """An unknown tensor still sized by the old track count fails the bake loudly."""
        sd = _state_dict()
        sd[_PREFIX + "future_per_track_thing"] = torch.zeros(3, 7)
        with self.assertRaisesRegex(click.ClickException, "unhandled track-sized"):
            _splice_synthetic_tracks(sd, _LAYER, [_synth("b", 3, 5)], old_n_tracks=3, old_n_poses=10)

    def test_leading_dim_mismatch_raises(self) -> None:
        """A handled entry whose leading dim disagrees with the expected count fails."""
        sd = _state_dict(n_poses=9)
        with self.assertRaisesRegex(click.ClickException, "leading dim"):
            _splice_synthetic_tracks(sd, _LAYER, [_synth("b", 3, 5)], old_n_tracks=3, old_n_poses=10)

    def test_extra_state_count_mismatch_raises(self) -> None:
        """An _extra_state n_tracks that disagrees with the model fails instead of drifting."""
        sd = _state_dict(n_tracks=3)
        sd[_PREFIX + "_extra_state"] = {"n_tracks": 7}
        with self.assertRaisesRegex(click.ClickException, "n_tracks"):
            _splice_synthetic_tracks(sd, _LAYER, [_synth("b", 3, 5)], old_n_tracks=3, old_n_poses=10)

    def test_obj_track_ids_count_mismatch_raises(self) -> None:
        """A persisted obj_track_ids list that disagrees with the track count fails the bake."""
        sd = _state_dict(n_tracks=3)
        sd["model._extra_state"]["obj_track_ids"][_LAYER] = ["only_one@ds"]
        with self.assertRaisesRegex(click.ClickException, "obj_track_ids"):
            _splice_synthetic_tracks(sd, _LAYER, [_synth("b", 3, 5)], old_n_tracks=3, old_n_poses=10)

    def test_track_albedos_and_scales_need_identity_rows(self) -> None:
        """Track appearance tensors splice caller-provided identity rows, or fail without them."""
        sd = _state_dict()
        sd[_PREFIX + "track_albedos"] = torch.randn(3, 3, 4)
        with self.assertRaisesRegex(click.ClickException, "identity row"):
            _splice_synthetic_tracks(sd, _LAYER, [_synth("b", 3, 5)], old_n_tracks=3, old_n_poses=10)

        sd = _state_dict()
        sd[_PREFIX + "track_albedos"] = torch.randn(3, 3, 4)
        sd[_PREFIX + "track_scales"] = torch.randn(3, 3)
        _splice_synthetic_tracks(
            sd,
            _LAYER,
            [_synth("b", 3, 5)],
            old_n_tracks=3,
            old_n_poses=10,
            track_albedo_identity=torch.eye(4)[:3],
            track_scale_identity=torch.zeros(3),
        )
        albedos = sd[_PREFIX + "track_albedos"]
        self.assertEqual(albedos.shape, (4, 3, 4))
        self.assertTrue(torch.equal(albedos[3], torch.eye(4)[:3]))
        self.assertEqual(sd[_PREFIX + "track_scales"].shape, (4, 3))


def _serialized_tracks(n: int = 2) -> dict:
    """A serialized CuboidTracks mapping in the datasource_summary.json shape."""
    return {
        "tracks_data": {
            "tracks_id": [str(100 + i) for i in range(n)],
            "tracks_poses": [[[float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 3 for i in range(n)],
            "tracks_timestamps_us": [[1, 2, 3] for _ in range(n)],
            "tracks_label_class": ["automobile"] * n,
            "tracks_flags": ["DYNAMIC"] * n,
        },
        "cuboidtracks_data": {"cuboids_dims": [[4.0, 2.0, 1.5]] * n},
    }


class AppendSerializedTracksTest(unittest.TestCase):
    """_append_tracks_to_serialized_cuboid_tracks mirrors the on-disk track shape."""

    def test_appends_in_frame(self) -> None:
        """NuRec and world frames pick their own pose/dims; every parallel list grows."""
        ct = _serialized_tracks(2)
        _append_tracks_to_serialized_cuboid_tracks(ct, [_synth("b", 2, 2)], "nurec", expect_global_index=True)
        td = ct["tracks_data"]
        self.assertEqual(td["tracks_id"], ["100", "101", "prop:b"])
        self.assertEqual(td["tracks_poses"][2], [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]] * 2)
        self.assertEqual(td["tracks_timestamps_us"][2], [8_000_000, 22_000_000])
        self.assertEqual(td["tracks_label_class"][2], "protruding_object")
        self.assertEqual(td["tracks_flags"][2], "DYNAMIC")
        self.assertEqual(ct["cuboidtracks_data"]["cuboids_dims"][2], [0.5, 2.0, 0.6])

        world = _serialized_tracks(2)
        _append_tracks_to_serialized_cuboid_tracks(world, [_synth("b", 2, 2)], "world")
        self.assertEqual(world["tracks_data"]["tracks_poses"][2], [[11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 1.0]] * 2)

    def test_global_index_mismatch_raises(self) -> None:
        """The reload-agreement invariant: append position must equal the recorded global index."""
        ct = _serialized_tracks(2)
        with self.assertRaisesRegex(click.ClickException, "global index"):
            _append_tracks_to_serialized_cuboid_tracks(ct, [_synth("b", 2, 5)], "nurec", expect_global_index=True)

    def test_duplicate_track_id_raises(self) -> None:
        """A synthetic id already present in the serialized tracks fails loud."""
        ct = _serialized_tracks(2)
        ct["tracks_data"]["tracks_id"][1] = "prop:b"
        with self.assertRaisesRegex(click.ClickException, "already present"):
            _append_tracks_to_serialized_cuboid_tracks(ct, [_synth("b", 2, 2)], "nurec")


class CompileGuardsTest(unittest.TestCase):
    """Upfront CLI guards: destructive path combinations fail before any expensive load."""

    @staticmethod
    def _artifact(tmp: str, members: dict[str, bytes]) -> Path:
        path = Path(tmp) / "last.usdz"
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return path

    @staticmethod
    def _yaml(tmp: str) -> Path:
        path = Path(tmp) / "edits.yaml"
        path.write_text("edits: []\n")
        return path

    def _invoke(self, artifact: Path, yaml: Path, output: Path):
        return CliRunner().invoke(
            compile_asset_edits,
            ["--artifact-path", str(artifact), "--asset-edits-yaml", str(yaml), "--output", str(output)],
        )

    def test_output_equal_to_input_rejected(self) -> None:
        """--output == --artifact-path would delete the trained artifact at repack time."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self._artifact(tmp, {"checkpoint.ckpt": b"x"})
            result = self._invoke(artifact, self._yaml(tmp), artifact)
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--output must differ", result.output)
            self.assertTrue(artifact.exists())

    def test_already_baked_input_rejected(self) -> None:
        """Re-baking a baked artifact would orphan the first bake's geometry."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self._artifact(tmp, {"checkpoint.ckpt": b"x", EDIT_MANIFEST_FILENAME: b"{}"})
            result = self._invoke(artifact, self._yaml(tmp), Path(tmp) / "out.usdz")
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("already a baked artifact", result.output)


class RepackArtifactTest(unittest.TestCase):
    """_repack_artifact + _rewrite_track_members: the artifact-surgery pair.

    Pins the checkpoint replacement, the fresh manifest member, the synthetic-track
    persistence into BOTH track members, ZIP_STORED preservation (a USDZ format
    requirement), and the no-partial-output failure contract.
    """

    def _source_usdz(self, tmp: str) -> Path:
        path = Path(tmp) / "in.usdz"
        ct = {
            "tracks_data": {
                "tracks_id": ["orig_0"],
                "tracks_poses": [[[0.0] * 7] * 2],
                "tracks_timestamps_us": [[0, 1]],
                "tracks_label_class": ["automobile"],
                "tracks_flags": ["DYNAMIC"],
            },
            "cuboidtracks_data": {"cuboids_dims": [[1.0, 2.0, 3.0]]},
        }
        summary = {"sequence_tracks_all": {"seq": ct}, "sequence_tracks_dynamic": {"seq": json.loads(json.dumps(ct))}}
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            buf = Path(tmp) / "ckpt.bin"
            torch.save({"state_dict": {"old": torch.zeros(1)}}, buf)
            zf.writestr("checkpoint.ckpt", buf.read_bytes())
            zf.writestr("datasource_summary.json", json.dumps(summary))
            zf.writestr("sequence_tracks.json", json.dumps({"chunk": json.loads(json.dumps(ct))}))
            zf.writestr(EDIT_MANIFEST_FILENAME, json.dumps({"clusters": [], "assets": [{"stale": True}]}))
            zf.writestr("scene.usdc", b"usd-bytes")
        return path

    def test_repack_replaces_checkpoint_manifest_and_persists_tracks(self) -> None:
        from nurec.asset_edits.runtime import EditManifest
        from nurec.utils.edit.asset_edits_cli import _repack_artifact

        with tempfile.TemporaryDirectory() as tmp:
            src = self._source_usdz(tmp)
            out = Path(tmp) / "out.usdz"
            manifest = EditManifest(clusters=[], assets=[{"edit_id": "fx"}])
            new_ckpt = {"state_dict": {"new": torch.ones(2)}}
            _repack_artifact(src, out, new_ckpt, manifest, [_synth("p1", 1, 1)])

            with zipfile.ZipFile(out) as zf:
                names = zf.namelist()
                self.assertEqual(names.count(EDIT_MANIFEST_FILENAME), 1)  # replaced, not duplicated
                self.assertEqual(json.loads(zf.read(EDIT_MANIFEST_FILENAME)), manifest.to_dict())
                for info in zf.infolist():
                    self.assertEqual(info.compress_type, zipfile.ZIP_STORED, info.filename)
                self.assertEqual(zf.read("scene.usdc"), b"usd-bytes")  # untouched member streams through
                with zf.open("checkpoint.ckpt") as f:
                    loaded = torch.load(f, weights_only=False)
                self.assertIn("new", loaded["state_dict"])
                summary = json.loads(zf.read("datasource_summary.json"))
                for member in ("sequence_tracks_all", "sequence_tracks_dynamic"):
                    ids = summary[member]["seq"]["tracks_data"]["tracks_id"]
                    self.assertEqual(ids, ["orig_0", "prop:p1"], member)
                world = json.loads(zf.read("sequence_tracks.json"))
                self.assertEqual(world["chunk"]["tracks_data"]["tracks_id"], ["orig_0", "prop:p1"])
                self.assertEqual(world["chunk"]["cuboidtracks_data"]["cuboids_dims"][-1], [0.5, 2.0, 0.6])

    def test_failed_repack_leaves_no_output(self) -> None:
        from nurec.asset_edits.runtime import EditManifest
        from nurec.utils.edit.asset_edits_cli import _repack_artifact

        with tempfile.TemporaryDirectory() as tmp:
            src = self._source_usdz(tmp)
            out = Path(tmp) / "out.usdz"
            # global_index 5 != append position 1 -> _rewrite_track_members raises.
            with self.assertRaises(click.ClickException):
                _repack_artifact(src, out, {"state_dict": {}}, EditManifest(clusters=[]), [_synth("p1", 1, 5)])
            self.assertFalse(out.exists())
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
