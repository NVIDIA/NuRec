# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import zipfile

from pathlib import Path

import click
import numpy as np
import pytest

from click.testing import CliRunner

from workflows.av_benchmark.reference_map import _ecef_pose, cli, inject, world_base_from_pose_record


ECEF = [[0.4, 0.6, 0.6, 4167885.5], [-0.9, 0.5, 0.1, 658409.1], [-0.3, -0.6, 0.8, 4767475.7], [0, 0, 0, 1]]
IDENTITY = np.eye(4).tolist()
REFERENCE_START_US = 1_854_558_037_000


def _rig(world_base, start_us=0):
    poses, timestamps = [], []
    for i in range(50):
        pose = np.eye(4)
        pose[:3, 3] = [2.0 * i, 0.1 * i, 0.0]
        poses.append(pose.tolist())
        timestamps.append(start_us + 100_000 * i)
    return {
        "T_world_base": world_base,
        "rig_trajectories": [{"T_rig_worlds": poses, "T_rig_world_timestamps_us": timestamps}],
    }


def _usdz(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def _usdz_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _scene(tmp_path: Path) -> Path:
    return _usdz(
        tmp_path / "scene.usdz",
        {
            "default.usda": b"#usda 1.0\n",
            "checkpoint.ckpt": b"\x00" * 1000,
            "rig_trajectories.json": json.dumps(_rig(IDENTITY)).encode(),
            "datasource_summary.json": json.dumps({"rig_trajectories": {"T_world_base": IDENTITY}}).encode(),
        },
    )


def _reference(tmp_path: Path, with_xodr=True) -> Path:
    members = {
        "rig_trajectories.json": json.dumps(_rig(ECEF, start_us=REFERENCE_START_US)).encode(),
        "pose_record.json": b'{"record": []}',
        "clipgt/lane.parquet": b"lanes",
        "frames/camera_front_wide_120fov/1.jpeg": b"jpeg",
    }
    if with_xodr:
        members["map.xodr"] = b"<OpenDRIVE/>"
    return _usdz(tmp_path / "reference.usdz", members)


def _placed_pair(tmp_path: Path, **scene_args) -> tuple[Path, Path, np.ndarray]:
    """A scene package and a reference carrying the pose record it was recorded on, and a placement of its own
    5 m off the scene's, so placing by anything but the record shows."""
    record, rig, world_base = _pose_record_and_scene(**scene_args)
    scene = _usdz(
        tmp_path / "scene.usdz",
        {
            "default.usda": b"#usda 1.0\n",
            "checkpoint.ckpt": b"\x00" * 1000,
            "rig_trajectories.json": json.dumps(rig).encode(),
            "datasource_summary.json": json.dumps({"rig_trajectories": {"T_world_base": IDENTITY}}).encode(),
        },
    )
    stale = world_base.copy()
    stale[:3, 3] += [5.0, 0.0, 0.0]
    reference = _usdz(
        tmp_path / "reference.usdz",
        {
            "rig_trajectories.json": json.dumps(_rig(stale.tolist(), start_us=REFERENCE_START_US)).encode(),
            "pose_record.json": json.dumps(record).encode(),
            "map.xodr": b"<OpenDRIVE/>",
            "clipgt/lane.parquet": b"lanes",
            "frames/camera_front_wide_120fov/1.jpeg": b"jpeg",
        },
    )
    return scene, reference, world_base


def test_map_members_and_world_base_are_carried_and_the_scene_keeps_its_own_time(tmp_path: Path):
    scene, reference, world_base = _placed_pair(tmp_path)
    out = tmp_path / "out" / "clip.usdz"

    report = inject(scene, str(reference), out)

    assert report["status"] == "injected"
    assert report["world_base_source"] == "pose_record.json"
    with zipfile.ZipFile(out) as z, zipfile.ZipFile(scene) as src:
        names = z.namelist()
        assert {"map.xodr", "pose_record.json", "clipgt/lane.parquet"} <= set(names)
        assert not any(n.startswith("frames/") for n in names)
        rig = json.loads(z.read("rig_trajectories.json"))
        np.testing.assert_allclose(rig["T_world_base"], world_base, rtol=0, atol=1e-6)
        assert rig["rig_trajectories"][0]["T_rig_world_timestamps_us"][0] == 0
        assert json.loads(z.read("datasource_summary.json"))["rig_trajectories"]["T_world_base"] == rig["T_world_base"]
        assert z.read("checkpoint.ckpt") == src.read("checkpoint.ckpt")
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
    assert json.loads((tmp_path / "out" / "clip.reference_map.json").read_text())["status"] == "injected"


def test_placing_an_already_placed_scene_again_changes_nothing(tmp_path: Path):
    scene, reference, _ = _placed_pair(tmp_path)
    first, second = tmp_path / "first.usdz", tmp_path / "second.usdz"
    inject(scene, str(reference), first)

    inject(first, str(first), second)

    with zipfile.ZipFile(first) as a, zipfile.ZipFile(second) as b:
        assert b.namelist() == a.namelist()
        assert len(set(b.namelist())) == len(b.namelist())
        np.testing.assert_allclose(
            json.loads(b.read("rig_trajectories.json"))["T_world_base"],
            json.loads(a.read("rig_trajectories.json"))["T_world_base"],
            rtol=0,
            atol=1e-6,
        )


@pytest.mark.parametrize("scene_args", [{"stretch": 5.0}, {"drift_m": 10.0}], ids=["stretched", "drifting"])
def test_a_scene_that_is_not_the_recorded_drive_fails_and_writes_no_scene(tmp_path: Path, scene_args):
    scene, reference, _ = _placed_pair(tmp_path, **scene_args)
    out = tmp_path / "clip.usdz"

    result = CliRunner().invoke(cli, [f"--scene-package={scene}", f"--reference={reference}", f"--output={out}"])

    assert result.exit_code != 0
    assert not out.exists()
    report = json.loads((tmp_path / "clip.reference_map.json").read_text())
    assert report["status"] == "failed"
    assert report["agreement"]["median_position_m"] > 1.0


@pytest.mark.parametrize(
    "reference",
    [
        lambda tmp_path: "",
        lambda tmp_path: str(tmp_path / "missing.usdz"),
        lambda tmp_path: str(_reference(tmp_path, with_xodr=False)),
        lambda tmp_path: str(_reference(tmp_path)),
        lambda tmp_path: str(_usdz_bytes(tmp_path / "corrupt.usdz", b"not a zip archive")),
        lambda tmp_path: str(_usdz(tmp_path / "no_record.usdz", {"map.xodr": b"<OpenDRIVE/>"})),
        lambda tmp_path: str(
            _usdz(tmp_path / "bad_json.usdz", {"map.xodr": b"<OpenDRIVE/>", "pose_record.json": b"{not json"})
        ),
    ],
    ids=["empty", "missing", "no-xodr", "empty-pose-record", "corrupt-archive", "no-pose-record", "malformed-record"],
)
def test_an_unusable_reference_is_skipped_with_a_warning(tmp_path: Path, reference):
    out = tmp_path / "clip.usdz"

    result = CliRunner().invoke(
        cli, [f"--scene-package={_scene(tmp_path)}", f"--reference={reference(tmp_path)}", f"--output={out}"]
    )

    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert not out.exists()
    assert json.loads((tmp_path / "clip.reference_map.json").read_text())["status"] == "skipped"


def _pose_record_and_scene(
    n_scene=50, lead=7, drift_m=0.0, stretch=1.0, turn_deg=0.05, noise_m=0.0, jitter_us=3000, attitude=None
):
    """A pose record along a road near Stuttgart and a scene trajectory that is its ECEF poses seen
    from a known T_world_base, starting ``lead`` records in, on its own time base."""
    rng = np.random.default_rng(0)
    steps = 100_000 + rng.integers(-jitter_us, jitter_us + 1, size=n_scene + 2 * lead)
    record_timestamps = REFERENCE_START_US + np.concatenate([[0], np.cumsum(steps)])
    records = [
        {
            "alignment_world_pose": {
                "lat_lng_alt": {"latitude": 48.6799 + 1.5e-5 * i, "longitude": 8.9777 + 1e-5 * i, "altitude": 481.0},
                "axis_angle": {"qx": 0.0027, "qy": -0.0174, "qz": -0.9998, "angle": 157.5 + turn_deg * i},
            },
            "timestamp_microseconds": int(t),
        }
        for i, t in enumerate(record_timestamps)
    ]
    ecef = np.stack([_ecef_pose(r) for r in records[lead : lead + n_scene]])
    world_base = ecef[1].copy()
    world_base[:3, 3] += [3.0, -2.0, 1.0]
    poses = np.linalg.inv(world_base) @ ecef
    poses[1:, 0, 3] += np.linspace(0, drift_m, n_scene - 1)
    poses[:, :3, 3] = poses[0, :3, 3] + stretch * (poses[:, :3, 3] - poses[0, :3, 3])
    poses[:, :3, 3] += rng.normal(0.0, noise_m, size=(n_scene, 3))
    if attitude is not None:
        poses[:, :3, :3] = poses[:, :3, :3] @ attitude
    timestamps = record_timestamps[lead : lead + n_scene] - record_timestamps[lead]
    rig = {
        "T_world_base": IDENTITY,
        "rig_trajectories": [{"T_rig_worlds": poses.tolist(), "T_rig_world_timestamps_us": timestamps.tolist()}],
    }
    return {"record": records}, rig, world_base


def test_world_base_is_solved_from_the_pose_record(tmp_path: Path):
    record, rig, world_base = _pose_record_and_scene()

    solved, agreement = world_base_from_pose_record(rig, record)

    np.testing.assert_allclose(solved, world_base, rtol=0, atol=1e-6)
    assert agreement["median_position_m"] < 1e-6
    assert (
        agreement["time_offset_us"]
        == record["record"][8]["timestamp_microseconds"] - rig["rig_trajectories"][0]["T_rig_world_timestamps_us"][1]
    )

    scene = _usdz(tmp_path / "scene.usdz", {"rig_trajectories.json": json.dumps(rig).encode()})
    reference = _usdz(
        tmp_path / "reference.usdz",
        {"map.xodr": b"<OpenDRIVE/>", "pose_record.json": json.dumps(record).encode()},
    )
    report = inject(scene, str(reference), tmp_path / "clip.usdz")

    assert report["status"] == "injected"
    assert report["world_base_source"] == "pose_record.json"
    with zipfile.ZipFile(tmp_path / "clip.usdz") as z:
        placed = json.loads(z.read("rig_trajectories.json"))["T_world_base"]
        np.testing.assert_allclose(placed, world_base, rtol=0, atol=1e-6)


def test_the_fit_absorbs_noise_and_drift():
    record, rig, world_base = _pose_record_and_scene(drift_m=1.0, noise_m=0.05)

    solved, agreement = world_base_from_pose_record(rig, record)

    assert agreement["median_position_m"] < 0.2
    assert agreement["p95_position_m"] < 0.5
    np.testing.assert_allclose(np.asarray(solved)[:3, :3], world_base[:3, :3], atol=1e-2)


def test_the_roll_about_a_straight_road_is_fitted_from_the_pose_axes():
    record, rig, world_base = _pose_record_and_scene(turn_deg=0.0)
    poses = np.asarray(rig["rig_trajectories"][0]["T_rig_worlds"])
    origins = poses[:, :3, 3]
    assert np.linalg.matrix_rank(origins[1:] - origins[0], tol=1e-3) == 1

    solved, agreement = world_base_from_pose_record(rig, record)

    np.testing.assert_allclose(solved, world_base, rtol=0, atol=1e-6)
    assert agreement["median_rotation_deg"] < 1e-3


def test_the_first_and_last_rig_poses_do_not_move_the_fit():
    record, rig, world_base = _pose_record_and_scene()
    poses = rig["rig_trajectories"][0]["T_rig_worlds"]
    poses[0][0][3] += 4.0
    poses[-1][1][3] -= 4.0

    solved, _ = world_base_from_pose_record(rig, record)

    np.testing.assert_allclose(solved, world_base, rtol=0, atol=1e-6)


def test_a_pose_record_shorter_than_the_scene_is_skipped(tmp_path: Path):
    record, rig, _ = _pose_record_and_scene()
    record["record"] = record["record"][:10]
    scene = _usdz(tmp_path / "scene.usdz", {"rig_trajectories.json": json.dumps(rig).encode()})
    reference = _usdz(
        tmp_path / "reference.usdz",
        {"map.xodr": b"<OpenDRIVE/>", "pose_record.json": json.dumps(record).encode()},
    )

    assert inject(scene, str(reference), tmp_path / "clip.usdz")["status"] == "skipped"


def _record(latitude, longitude, altitude, qx=0.0, qy=0.0, qz=1.0, angle=0.0):
    return {
        "alignment_world_pose": {
            "lat_lng_alt": {"latitude": latitude, "longitude": longitude, "altitude": altitude},
            "axis_angle": {"qx": qx, "qy": qy, "qz": qz, "angle": angle},
        }
    }


def test_ecef_pose_places_enu_axes_and_scales_the_axis_by_the_angle():
    at_origin = _ecef_pose(_record(0.0, 0.0, 0.0))
    np.testing.assert_allclose(at_origin[:3, 3], [6378137.0, 0.0, 0.0], atol=1e-6)
    # East, north and up of (0, 0) are ECEF y, z and x.
    np.testing.assert_allclose(at_origin[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-12)

    # The axis is not normalized: 2 x 45 deg about up is a quarter turn, so east maps to north.
    turned = _ecef_pose(_record(0.0, 0.0, 0.0, qz=2.0, angle=45.0))
    np.testing.assert_allclose(turned[:3, :3] @ [1, 0, 0], at_origin[:3, :3] @ [0, 1, 0], atol=1e-12)


@pytest.mark.parametrize(
    "inputs",
    [lambda tmp_path: (_scene(tmp_path), ""), lambda tmp_path: _placed_pair(tmp_path, stretch=5.0)[:2]],
    ids=["skipped", "failed"],
)
def test_an_earlier_runs_output_does_not_survive(tmp_path: Path, inputs):
    out = tmp_path / "clip.usdz"
    out.write_bytes(b"earlier run")
    scene, ref = inputs(tmp_path)

    CliRunner().invoke(cli, [f"--scene-package={scene}", f"--reference={ref}", f"--output={out}"])

    assert not out.exists()


def test_an_output_that_is_the_scene_package_is_refused_and_the_scene_survives(tmp_path: Path):
    scene = _scene(tmp_path)
    before = scene.read_bytes()

    result = CliRunner().invoke(cli, [f"--scene-package={scene}", "--reference=", f"--output={scene}"])

    assert result.exit_code != 0
    assert scene.read_bytes() == before


def test_evenly_spaced_timestamps_are_ambiguous_and_skip_rather_than_misplace_the_map(tmp_path: Path):
    record, rig, _ = _pose_record_and_scene()
    for i, r in enumerate(record["record"]):
        r["timestamp_microseconds"] = REFERENCE_START_US + 100_000 * i
    rig["rig_trajectories"][0]["T_rig_world_timestamps_us"] = [100_000 * i for i in range(50)]

    with pytest.raises(ValueError, match="do not pin"):
        world_base_from_pose_record(rig, record)

    scene = _usdz(tmp_path / "scene.usdz", {"rig_trajectories.json": json.dumps(rig).encode()})
    reference = _usdz(
        tmp_path / "reference.usdz",
        {"map.xodr": b"<OpenDRIVE/>", "pose_record.json": json.dumps(record).encode()},
    )
    assert inject(scene, str(reference), tmp_path / "clip.usdz")["status"] == "skipped"
    assert not (tmp_path / "clip.usdz").exists()


def test_an_output_that_is_the_reference_is_refused_and_the_reference_survives(tmp_path: Path):
    reference = _reference(tmp_path, with_xodr=False)
    before = reference.read_bytes()

    result = CliRunner().invoke(
        cli, [f"--scene-package={_scene(tmp_path)}", f"--reference={reference}", f"--output={reference}"]
    )

    assert result.exit_code != 0
    assert reference.read_bytes() == before


def test_a_pose_record_exactly_as_long_as_the_scene_has_one_alignment_and_is_used():
    record, rig, world_base = _pose_record_and_scene()
    n = len(rig["rig_trajectories"][0]["T_rig_world_timestamps_us"])
    record["record"] = record["record"][7 : 7 + n]

    solved, agreement = world_base_from_pose_record(rig, record)

    np.testing.assert_allclose(solved, world_base, rtol=0, atol=1e-6)
    assert agreement["runner_up_error_us2"] == float("inf")


@pytest.mark.parametrize("jitter_us, pinned", [(6, True), (1, False)])
def test_a_few_microseconds_of_timestamp_jitter_pin_the_alignment(jitter_us, pinned):
    record, rig, world_base = _pose_record_and_scene(jitter_us=jitter_us)

    if pinned:
        np.testing.assert_allclose(world_base_from_pose_record(rig, record)[0], world_base, rtol=0, atol=1e-6)
    else:
        with pytest.raises(ValueError, match="do not pin"):
            world_base_from_pose_record(rig, record)


def test_a_local_divergence_fails_on_the_95th_percentile_alone(tmp_path: Path):
    record, rig, _ = _pose_record_and_scene()
    poses = rig["rig_trajectories"][0]["T_rig_worlds"]
    for i in range(20, 26):
        poses[i][1][3] += 5.0
    scene = _usdz(tmp_path / "scene.usdz", {"rig_trajectories.json": json.dumps(rig).encode()})
    reference = _usdz(
        tmp_path / "reference.usdz", {"map.xodr": b"<OpenDRIVE/>", "pose_record.json": json.dumps(record).encode()}
    )

    with pytest.raises(click.ClickException):
        inject(scene, str(reference), tmp_path / "clip.usdz")

    report = json.loads((tmp_path / "clip.reference_map.json").read_text())
    assert report["agreement"]["median_position_m"] < 1.0 < 3.0 < report["agreement"]["p95_position_m"]
    assert not (tmp_path / "clip.usdz").exists()


@pytest.mark.parametrize("pitch_deg, placed", [(2.0, True), (5.0, False)])
def test_a_constant_rig_attitude_offset_passes_up_to_the_rotation_gate(tmp_path: Path, pitch_deg, placed):
    c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    scene, reference, _ = _placed_pair(tmp_path, attitude=np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))

    if placed:
        assert inject(scene, str(reference), tmp_path / "clip.usdz")["status"] == "injected"
    else:
        with pytest.raises(click.ClickException):
            inject(scene, str(reference), tmp_path / "clip.usdz")
