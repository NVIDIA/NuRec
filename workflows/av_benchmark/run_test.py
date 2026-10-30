# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AV Benchmark workflow runner."""

import json
import re

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import workflows.av_benchmark.run as run_module

from workflows.av_benchmark.run import BINARIES, RECONSTRUCTION_PHASES, load_spec, resolve, run_workflow


def test_resolve_substitutes_and_rejects_unknown():
    assert resolve("$WORK/converted", {"WORK": "/tmp/w"}) == "/tmp/w/converted"
    assert resolve("pai_$CLIP_ID/pai_$CLIP_ID.json", {"CLIP_ID": "abc"}) == "pai_abc/pai_abc.json"
    with pytest.raises(KeyError):
        resolve("$NOPE", {})


def test_shipped_spec_is_valid(tmp_path: Path):
    spec_path = Path("workflows/av_benchmark/specs/av_benchmark_7cam.yaml")
    spec = load_spec(spec_path)
    # Pinned so a new stage must choose whether its time is reconstruction time.
    assert [(t["name"], t.get("phase")) for t in spec["tasks"]] == [
        ("download", None),
        ("convert", "input_preparation"),
        ("aux", "input_preparation"),
        ("train", "training_and_optimization"),
        ("export", "export_and_packaging"),
        ("export_plys", "export_and_packaging"),
        ("benchmark_gt", None),
        ("render", None),
        ("render_metrics", None),
        ("render_metrics_objects", None),
        ("gaussian_stats", None),
        ("validate", None),
        # No phase: reducing three metric files to the published row is bookkeeping, and counting
        # it would inflate the reconstruction time the benchmark publishes.
        ("benchmark_row", None),
        ("render_video", None),
        # No phase: it may fetch its reference over the network.
        ("inject_reference_map", None),
    ]
    # Every placeholder must resolve from spec vars, runner built-ins, or an earlier task's outputs.
    available = set(spec["vars"]) | {"WORK", "CLIP_ROOT", "CLIP_ID"}
    for task in spec["tasks"]:
        for arg in task["args"]:
            resolve(str(arg), {name: "x" for name in available})
        available |= set(task.get("produces") or {})


def test_checked_in_preset_carries_no_absolute_paths():
    """It is captured from a run, so it can only be shared if nothing machine-specific survived."""
    spec = load_spec(Path("workflows/av_benchmark/specs/av_benchmark_7cam.yaml"))
    # Every shipped config, not only the entry point: the lidar models are a separate file and a
    # captured path there would ship just the same.
    configs = sorted(Path(spec["vars"]["CONFIG_DIR"]).rglob("*.yaml"))
    assert len(configs) >= 2, f"expected the preset and its lidar group, found {configs}"

    # Any absolute value, not a list of the prefixes one capture happened to contain: a run on a
    # machine that stages under /tmp or /mnt would otherwise re-introduce this silently.
    absolute = re.compile(r":\s+(/\S*)")
    offenders = [
        f"{path}: {ln.strip()}" for path in configs for ln in path.read_text().splitlines() if absolute.search(ln)
    ]
    assert not offenders, offenders[:5]
    text = Path(spec["vars"]["CONFIG"]).read_text()
    # The two the spec always supplies are declared missing rather than left at a captured value,
    # so running without them fails loudly instead of writing somewhere arbitrary.
    assert "out_dir: ???" in text
    assert "  path: ???" in text


@pytest.mark.parametrize("task_name", ["train", "export", "export_plys", "gaussian_stats", "validate"])
def test_shipped_spec_config_consumers_use_no_config_dir(task_name: str):
    """The public CLI has no --config-dir: consumers pass a config name only (the runner
    absolutizes the repo-relative preset; PARSED_CONFIG is already absolute)."""
    spec = load_spec(Path("workflows/av_benchmark/specs/av_benchmark_7cam.yaml"))
    tasks_by_name = {task["name"]: task for task in spec["tasks"]}

    assert all("--config-dir" not in str(arg) for arg in tasks_by_name[task_name]["args"])


def test_load_spec_rejects_unknown_binary(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"tasks": [{"name": "t", "binary": "nope", "args": []}]}))
    with pytest.raises(ValueError):
        load_spec(bad)
    assert set(BINARIES) == {"nurec", "nurec_tools"}


def test_load_spec_rejects_task_without_name(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"tasks": [{"binary": "nurec", "args": []}]}))
    with pytest.raises(ValueError, match="non-empty 'name'"):
        load_spec(bad)


def test_load_spec_rejects_non_mapping_shapes(tmp_path: Path):
    """A scalar task or produces block must fail at load time, not mid-run."""
    task = {"name": "t", "binary": "nurec", "args": []}
    for spec, match in (
        ({"tasks": ["convert"]}, "must be a mapping"),
        ({"tasks": [task], "vars": "CLIP_ID"}, "'vars' must be a mapping"),
        ({"tasks": [{**task, "produces": "OUT"}]}, "'produces' as a mapping"),
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(spec))
        with pytest.raises(ValueError, match=match):
            load_spec(bad)


def test_load_spec_rejects_unknown_phase(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"tasks": [{"name": "t", "binary": "nurec", "args": [], "phase": "packaging"}]}))
    with pytest.raises(ValueError, match="unknown phase"):
        load_spec(bad)
    assert set(RECONSTRUCTION_PHASES) == {
        "input_preparation",
        "training_and_optimization",
        "export_and_packaging",
    }


def test_manifest_sums_phases_and_excludes_unphased_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reconstruction time is the sum of phased tasks only; an unphased task is timed, not counted."""
    spec = {
        "name": "s",
        "tasks": [
            {"name": "convert", "phase": "input_preparation", "binary": "nurec", "args": []},
            {"name": "aux", "phase": "input_preparation", "binary": "nurec", "args": []},
            {"name": "train", "phase": "training_and_optimization", "binary": "nurec", "args": []},
            {"name": "validate", "binary": "nurec", "args": []},
        ],
    }
    monkeypatch.setattr(run_module, "_binary_path", lambda binary: "/bin/true")
    monkeypatch.setattr(run_module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    clock = iter([0.0, 1.0, 1.0, 3.0, 3.0, 7.0, 7.0, 99.0])
    monkeypatch.setattr(run_module.time, "monotonic", lambda: next(clock))

    assert run_workflow(spec, {}, tmp_path) == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["phase_seconds"] == {"input_preparation": 3.0, "training_and_optimization": 4.0}
    assert manifest["reconstruction_seconds"] == 7.0
    assert [(t["task"], t["phase"], t["seconds"]) for t in manifest["tasks"]][-1] == ("validate", None, 92.0)


def test_manifest_reports_only_the_phases_that_ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A chain that stops early must not report a zeroed phase it never reached."""
    spec = {
        "name": "s",
        "tasks": [
            {"name": "convert", "phase": "input_preparation", "binary": "nurec", "args": []},
            {"name": "train", "phase": "training_and_optimization", "binary": "nurec", "args": []},
        ],
    }
    monkeypatch.setattr(run_module, "_binary_path", lambda binary: "/bin/false")
    monkeypatch.setattr(run_module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(run_module.time, "monotonic", iter([0.0, 2.0]).__next__)

    assert run_workflow(spec, {}, tmp_path) == 1

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["phase_seconds"] == {"input_preparation": 2.0}
    assert manifest["reconstruction_seconds"] == 2.0
    assert [t["task"] for t in manifest["tasks"]] == ["convert"]


def test_shipped_spec_makes_only_the_network_tasks_optional():
    """CI must run everything except the clip download and the reference map, which need a token and network."""
    spec = load_spec(Path("workflows/av_benchmark/specs/av_benchmark_7cam.yaml"))
    optional = [t["name"] for t in spec["tasks"] if t.get("optional")]

    assert optional == ["download", "inject_reference_map"]


@pytest.mark.parametrize("value", ["yes", 1, 0, "", []])
def test_load_spec_rejects_non_boolean_optional(tmp_path: Path, value: Any):
    """Falsy non-booleans must fail validation too, not be coerced into False."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"tasks": [{"name": "t", "binary": "nurec", "args": [], "optional": value}]}))
    with pytest.raises(ValueError, match="'optional' as a boolean"):
        load_spec(bad)


def test_optional_task_is_skipped_and_does_not_bind_its_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A skipped task produced nothing, so later tasks must not resolve against its `produces`."""
    spec = {
        "name": "s",
        "tasks": [
            {
                "name": "fetch",
                "binary": "nurec",
                "args": [],
                "optional": True,
                "produces": {"CLIP_ROOT": "$WORK/fetched"},
            },
            {"name": "use", "binary": "nurec", "args": ["--root=$CLIP_ROOT"]},
        ],
    }
    seen: list[list[str]] = []

    def record(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module, "_binary_path", lambda binary: "/bin/true")
    monkeypatch.setattr(run_module.subprocess, "run", record)
    monkeypatch.setattr(run_module.time, "monotonic", iter([0.0, 1.0]).__next__)

    assert run_workflow(spec, {"WORK": "/w", "CLIP_ROOT": "/fixture"}, tmp_path) == 0

    assert seen == [["/bin/true", "--root=/fixture"]]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["tasks"][0] == {"task": "fetch", "phase": None, "skipped": True}


def test_enabled_optional_task_runs_and_binds_its_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = {
        "name": "s",
        "tasks": [
            {
                "name": "fetch",
                "binary": "nurec",
                "args": [],
                "optional": True,
                "produces": {"CLIP_ROOT": "$WORK/fetched"},
            },
            {"name": "use", "binary": "nurec", "args": ["--root=$CLIP_ROOT"]},
        ],
    }
    seen: list[list[str]] = []

    def record(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module, "_binary_path", lambda binary: "/bin/true")
    monkeypatch.setattr(run_module.subprocess, "run", record)
    monkeypatch.setattr(run_module.time, "monotonic", iter([0.0, 1.0, 1.0, 2.0]).__next__)

    assert run_workflow(spec, {"WORK": "/w", "CLIP_ROOT": "/fixture"}, tmp_path, frozenset({"fetch"})) == 0

    assert seen[-1] == ["/bin/true", "--root=/w/fetched"]


def test_shipped_spec_defaults_are_the_benchmark_not_ci() -> None:
    """The spec is the benchmark definition; CI narrows it with --set.

    Pinned because the failure is silent in both directions: defaults dropped to CI values would
    publish a benchmark that never densifies, and the CI job would still pass.
    """
    spec = load_spec(Path("workflows/av_benchmark/specs/av_benchmark_7cam.yaml"))
    assert int(spec["vars"]["TRAIN_SAMPLES"]) >= 20000, (
        "below ~20000 steps the preset never densifies; that is a CI budget, not a benchmark one"
    )
    assert int(spec["vars"]["DURATION_SEC"]) > 2, "2 s is the CI fixture length, not a clip"
    image_level = next(t for t in spec["tasks"] if t["name"] == "render_metrics")
    objects = next(t for t in spec["tasks"] if t["name"] == "render_metrics_objects")
    assert "--object-level-metrics" in objects["args"], "the object-level pass is part of the definition"
    assert "--require-object-metrics" in objects["args"], "the object-level pass must fail when it scores no crops"
    # eval-rendering-metrics computes image-level OR object-level, never both. Combining them
    # drops SSIM and LPIPS, which no other stage produces.
    assert "--object-level-metrics" not in image_level["args"], (
        "the object-level flag replaces the whole-frame pass rather than adding to it"
    )
    for metric in ("--metrics=psnr", "--metrics=ssim", "--metrics=lpips"):
        assert metric in image_level["args"], f"{metric} has no other producer in the chain"


def test_camera_set_defaults_to_the_presets_own_list():
    """CAMERAS overrides the preset's three camera lists; defaulting it to the same list keeps the
    resolved config, and so every default run, unchanged."""
    spec = load_spec(Path("workflows/av_benchmark/specs/av_benchmark_7cam.yaml"))
    dataset = yaml.safe_load(Path(spec["vars"]["CONFIG"]).read_text())["dataset"]
    cameras = spec["vars"]["CAMERAS"]
    assert isinstance(cameras, str) and cameras == "[" + ",".join(dataset["camera_ids"]) + "]"
    assert dataset["train_camera_ids"] == dataset["val_camera_ids"] == dataset["camera_ids"]
    assert spec["vars"]["RENDER_CAMERA"] in dataset["camera_ids"], "render reads a camera the model must have"

    tasks = {task["name"]: task for task in spec["tasks"]}
    for name in ("train", "validate"):
        for key in ("camera_ids", "train_camera_ids", "val_camera_ids"):
            assert f"dataset.{key}=$CAMERAS" in tasks[name]["args"], f"{name} does not follow CAMERAS"
    assert "--cameras=$CAMERAS" in tasks["benchmark_row"]["args"]
    assert resolve("dataset.camera_ids=$CAMERAS", {"CAMERAS": "[a,b]"}) == "dataset.camera_ids=[a,b]"


def test_ground_mesh_is_off_by_default():
    """The default run exports no mesh; GROUND_MESH turns the ground mesh on for publishing."""
    spec = load_spec(Path("workflows/av_benchmark/specs/av_benchmark_7cam.yaml"))
    assert spec["vars"]["GROUND_MESH"] == "false"
    export = next(task for task in spec["tasks"] if task["name"] == "export")
    assert "checkpoint.artifact.mesh.ground.enabled=$GROUND_MESH" in export["args"]


@pytest.fixture
def cli_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "name": "download",
                        "binary": "nurec_tools",
                        "args": ["download", "--output-dir=$WORK/clip_hf"],
                        "optional": True,
                        "produces": {"CLIP_ROOT": "$WORK/clip_hf"},
                    },
                    {"name": "convert", "binary": "nurec_tools", "args": ["--root-dir=$CLIP_ROOT"]},
                ]
            }
        )
    )
    return spec_path


@pytest.mark.parametrize(
    "options, expected_root, downloads",
    [
        (["--with", "download"], "downloaded", True),
        (["--clip-root", "/local/clip"], "/local/clip", False),
        (["--clip-root", "/local/clip", "--with", "download"], "downloaded", True),
        (["--clip-root", "/local/clip", "--set", "CLIP_ROOT=/override/clip"], "/override/clip", False),
    ],
)
def test_main_uses_local_or_downloaded_clip(
    tmp_path: Path,
    cli_spec: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    expected_root: str,
    downloads: bool,
):
    """Exercise the CLI through task execution without fetching data or running GPU binaries."""
    seen: list[list[str]] = []

    def record(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module, "_binary_path", lambda binary: "/bin/true")
    monkeypatch.setattr(run_module.subprocess, "run", record)
    work_dir = tmp_path / "out"
    assert run_module.main(["--spec", str(cli_spec), "--work-dir", str(work_dir), *options]) == 0
    root = str(work_dir / "clip_hf") if expected_root == "downloaded" else expected_root
    expected = [["/bin/true", "--root-dir=" + root]]
    if downloads:
        expected.insert(0, ["/bin/true", "download", "--output-dir=" + str(work_dir / "clip_hf")])
    assert seen == expected


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--clip-root", ""],
        ["--set", "CLIP_ROOT=/override/clip"],
        ["--clip-root", "/local/clip", "--set", "CLIP_ROOT="],
    ],
)
def test_main_requires_clip_when_download_disabled(
    tmp_path: Path, cli_spec: Path, capsys: pytest.CaptureFixture[str], options: list[str]
):
    with pytest.raises(SystemExit) as exc:
        run_module.main(["--spec", str(cli_spec), "--work-dir", str(tmp_path / "out"), *options])
    assert exc.value.code == 2
    assert "--clip-root is required unless --with download is enabled" in capsys.readouterr().err


def test_main_rejects_unknown_optional_task_before_clip_requirement(tmp_path: Path, cli_spec: Path):
    with pytest.raises(SystemExit, match="--with names no optional task"):
        run_module.main(["--spec", str(cli_spec), "--work-dir", str(tmp_path / "out"), "--with", "not_a_task"])
