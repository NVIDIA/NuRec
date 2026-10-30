# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math

import pytest

from workflows.av_benchmark.benchmark_row import build_row, camera_count, gpu_driver_version


def _validate_doc(cameras=7):
    return {
        "schema_version": 5,
        "hardware": {"gpu_name": "NVIDIA L40S"},
        "program_version": {"version_string": "26.10.32-77a7b7e1"},
        "aggregated_metrics": {
            "test/psnr": {"value": 24.914},
            "test/depth_mae": {"value": 1.5006},
            "test/raydrop_accuracy": {"value": 0.7684},
            "test/raydrop_IoU": {"value": 0.4353},
        },
        "metrics": {
            "general": {},
            "per_sequence": {"scene": {"per_camera": {"cam_%d" % i: {} for i in range(cameras)}, "per_lidar": {}}},
        },
    }


def _rgb_doc():
    return {
        "cross_sequence_metrics": {
            "static_camera_mask": {
                "psnr": {"avg": 25.124, "cnt": 599},
                "ssim": {"avg": 0.8807, "cnt": 599},
                "lpips": {"avg": 0.2714, "cnt": 599},
            }
        }
    }


def _object_doc():
    # Values from a Brev run of the object-level pass on a real clip.
    def result(key, headline, per_class_key, per_class, extra=None):
        return {
            "aggregated_results": [
                {
                    "method": "mean",
                    "result": {
                        "values": {key: headline},
                        "metadata": {
                            "per_class": {
                                name: {"num_objects": count, per_class_key: value, **(extra or {}).get(name, {})}
                                for name, (count, value) in per_class.items()
                            },
                            "per_track": {"12": {}, "17": {}, "18": {}, "36": {}},
                        },
                    },
                }
            ]
        }

    return {
        "metrics": {
            "perceptual": result(
                "object_level_perceptual",
                0.7413988709449768,
                "perceptual_score_mean",
                {"automobile": (22, 0.7606377525532722), "heavy_truck": (6, 0.6708561425715801)},
                {"automobile": {"psnr_mean": 26.0}, "heavy_truck": {"psnr_mean": 23.2}},
            ),
            "semantic": result(
                "object_level_semantic",
                0.9693661332130432,
                "semantic_raw_mean",
                {"automobile": (22, 0.9845677180723711), "heavy_truck": (6, 0.9136266907056173)},
            ),
        }
    }


def test_published_row_maps_every_tier_one_metric():
    render = {"aggregated_metrics": {"cuda_ms": {"value": 20.0}, "render_time_ms": {"value": 25.0}}}
    row = build_row(_validate_doc(), _rgb_doc(), render, clip_id="clip-a")

    assert row["clip_id"] == "clip-a"
    assert row["camera_psnr"] == 25.124  # the single-camera figure, which carries SSIM and LPIPS
    assert row["camera_psnr_7cam"] == 24.914  # the broader measurement, never averaged with it
    assert row["camera_ssim"] == 0.8807
    assert row["camera_lpips"] == 0.2714
    assert row["lidar_distance_mae"] == 1.5006
    assert row["lidar_return_agreement"] == 0.7684  # raydrop accuracy, not IoU
    assert row["lidar_raydrop_iou"] == 0.4353
    assert row["camera_count"] == 7
    assert "warnings" not in row


def test_render_rate_comes_from_cuda_events_not_wall_clock():
    # 20ms -> 50 fps; wall clock would give 40 fps. A 5% regression tolerance cannot survive the
    # two being mixed across releases, so the source is pinned and recorded.
    render = {"aggregated_metrics": {"cuda_ms": {"value": 20.0}, "render_time_ms": {"value": 25.0}}}
    row = build_row(_validate_doc(), _rgb_doc(), render)
    assert row["single_camera_render_rate_fps"] == 50.0
    assert row["render_rate_timing_method"] == "cuda_event"


def test_render_rate_falls_back_to_wall_clock_and_says_so():
    render = {"aggregated_metrics": {"render_time_ms": {"value": 25.0}}}
    row = build_row(_validate_doc(), _rgb_doc(), render)
    assert row["single_camera_render_rate_fps"] == 40.0
    assert row["render_rate_timing_method"] == "wall_clock"


def test_camera_dropout_is_flagged_rather_than_silent():
    row = build_row(_validate_doc(cameras=5), _rgb_doc(), None)
    assert row["warnings"] == ["camera_dropout:5/7"]
    assert row["camera_count"] == 5


def test_absent_inputs_each_carry_a_reason():
    row = build_row(None, None, None)
    assert row["fidelity_absent_reason"]
    assert row["rgb_absent_reason"]
    assert row["render_rate_absent_reason"]
    assert row["reconstruction_seconds_absent_reason"]
    assert row["object_level_absent_reason"]
    assert row["gpu_driver_version_absent_reason"]
    assert "camera_psnr" not in row


def test_manifest_supplies_the_reconstruction_time_rows():
    manifest = {"reconstruction_seconds": 3289.0, "phase_seconds": {"training_and_optimization": 1759.0}}
    row = build_row(_validate_doc(), _rgb_doc(), None, manifest=manifest)
    assert row["reconstruction_seconds"] == 3289.0
    assert row["phase_seconds_training_and_optimization"] == 1759.0
    assert "reconstruction_seconds_absent_reason" not in row


def test_a_key_missing_from_a_present_file_carries_a_reason():
    validate = _validate_doc()
    del validate["aggregated_metrics"]["test/raydrop_accuracy"]
    rgb = _rgb_doc()
    del rgb["cross_sequence_metrics"]["static_camera_mask"]["lpips"]
    row = build_row(validate, rgb, None, manifest={"phase_seconds": {}})

    assert "lidar_return_agreement" not in row
    assert row["lidar_return_agreement_absent_reason"] == "validate metrics carry no test/raydrop_accuracy"
    assert "camera_lpips" not in row
    assert row["camera_lpips_absent_reason"] == "render metrics carry no lpips average"
    assert row["reconstruction_seconds_absent_reason"] == "runner manifest carries no reconstruction_seconds"
    assert row["camera_psnr_7cam"] == 24.914  # the fields that are present are unaffected


def test_non_finite_metrics_are_absent_and_the_row_stays_valid_json():
    validate = _validate_doc()
    validate["aggregated_metrics"]["test/psnr"] = {"value": math.nan}
    rgb = _rgb_doc()
    rgb["cross_sequence_metrics"]["static_camera_mask"]["lpips"]["avg"] = math.inf
    # A NaN CUDA-event mean falls back to wall clock, and says so.
    render = {"aggregated_metrics": {"cuda_ms": {"value": math.nan}, "render_time_ms": {"value": 25.0}}}
    row = build_row(validate, rgb, render)

    assert "camera_psnr_7cam" not in row
    assert row["camera_psnr_7cam_absent_reason"] == "camera_psnr_7cam is not finite (nan)"
    assert "camera_lpips" not in row
    assert row["single_camera_render_rate_fps"] == 40.0
    assert row["render_rate_timing_method"] == "wall_clock"
    json.dumps(row, allow_nan=False)  # raises if any non-finite value reached the row

    objects = _object_doc()
    objects["metrics"]["perceptual"]["aggregated_results"][0]["result"]["values"]["object_level_perceptual"] = math.nan
    row = build_row(None, None, None, object_metrics=objects)
    assert row["object_perceptual_score_absent_reason"] == "object_perceptual_score is not finite (nan)"
    json.dumps(row, allow_nan=False)


def test_object_level_headlines_and_counts():
    row = build_row(None, None, None, object_metrics=_object_doc())
    assert row["object_perceptual_score"] == 0.7413988709449768
    assert row["object_semantic_score"] == 0.9693661332130432
    assert row["object_count"] == 28
    assert row["object_track_count"] == 4


def test_object_groups_reproduce_the_evaluators_weighting():
    row = build_row(None, None, None, object_metrics=_object_doc())
    assert row["object_count_vehicle"] == 28
    assert row["object_perceptual_vehicle"] == pytest.approx(row["object_perceptual_score"])
    assert row["object_semantic_vehicle"] == pytest.approx(row["object_semantic_score"])
    assert row["object_psnr_vehicle"] == pytest.approx((22 * 26.0 + 6 * 23.2) / 28)
    assert row["object_count_vru"] == 0
    assert row["object_perceptual_vru_absent_reason"] == "no vru objects scored"
    assert row["object_psnr_vru_absent_reason"] == "no vru objects scored"


@pytest.mark.parametrize(
    ("automobile_perceptual", "expected_vehicle"),
    [
        ({"perceptual_score_mean": 0.2, "perceptual_score_count": 1}, 0.5),
        ({"perceptual_score_count": 0}, 0.8),
    ],
    ids=["mixed_finite_weighting", "class_without_finite_value_drops_out"],
)
def test_object_groups_weight_each_class_by_its_finite_values(automobile_perceptual, expected_vehicle):
    objects = _object_doc()
    objects["metrics"]["perceptual"]["aggregated_results"][0]["result"]["metadata"]["per_class"].update(
        automobile={"num_objects": 2, "psnr_mean": 20.0, "psnr_count": 2, **automobile_perceptual},
        heavy_truck={
            "num_objects": 1,
            "perceptual_score_mean": 0.8,
            "perceptual_score_count": 1,
            "psnr_mean": 23.0,
            "psnr_count": 1,
        },
    )
    row = build_row(None, None, None, object_metrics=objects)
    assert row["object_count_vehicle"] == 3
    assert row["object_perceptual_vehicle"] == pytest.approx(expected_vehicle)
    assert row["object_psnr_vehicle"] == pytest.approx((2 * 20.0 + 23.0) / 3)


def test_object_psnr_is_absent_when_the_pass_predates_it():
    objects = _object_doc()
    per_class = objects["metrics"]["perceptual"]["aggregated_results"][0]["result"]["metadata"]["per_class"]
    for stats in per_class.values():
        del stats["psnr_mean"]
    row = build_row(None, None, None, object_metrics=objects)
    assert "object_psnr_vehicle" not in row
    assert row["object_psnr_vehicle_absent_reason"] == "object metrics carry no psnr_mean"
    assert row["object_perceptual_vehicle"] == pytest.approx(row["object_perceptual_score"])


def test_unlisted_classes_count_as_other_and_people_as_vru():
    objects = _object_doc()
    for metric, key in (("perceptual", "perceptual_score_mean"), ("semantic", "semantic_raw_mean")):
        per_class = objects["metrics"][metric]["aggregated_results"][0]["result"]["metadata"]["per_class"]
        per_class["person"] = {"num_objects": 2, key: 0.5}
        per_class["traffic_cone"] = {"num_objects": 3, key: 0.25}
        per_class["CYCLIST"] = {"num_objects": 1, key: 0.5}
    row = build_row(None, None, None, object_metrics=objects)
    assert row["object_count_vru"] == 3  # class names are matched case-insensitively
    assert row["object_perceptual_vru"] == 0.5
    assert row["object_count_other"] == 3
    assert row["object_semantic_other"] == 0.25
    assert row["object_count"] == 34


def test_tram_cars_and_other_vehicles_count_as_vehicles():
    objects = _object_doc()
    for metric, key in (("perceptual", "perceptual_score_mean"), ("semantic", "semantic_raw_mean")):
        per_class = objects["metrics"][metric]["aggregated_results"][0]["result"]["metadata"]["per_class"]
        per_class["train_or_tram_car"] = {"num_objects": 1, key: 0.5}
        per_class["other_vehicle"] = {"num_objects": 2, key: 0.5}
    row = build_row(None, None, None, object_metrics=objects)
    assert row["object_count_vehicle"] == 31
    assert row["object_count_other"] == 0


def test_camera_set_follows_the_expected_count():
    row = build_row(_validate_doc(cameras=6), _rgb_doc(), None, expected_camera_count=6)
    assert row["camera_psnr_6cam"] == 24.914
    assert "camera_psnr_7cam" not in row
    assert row["expected_camera_count"] == 6
    assert "warnings" not in row

    row = build_row(_validate_doc(cameras=5), _rgb_doc(), None, expected_camera_count=6)
    assert row["warnings"] == ["camera_dropout:5/6"]


def test_camera_count_reads_the_specs_hydra_list():
    assert camera_count("[camera_front_wide_120fov,camera_front_tele_30fov]") == 2
    assert camera_count("[]") == 0


def test_gpu_driver_version_parses_the_kernel_module_line():
    proprietary = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.65.06  Sun Jul 27 2025\nGCC version:"
    open_module = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.65.06  Release Build  (builder)"
    assert gpu_driver_version(proprietary) == "580.65.06"
    assert gpu_driver_version(open_module) == "580.65.06"
    assert gpu_driver_version("NVRM version: NVIDIA UNIX Kernel Module 580.65.06") == "580.65.06"
    assert gpu_driver_version("") is None
    assert build_row(None, None, None, driver_version="580.65.06")["gpu_driver_version"] == "580.65.06"


def test_nre_version_comes_from_validate():
    assert build_row(_validate_doc(), None, None)["nre_version"] == "26.10.32-77a7b7e1"
