# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT initialization bypass and checkpoint metadata compatibility."""

import warnings

from unittest.mock import Mock, call

import mmseg.apis.inference as inference  # type: ignore[import-not-found]
import pytest

from mmengine import Config


META = {"classes": ("road", "building"), "palette": [[1, 2, 3], [4, 5, 6]]}
CHECKPOINT_LOAD = call("weights.pth", map_location="cpu", weights_only=False, mmap=True)


def _config():
    return Config(
        dict(
            default_scope="mmseg",
            model=dict(pretrained=None, train_cfg=None, decode_head=dict(num_classes=2)),
            test_pipeline=[dict(type="Resize", scale=(1, 1))],
            trt_settings=dict(batch_size=1, channel_size=3),
        )
    )


@pytest.mark.parametrize(
    "saved,model_options,checkpoint,expected_meta,warning_count,palette_calls,load_calls",
    [
        ({"dataset_meta": META}, {}, "weights.pth", META, 0, [], [CHECKPOINT_LOAD]),
        (
            {"CLASSES": META["classes"], "PALETTE": META["palette"]},
            {"decode_head": dict(num_classes=2)},
            "weights.pth",
            META,
            0,
            [],
            [CHECKPOINT_LOAD],
        ),
        ({}, {"decode_head": dict(num_classes=2)}, "weights.pth", META, 1, [call("test")], [CHECKPOINT_LOAD]),
        ({}, {"decode_head": dict(num_classes=3)}, "weights.pth", META, 2, [call("cityscapes")], [CHECKPOINT_LOAD]),
        ({"dataset_meta": None}, {}, "weights.pth", None, 0, [], [CHECKPOINT_LOAD]),
        ({}, {"decode_head": dict(num_classes=2)}, None, None, 0, [], []),
    ],
    ids=["modern", "legacy", "matched", "default", "null", "no_checkpoint"],
)
def test_trt_init_preserves_metadata_without_pytorch(
    monkeypatch, saved, model_options, checkpoint, expected_meta, warning_count, palette_calls, load_calls
):
    config = _config()
    del config.model.decode_head  # Saved metadata must not require a PyTorch decode head.
    config.model.update(model_options)
    load = Mock(return_value={"meta": saved})
    engine = Mock(return_value=Mock())
    palette = Mock(return_value=META["palette"])
    monkeypatch.setattr(inference, "init_default_scope", lambda _: None)
    monkeypatch.setattr(inference.torch, "load", load)
    monkeypatch.setattr(inference.MODELS, "build", Mock(side_effect=AssertionError("PyTorch build")))
    monkeypatch.setattr("mmengine.runner.load_checkpoint", Mock(side_effect=AssertionError("PyTorch weights")))
    monkeypatch.setattr(inference.torch.nn.Module, "to", Mock(side_effect=AssertionError("PyTorch upload")))
    monkeypatch.setattr(inference, "Mask2formerInferencer", engine)
    monkeypatch.setattr(inference, "dataset_aliases", {"test": []})
    monkeypatch.setattr(inference, "get_classes", lambda _: META["classes"])
    monkeypatch.setattr(inference, "get_palette", palette)
    with warnings.catch_warnings(record=True) as caught:
        model = inference.init_model(config, checkpoint, resolution=(64, 32), trt_checkpoint="engine.plan")
    assert model.cfg is config
    assert model.cfg.test_pipeline == [dict(type="Resize", scale=(64, 32))]
    assert model.dataset_meta == expected_meta
    assert len(caught) == warning_count
    assert palette.call_args_list == palette_calls
    engine.assert_called_once_with(
        engine_path="engine.plan",
        input_shape=(1, 3, 32, 64),
        batch_size=1,
        data_format="channel_first",
        is_inference=True,
    )
    assert load.call_args_list == load_calls


@pytest.mark.parametrize("missing_settings", [False, True])
def test_pytorch_init_loads_moves_and_evaluates(monkeypatch, missing_settings):
    config = _config()
    if missing_settings:
        config = Config({key: value for key, value in config.items() if key != "trt_settings"})
    model = Mock()
    model.decode_head = None
    build = Mock(return_value=model)
    load = Mock(return_value={"meta": {"CLASSES": META["classes"], "PALETTE": META["palette"]}})
    monkeypatch.setattr(inference, "init_default_scope", lambda _: None)
    monkeypatch.setattr(inference.MODELS, "build", build)
    monkeypatch.setattr("mmengine.runner.load_checkpoint", load)
    monkeypatch.setattr(inference, "Mask2formerInferencer", Mock(side_effect=AssertionError("TRT selected")))
    assert (
        inference.init_model(
            config, "weights.pth", device="cuda", trt_checkpoint="engine.plan" if missing_settings else None
        )
        is model
    )
    assert model.cfg is config and model.dataset_meta == META
    build.assert_called_once_with(config.model)
    load.assert_called_once_with(model, "weights.pth", map_location="cpu")
    model.to.assert_called_once_with("cuda")
    model.eval.assert_called_once_with()


def test_mmseg_import_defers_torch_model_and_visualization():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mmseg.apis; "
            "assert not any(name in sys.modules for name in "
            "('mmseg.models', 'mmdet', 'mmengine.runner', 'mmseg.visualization')); "
            "from mmseg.registry import MODELS; assert MODELS.get('EncoderDecoder') is not None",
        ],
        check=True,
    )
