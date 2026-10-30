# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CLI utilities module."""

from __future__ import annotations

import json

from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import click
import pytest

from nurec.config.scopedtimer import ProfilerBackend, VerbosityLiteral
from nurec.utils.cli import ScopedTimerConfig, SettingsCollector, config_to_cli_options


class TestSettingsCollector:
    """Tests for SettingsCollector class."""

    def test_serialize_params_basic_types(self) -> None:
        """Test serialization of basic types (str, int, float, bool, None)."""
        params = {
            "host": "localhost",
            "port": 8080,
            "scale": 1.5,
            "enabled": True,
            "disabled": False,
            "optional": None,
        }
        result = SettingsCollector._serialize_params(params)

        assert result == params

    def test_serialize_params_path(self) -> None:
        """Test serialization of Path objects to strings."""
        params = {
            "config": Path("/path/to/config.yaml"),
            "output_dir": Path("relative/path"),
        }
        result = SettingsCollector._serialize_params(params)

        assert result["config"] == "/path/to/config.yaml"
        assert result["output_dir"] == "relative/path"

    def test_serialize_params_tuple(self) -> None:
        """Test serialization of tuples to lists."""
        params = {
            "resolution": (576, 1024),
            "offset": (1.0, 2.0, 3.0),
            "cameras": ("front", "rear", "side"),
        }
        result = SettingsCollector._serialize_params(params)

        assert result["resolution"] == [576, 1024]
        assert result["offset"] == [1.0, 2.0, 3.0]
        assert result["cameras"] == ["front", "rear", "side"]

    def test_serialize_params_nested_tuple_with_path(self) -> None:
        """Test serialization of nested structures."""
        params = {
            "paths": (Path("/a"), Path("/b")),
        }
        result = SettingsCollector._serialize_params(params)

        assert result["paths"] == ["/a", "/b"]

    def test_serialize_params_enum(self) -> None:
        """Test serialization of Enum types."""
        from enum import Enum

        class Color(Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        params = {
            "color": Color.RED,
            "colors": (Color.GREEN, Color.BLUE),
        }
        result = SettingsCollector._serialize_params(params)

        assert result["color"] == "RED"
        assert result["colors"] == ["GREEN", "BLUE"]

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        timestamp = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        collector = SettingsCollector(
            command_name="test-command",
            args={"host": "localhost", "port": 8080},
            timestamp=timestamp,
            nurec_version="1.2.3-abc1234",
        )

        result = collector.to_dict()

        assert result["nurec_version"] == "1.2.3-abc1234"
        assert result["timestamp"] == "2025-01-15T10:30:00+00:00"
        assert result["command"] == "test-command"
        assert result["args"] == {"host": "localhost", "port": 8080}

    def test_save_json(self, tmp_path: Path) -> None:
        """Test saving settings to JSON file."""
        timestamp = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        collector = SettingsCollector(
            command_name="render",
            args={"output_dir": "/tmp/output", "height": 300},
            timestamp=timestamp,
            nurec_version="1.0.0-test",
        )

        output_file = tmp_path / "render_cli_args.json"
        output_path = collector.save_json(output_file)

        assert output_path == output_file
        assert output_path.exists()

        with open(output_path) as f:
            saved_data = json.load(f)

        assert saved_data["command"] == "render"
        assert saved_data["nurec_version"] == "1.0.0-test"
        assert saved_data["args"]["height"] == 300

    def test_save_json_nested_path(self, tmp_path: Path) -> None:
        """Test that save_json works with nested directory paths."""
        collector = SettingsCollector(
            command_name="test",
            args={},
            timestamp=datetime.now(timezone.utc),
            nurec_version="1.0.0",
        )

        output_file = tmp_path / "nested" / "output" / "dir" / "test_cli_args.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_path = collector.save_json(output_file)

        assert output_path.exists()

    @patch("nurec.utils.cli.get_version")
    def test_from_click_context(self, mock_get_version: MagicMock) -> None:
        """Test creating SettingsCollector from Click context."""
        mock_version = MagicMock()
        mock_version.version_string = "2.0.0-def5678"
        mock_get_version.return_value = mock_version

        mock_ctx = MagicMock()
        mock_ctx.params = {
            "host": "0.0.0.0",
            "port": 9090,
            "config": Path("/config.yaml"),
            "resolution": (720, 1280),
        }

        collector = SettingsCollector.from_click_context(mock_ctx, "serve-grpc")

        assert collector.command_name == "serve-grpc"
        assert collector.nurec_version == "2.0.0-def5678"
        assert collector.args["host"] == "0.0.0.0"
        assert collector.args["port"] == 9090
        assert collector.args["config"] == "/config.yaml"
        assert collector.args["resolution"] == [720, 1280]

    @patch("nurec.utils.cli.get_version")
    def test_from_click_context_no_version(self, mock_get_version: MagicMock) -> None:
        """Test creating SettingsCollector when version is unavailable."""
        mock_get_version.return_value = None

        mock_ctx = MagicMock()
        mock_ctx.params = {"host": "localhost"}

        collector = SettingsCollector.from_click_context(mock_ctx, "test")

        assert collector.nurec_version == "unknown"

    def test_log_settings(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging of settings."""
        import logging

        collector = SettingsCollector(
            command_name="render-grpc",
            args={"port": 8080, "enabled": True},
            timestamp=datetime.now(timezone.utc),
            nurec_version="1.0.0",
        )

        with caplog.at_level(logging.INFO):
            collector.log_settings(logging.getLogger("test"))

        assert "render-grpc args" in caplog.text
        assert "nurec_version=1.0.0" in caplog.text
        assert '"port": 8080' in caplog.text


class TestScopedTimerConfig:
    """Tests for ScopedTimerConfig class."""

    def test_config_to_cli_options(self) -> None:
        """Test conversion of ScopedTimerConfig to CLI options."""
        config = ScopedTimerConfig()
        options = config_to_cli_options(config)

        reference_options = [
            click.option(
                "--enable-timing/--no-enable-timing",  # NOTE: added --no- prefix to be consistent with other flag declarations
                is_flag=True,
                help="Enable timing of the different parts of the rendering pipeline.",
                default=False,
            ),
            click.option(
                "--timing-verbosity",
                type=click.Choice(["NONE", "SUMMARY", "BASIC", "DETAILS"], case_sensitive=False),
                help="Verbosity level for timing output.",
                default="BASIC",
                callback=lambda _, __, value: cast(VerbosityLiteral, value.upper()),
            ),
            click.option(
                "--timing-logfile",
                type=str,
                help="File to write timing results to (e.g., timing.log).",
                default=None,
            ),
            click.option(
                "--timing-synchronize",
                is_flag=True,
                help="Synchronize GPU before timing measurements.",
                default=False,
            ),
            click.option(
                "--profiling-backend",
                type=click.Choice(["NONE", "TRACY", "NVTX"], case_sensitive=False),
                help="Profiling backend to use: NONE (default), TRACY, or NVTX.",
                default="NONE",
                callback=lambda _, __, value: ProfilerBackend[value.upper()],
            ),
        ]

        def cli_options_decorator(input_options):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    pass

                setattr(wrapper, "__click_params__", list(getattr(func, "__click_params__", [])))
                for option in reversed(input_options):
                    wrapper = option(wrapper)
                return wrapper

            return decorator

        def dummy(*args, **kwargs):
            pass

        func = cli_options_decorator(options)(dummy)
        reference_func = cli_options_decorator(reference_options)(dummy)

        params = getattr(func, "__click_params__", [])
        old_params = getattr(reference_func, "__click_params__", [])

        for param, old_param in zip(params, old_params, strict=True):
            assert param.name == old_param.name
            assert param.default == old_param.default

            def type_key(tp):
                if isinstance(tp, click.Choice):
                    return ("choice", list(tp.choices), tp.case_sensitive)
                if isinstance(tp, click.Tuple):
                    return ("tuple", tuple(type_key(t) for t in tp.types))
                return ("basic", getattr(tp, "name", tp.__class__.__name__))

            assert type_key(param.type) == type_key(old_param.type)
            assert param.required == old_param.required


class TestTracksTimestampRayStrideCliOption:
    """Tests for the shared --tracks-timestamp-ray-stride option.

    `render` and `serve-grpc` both take the option from this one factory, so pinning the factory
    pins what those commands expose. Asserting it on the built commands themselves would drag the
    gRPC server into this target's deps; the end-to-end value the commands hand to
    `load_from_artifact` is covered in `nurec/render/render_test.py` instead.
    """

    @staticmethod
    def _param():
        """Build a real command and read the option off it through click's public surface."""
        from nurec.utils.cli import tracks_timestamp_ray_stride_cli_option

        @click.command()
        @tracks_timestamp_ray_stride_cli_option()
        def cmd(tracks_timestamp_ray_stride):  # pragma: no cover - never invoked
            pass

        return next(p for p in cmd.params if p.name == "tracks_timestamp_ray_stride")

    def test_defaults_to_16(self) -> None:
        """Absent the flag, both entry points hand 16 down to load_from_artifact."""
        assert self._param().default == 16

    def test_one_is_accepted_as_the_opt_out(self) -> None:
        """`1` is the documented way to restore full estimation, so it must pass validation."""
        param = self._param()
        assert param.type.convert(1, param, None) == 1

    def test_zero_is_rejected_at_parse_time(self) -> None:
        param = self._param()
        with pytest.raises(click.BadParameter):
            param.type.convert(0, param, None)
