# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared frame selection and output naming for render exports."""

import enum
import re

from collections.abc import Iterable
from pathlib import PureWindowsPath


class CameraOutputSubdir(str, enum.Enum):
    COLOR = "color"
    DISTANCE_COLORIZED = "distance_colorized"
    SEMANTICS_COLORIZED = "semantics_colorized"


def validate_sensor_output_names(sensor_names: Iterable[str], *, sensor_type: str) -> None:
    """Require every output name to be a single directory component without renaming it."""
    for sensor_name in sensor_names:
        if (
            sensor_name in {"", ".", ".."}
            or any(character in sensor_name for character in ("/", "\\", "\0"))
            or PureWindowsPath(sensor_name).drive
        ):
            raise ValueError(
                f"Invalid {sensor_type} sensor name {sensor_name!r}: expected a nonempty directory name without "
                "path separators, drive prefixes, NUL bytes, '.' or '..'"
            )


def parse_frame_index(selection: str) -> tuple[int, int] | None:
    """Parse a single frame index, an inclusive range, or "all".

    Returns:
        None for "all", otherwise the inclusive (first, last) bounds. A single
        index returns the same value for both bounds.

    Raises:
        ValueError: If the selection is malformed, contains a negative index,
            or specifies a range whose end precedes its start.
    """
    if selection == "all":
        return None
    match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", selection)
    if match is None:
        raise ValueError("Use all, a non-negative frame index, or an inclusive range such as 10-20.")
    first = int(match[1])
    last = first if match[2] is None else int(match[2])
    if last < first:
        raise ValueError("The frame range end must be greater than or equal to its start.")
    return first, last


def select_frame_indices(selection: str, frame_count: int, frame_step: int = 1) -> range:
    """Apply the step from the selected start; explicit endpoints must both exist."""
    if frame_step < 1:
        raise ValueError("Frame step must be greater than zero.")
    if frame_count < 1:
        raise ValueError("No frames are available for rendering.")
    bounds = parse_frame_index(selection)
    first, last = (0, frame_count - 1) if bounds is None else bounds
    if last >= frame_count:
        raise ValueError(f"Frame index {last} is outside the available range 0-{frame_count - 1}.")
    return range(first, last + 1, frame_step)
