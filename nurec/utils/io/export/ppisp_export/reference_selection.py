# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PPISP reference (camera, frame) selection and validation.

Resolves raw user-supplied ``ppisp_reference_camera_id`` and
``ppisp_reference_frame_id`` parameters into a validated, dispatch-ready
:class:`ResolvedPPISPReferenceIds` value with explicit "was user set"
flags. This module is the single source of truth for the PPISP
reference fallback semantics shared by:

* The SH optimization (``sh-optimized`` export mode) -- which always
  needs a fully specified ``(camera_id, frame_id)`` pair as its bake
  target. Either side unset defaults to ``0``; both unset defaults to
  ``(0, 0)``.
* The SPG writer (``spg-runtime`` export mode) -- which uses the
  user-set flags to pick one of three authoring strategies:

  =======================  ======================  ====================================================================
  ``camera_was_user_set``  ``frame_was_user_set``  Resulting SPG attributes
  =======================  ======================  ====================================================================
  False                    False                   Time-sampled per camera (default)
  True                     False                   Single-frame; vignetting+CRF from camera, neutral exposure / identity color
  True                     True                    Single-frame; all PPISP attributes from the configured camera/frame
  False                    True                    Rejected (``ValueError``): frame-only fixing is ambiguous because
                                                   vignetting and CRF live on the camera axis
  =======================  ======================  ====================================================================

The asymmetry between modes is intentional: ``sh-optimized`` always
collapses to a single bake target so frame-only is meaningful (camera
defaults to ``0``), but ``spg-runtime`` authors per-camera SPG
attributes whose interpretation depends on knowing which camera the
fix applies to.

The resolver is hermetically testable: it accepts a duck-typed PPISP
module that only needs to expose ``num_cameras`` (int) and
``exposure_params`` (tensor with leading dim = ``num_frames``), so
unit tests do not require the real ``ppisp`` library or any GPU
state.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple


if TYPE_CHECKING:
    from ppisp import PPISP  # type: ignore[import-not-found]


log = logging.getLogger(__name__)


PPISPReferenceMode = Literal["sh-optimized", "spg-runtime"]
"""Export mode that drives the resolver's fallback semantics.

* ``"sh-optimized"``: PPISP is folded into Gaussian SH coefficients via
  the bake; resolver always returns a fully specified
  ``(camera_id, frame_id)`` pair (defaults: ``0``, ``0``).
* ``"spg-runtime"``: PPISP is kept as Omniverse SPG shaders applied at
  runtime; resolver's user-set flags drive the time-sampled vs
  single-frame attribute authoring decision.
"""


_VALID_MODES: Tuple[PPISPReferenceMode, ...] = ("sh-optimized", "spg-runtime")


@dataclass(frozen=True)
class ResolvedPPISPReferenceIds:
    """Validated, dispatch-ready PPISP reference (camera, frame) selection.

    Both ``camera_id`` and ``frame_id`` are always populated after
    resolution; the ``camera_was_user_set`` and ``frame_was_user_set``
    flags capture user intent so downstream dispatch can distinguish
    the four ``spg-runtime`` fallback rows without re-checking the raw
    inputs. ``sh-optimized`` consumers can ignore the flags.

    Attributes:
        camera_id: Resolved camera index. Always satisfies
            ``0 <= camera_id < ppisp_module.num_cameras``.
        frame_id: Resolved frame index. Always satisfies
            ``0 <= frame_id < ppisp_module.exposure_params.shape[0]``.
        camera_was_user_set: ``True`` iff the caller explicitly
            provided ``ppisp_reference_camera_id``. ``spg-runtime``
            uses this to switch from time-sampled to single-frame
            attribute authoring; ``sh-optimized`` ignores it.
        frame_was_user_set: ``True`` iff the caller explicitly
            provided ``ppisp_reference_frame_id``. ``spg-runtime``
            uses this to decide whether to author neutral exposure
            and identity color (when ``False`` and
            ``camera_was_user_set`` is ``True``) or use the
            configured frame's exposure/color; ``sh-optimized``
            ignores it.
    """

    camera_id: int
    frame_id: int
    camera_was_user_set: bool
    frame_was_user_set: bool


def resolve_ppisp_reference_ids(
    raw_camera_id: Optional[int],
    raw_frame_id: Optional[int],
    ppisp_module: "PPISP",
    *,
    mode: PPISPReferenceMode,
) -> ResolvedPPISPReferenceIds:
    """Resolve raw user-supplied PPISP reference IDs into a validated pair.

    Validation order:

    1. ``mode`` is one of the documented literals.
    2. ``raw_camera_id`` and ``raw_frame_id`` are each ``int`` or
       ``None`` (``bool`` rejected even though it is an ``int``
       subclass).
    3. ``ppisp_module`` exposes ``num_cameras`` and ``exposure_params``.
    4. Each set ID is in range against the PPISP module bounds.
    5. ``mode``-specific dispatch and unset defaults.

    The function is fail-loud at every stage; all branches raise with
    actionable error messages.

    Args:
        raw_camera_id: User-supplied camera index, or ``None`` when
            the caller did not specify one. ``None`` is the unset
            sentinel and is *not* a wrong type.
        raw_frame_id: User-supplied frame index, or ``None`` when the
            caller did not specify one.
        ppisp_module: Trained PPISP module exposing ``num_cameras``
            (``int``) and ``exposure_params`` (tensor whose leading
            dimension is the number of training frames).
        mode: Which export mode the resolution is for. See
            :data:`PPISPReferenceMode` for the per-mode semantics.

    Returns:
        A :class:`ResolvedPPISPReferenceIds` with both IDs populated
        and the user-set flags reflecting caller intent.

    Raises:
        TypeError: ``mode`` is not a valid literal, or
            ``raw_camera_id`` / ``raw_frame_id`` is neither ``int``
            nor ``None`` (``bool`` is rejected explicitly).
        ValueError: A set ID is out of range, or
            ``mode='spg-runtime'`` was called with frame-only
            specification (``raw_frame_id`` set, ``raw_camera_id``
            unset).
        AttributeError: ``ppisp_module`` is missing ``num_cameras``
            or ``exposure_params``.
    """
    _validate_mode(mode)
    _validate_id_type("ppisp_reference_camera_id", raw_camera_id)
    _validate_id_type("ppisp_reference_frame_id", raw_frame_id)

    num_cameras, num_frames = _read_ppisp_bounds(ppisp_module)

    if raw_camera_id is not None:
        _validate_id_range("ppisp_reference_camera_id", raw_camera_id, num_cameras)
    if raw_frame_id is not None:
        _validate_id_range("ppisp_reference_frame_id", raw_frame_id, num_frames)

    camera_was_user_set = raw_camera_id is not None
    frame_was_user_set = raw_frame_id is not None

    if mode == "spg-runtime" and frame_was_user_set and not camera_was_user_set:
        raise ValueError(
            "ppisp_reference_frame_id was set without ppisp_reference_camera_id "
            "in spg-runtime export mode. Frame-only fixing is ambiguous because "
            "vignetting and CRF live on the camera axis. Set "
            "ppisp_reference_camera_id as well, or leave both unset for "
            "time-sampled SPG authoring."
        )

    # Use `is not None` rather than the bool flags so mypy can narrow
    # `Optional[int]` to `int` inside each branch.
    camera_id = int(raw_camera_id) if raw_camera_id is not None else 0
    frame_id = int(raw_frame_id) if raw_frame_id is not None else 0

    resolved = ResolvedPPISPReferenceIds(
        camera_id=camera_id,
        frame_id=frame_id,
        camera_was_user_set=camera_was_user_set,
        frame_was_user_set=frame_was_user_set,
    )
    log.info(
        "ppisp_export.reference_selection: mode=%s camera_id=%d (user_set=%s) frame_id=%d (user_set=%s)",
        mode,
        resolved.camera_id,
        resolved.camera_was_user_set,
        resolved.frame_id,
        resolved.frame_was_user_set,
    )
    return resolved


# =============================================================================
# Internals
# =============================================================================


def _validate_mode(mode: object) -> None:
    """Reject any ``mode`` outside the documented literals."""
    if mode not in _VALID_MODES:
        raise TypeError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}")


def _validate_id_type(name: str, value: object) -> None:
    """Accept ``None`` or non-bool ``int``; reject everything else."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int or None, got {type(value).__name__}")


def _validate_id_range(name: str, value: int, upper_exclusive: int) -> None:
    """Range-check a set ID against the PPISP module's bound."""
    if value < 0 or value >= upper_exclusive:
        raise ValueError(f"{name} must be in [0, {upper_exclusive - 1}], got {value}")


def _read_ppisp_bounds(ppisp_module: object) -> Tuple[int, int]:
    """Read ``num_cameras`` and ``num_frames`` from a duck-typed PPISP module.

    The function only requires the two attributes used downstream,
    keeping the resolver hermetically testable against minimal mocks.
    """
    if not hasattr(ppisp_module, "num_cameras"):
        raise AttributeError(
            "ppisp_module is missing required attribute 'num_cameras'. "
            "PPISP reference resolution requires a trained PPISP module."
        )
    if not hasattr(ppisp_module, "exposure_params"):
        raise AttributeError(
            "ppisp_module is missing required attribute 'exposure_params'. "
            "PPISP reference resolution requires a trained PPISP module."
        )
    num_cameras = int(ppisp_module.num_cameras)  # type: ignore[attr-defined]
    num_frames = int(ppisp_module.exposure_params.shape[0])  # type: ignore[attr-defined]
    return num_cameras, num_frames
