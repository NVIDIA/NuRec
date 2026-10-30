# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converter-author toolkit for the example NCore V4 converter.

Reusable helpers for the converter, grouped by sensor family:

* **Sequence helpers** — :func:`sanitize_sequence_id` converts dotted source
  identifiers into names compatible with NuRec auxiliary-data discovery and
  rejects characters that are unsafe in store filenames and paths.
  :func:`cuboid_track_sample_timestamps` selects the observation timestamps
  to store for one cuboid track clipped to the stored sensor coverage, and
  :func:`check_cuboid_track_timestamps` rejects a track whose stored
  timestamps differ from that selection.

* **LiDAR helpers** — :func:`bin_per_ray_to_columns` and
  :func:`build_lidar_model_from_per_frame_azimuth` build a
  ``RowOffsetStructuredSpinningLidarModelParameters`` from raw per-ray /
  per-frame azimuth streams. Both helpers implement the three
  ``row_azimuth_offsets_rad`` strategies that pair an XYZ-derived stored
  ``direction`` with a valid LiDAR model:

    1. **Sensor-native column schedule + preserve-Δ in
       row_azimuth_offsets_rad** — the rotor-phase schedule lives in
       ``column_azimuths_rad`` and the per-beam offset ``Δ`` is stored
       separately on ``row_azimuth_offsets_rad``. Use when the sensor
       publishes a clean per-beam ``Δ`` table and the rotor schedule is
       recoverable from per-frame absolute azimuth.
    2. **Sensor-native column schedule with Δ encoded in the column
       index** — ``column_azimuths_rad`` is fine enough (the sensor's own
       dense schedule) to encode ``Δ`` via column index alone, and the
       converter stores ``row_azimuth_offsets_rad = zeros``. Use when
       ``column_azimuths_rad`` already resolves ``(rotor + Δ)`` per ray.
    3. **Synthetic-uniform column schedule with Δ encoded in the column
       index** — same shape as #2 but the converter synthesizes a fine
       uniform schedule (e.g. ``n_columns >= 3600`` for a 360-deg
       spinner) when the native rotor schedule cannot be cleanly
       recovered. Use for rotor-phase-only streams, dual-return /
       near-field run patterns, or when ``|Δ|`` exceeds the native
       column step.

  The recommended stored ``direction`` across all three is XYZ-derived
  (``direction = xyz / ||xyz||``); model-derived directions paired with a
  rotor-phase-only ``column_azimuths_rad`` and zero
  ``row_azimuth_offsets_rad`` silently drop ``Δ`` and produce the
  LiDAR-smearing failure mode.

* **Camera helpers** — :func:`build_pinhole_model_from_opencv_K_D`
  converts an OpenCV ``(K, D)`` calibration into NCore's
  ``OpenCVPinholeCameraModelParameters``. :func:`verify_camera_model_projection`
  compares built NCore camera parameters against source-derived image-point
  references through the same projection path NuRec uses.

* **Pose helpers** — :func:`solve_static_extrinsic_mean` derives a single
  static ``T_sensor_rig`` from per-frame world-frame sensor poses.
"""

from __future__ import annotations

import logging

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ncore.data import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.impl.common.transformations import (  # internal API; no public equivalent yet
    PoseInterpolator,
    se3_inverse,
)
from ncore.sensors import CameraModel


_log = logging.getLogger(__name__)


MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG = 0.5
"""NuRec viability threshold: stored ray directions must match the LiDAR model
within this angular tolerance (degrees). The validator enforces the same bound
on a converted shard; keep this value in sync with the validator's threshold if
the validator changes it."""


MAX_CAMERA_MODEL_REPROJ_PX = 1.0
"""Default camera-model projection tolerance for source-derived image-point checks.

Use this as a fail-closed converter-side guard after building NCore camera
parameters. The tolerance is in pixel units: coefficient ordering errors are
most visible at the image edge, where small normalized-coordinate mismatches
produce large image-point errors.
"""


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------


def sanitize_sequence_id(name: str) -> str:
    """Convert a source sequence name into a NuRec-safe sequence ID.

    NuRec finds a sequence's auxiliary stores by truncating the data shard
    filename at its first ``"."``, so a dotted sequence ID matches nothing and
    the reconstruction silently runs without auxiliary data, so dots become
    ``"-"``.  Path separators and whitespace do not break that match, but the
    id also names the output directory and the store files, so they are
    rejected rather than silently rewritten.

    The mapping is not injective: ``"a.b"`` and ``"a-b"`` both yield
    ``"a-b"``.  Since the sequence ID also names the output directory and the
    auxiliary store prefix, callers converting a set of sequences should check
    the sanitized IDs for duplicates before writing, or one conversion
    overwrites another.

    Args:
        name: Source sequence name.

    Returns:
        The sequence ID with dots replaced by ``"-"``.

    Raises:
        ValueError: If ``name`` contains whitespace or a path separator.
    """
    invalid_characters = sorted({char for char in name if char == "/" or char.isspace()})
    if invalid_characters:
        formatted = ", ".join(repr(char) for char in invalid_characters)
        raise ValueError(f"sequence name contains unsafe character(s): {formatted}")
    return name.replace(".", "-")


def cuboid_track_sample_timestamps(
    source_timestamps_us: np.ndarray,
    coverage_start_us: int,
    coverage_end_us: int,
) -> np.ndarray:
    """Select the observation timestamps to store for one cuboid track.

    NCore rejects observations outside the sequence interval, and NuRec places
    a track only in frames inside its first-to-last timestamp span. Every source
    timestamp inside the coverage is kept, and each coverage boundary the
    source span crosses gains one timestamp so the stored span reaches it; the
    caller must interpolate the pose there from the bracketing source samples.
    No other timestamps are added.

    Args:
        source_timestamps_us: The track's source observation timestamps, in
            any order.
        coverage_start_us: First timestamp of the stored sensor coverage.
        coverage_end_us: Last timestamp of the stored sensor coverage.

    Returns:
        Sorted unique ``int64`` timestamps to store. May hold fewer than two
        entries, which NuRec cannot use as a track.

    Raises:
        ValueError: If ``source_timestamps_us`` is not 1-D or repeats a
            timestamp, or the coverage bounds are reversed.
    """
    source_us = np.asarray(source_timestamps_us, dtype=np.int64)
    if source_us.ndim != 1:
        raise ValueError(f"source_timestamps_us must be 1-D, got shape {source_us.shape}")
    native_us = np.sort(source_us)
    repeated_us = np.unique(native_us[1:][native_us[1:] == native_us[:-1]])
    if repeated_us.size:
        raise ValueError(f"source_timestamps_us repeats timestamp(s) {repeated_us.tolist()}")
    if coverage_start_us > coverage_end_us:
        raise ValueError(f"coverage bounds reversed: [{coverage_start_us}, {coverage_end_us}]")

    inside_us = native_us[(native_us >= coverage_start_us) & (native_us <= coverage_end_us)]
    if native_us.size == 0:
        return inside_us

    boundaries_us = []
    if native_us[0] < coverage_start_us <= native_us[-1]:
        boundaries_us.append(coverage_start_us)
    if native_us[0] <= coverage_end_us < native_us[-1]:
        boundaries_us.append(coverage_end_us)
    return np.unique(np.concatenate([inside_us, np.asarray(boundaries_us, dtype=np.int64)]))


def check_cuboid_track_timestamps(
    track_id: str,
    stored_timestamps_us: np.ndarray,
    source_timestamps_us: np.ndarray,
    coverage_start_us: int,
    coverage_end_us: int,
) -> None:
    """Fail when a track's stored timestamps differ from its source selection.

    Call on every track before ``store_observations``. The stored timestamps
    must equal :func:`cuboid_track_sample_timestamps` of the source timestamps,
    so any resampled, padded, repeated, or dropped observation is rejected.

    Args:
        track_id: Track identifier, used in the error message.
        stored_timestamps_us: Timestamps about to be stored for the track.
        source_timestamps_us: The track's source observation timestamps.
        coverage_start_us: First timestamp of the stored sensor coverage.
        coverage_end_us: Last timestamp of the stored sensor coverage.

    Raises:
        ValueError: If the stored timestamps are not exactly the selected
            source timestamps plus crossed coverage boundaries.
    """
    expected_us = cuboid_track_sample_timestamps(source_timestamps_us, coverage_start_us, coverage_end_us)
    stored_us = np.sort(np.asarray(stored_timestamps_us, dtype=np.int64))
    if np.array_equal(stored_us, expected_us):
        return
    extra_us = np.setdiff1d(stored_us, expected_us)
    missing_us = np.setdiff1d(expected_us, stored_us)
    repeated_us = np.unique(stored_us[1:][stored_us[1:] == stored_us[:-1]])
    raise ValueError(
        f"cuboid track {track_id}: stored timestamps differ from its source observations "
        f"({extra_us.size} not in the source selection, {missing_us.size} missing, "
        f"{repeated_us.size} repeated); first extra {extra_us[:3].tolist()}, "
        f"first missing {missing_us[:3].tolist()}"
    )


# ---------------------------------------------------------------------------
# LiDAR helpers
# ---------------------------------------------------------------------------


def _detect_column_stride(
    azimuth_rad: np.ndarray,
    tol_rad: float = 1e-4,
    max_probe: int = 1024,
) -> int:
    """Detect the column-major stride of a per-ray azimuth array.

    Assumes **row-varies-fastest** ordering: consecutive ``n_rows`` rays
    share a single column azimuth, so the first run of ~constant values
    reveals the stride.  For column-varies-fastest layouts (stride = 1),
    pass ``column_stride=1`` explicitly to
    :func:`build_lidar_model_from_per_frame_azimuth` instead of relying
    on auto-detection.

    Args:
        azimuth_rad: Per-ray azimuth array (one frame's worth), radians.
        tol_rad: Maximum spread within a column run, radians.
        max_probe: Cap on probe length to avoid scanning huge arrays.

    Returns:
        The detected stride (``>= 1``).  Returns ``1`` when consecutive
        rays already differ (per-column input rather than per-ray).

    Raises:
        ValueError: If no stride is found within ``max_probe`` rays.
    """
    n = int(min(azimuth_rad.size, max_probe))
    if n < 2:
        return 1
    a0 = float(azimuth_rad[0])
    for i in range(1, n):
        delta = float(azimuth_rad[i] - a0)
        if abs(((delta + np.pi) % (2.0 * np.pi)) - np.pi) > tol_rad:
            return i
    raise ValueError(
        f"Failed to detect column-major stride: first {n} azimuth values are constant within {tol_rad:.1e} rad."
    )


def _order_preserving_unique(
    azimuth_rad: np.ndarray,
    tol_rad: float = 1e-4,
) -> np.ndarray:
    """Return azimuths in firing order with adjacent run duplicates collapsed.

    Walks ``azimuth_rad`` in order and keeps the first occurrence of each
    new value (within ``tol_rad`` of the previously-kept value).  Unlike
    ``np.unique`` this preserves spin-direction order and the cyclic
    wrap-around between the last and first columns.

    Only adjacent duplicates are collapsed -- the caller is expected to
    have reduced the input to one azimuth per column already (e.g. via
    ``azimuth_rad[:, ::n_rows]`` or :func:`_detect_column_stride`).
    Non-adjacent equal azimuths within the same revolution would
    indicate the input spans more than one revolution; that is rejected
    by the strict-monotonicity check downstream.

    Args:
        azimuth_rad: 1-D azimuth array, radians.
        tol_rad: Equality tolerance, radians.

    Returns:
        1-D array of kept azimuths in original firing order.
    """
    if azimuth_rad.size == 0:
        return azimuth_rad.astype(np.float64, copy=True)
    out = [float(azimuth_rad[0])]
    two_pi = 2.0 * np.pi
    for v in azimuth_rad[1:]:
        d = float(v) - out[-1]
        # Wrap signed difference into [-pi, pi] so cyclic neighbors compare correctly.
        d_wrapped = ((d + np.pi) % two_pi) - np.pi
        if abs(d_wrapped) > tol_rad:
            out.append(float(v))
    # Cyclic wrap-around: when a run-length pattern straddles the rotation seam,
    # a cyclic shift can leave the same azimuth at both the first and last
    # positions (with non-equal values in between).  These represent two halves
    # of the same logical run and must collapse to one entry to keep the
    # unique-azimuth count invariant under rotation.
    if len(out) >= 2:
        d_seam = ((out[0] - out[-1] + np.pi) % two_pi) - np.pi
        if abs(d_seam) <= tol_rad:
            out.pop()
    return np.asarray(out, dtype=np.float64)


def _relative_angle_rad(ref: float, angle: np.ndarray, direction: Literal["cw", "ccw"]) -> np.ndarray:
    """Mirror of :func:`ncore.impl.data.util.relative_angle` (radians only).

    Returns the relative angle from ``ref`` to each value in ``angle``
    along ``direction``, projected to ``[0, 2*pi)``.  Used to validate
    the same monotonicity invariant NCore enforces on
    ``column_azimuths_rad``.
    """
    two_pi = 2.0 * np.pi
    ref_w = ref % two_pi
    a_w = angle % two_pi
    if direction == "cw":
        d = ref_w - a_w
    else:
        d = a_w - ref_w
    return d % two_pi


def _assert_monotonic_in_direction(column_azimuths_rad: np.ndarray, direction: Literal["cw", "ccw"]) -> None:
    """Raise if ``column_azimuths_rad`` violates NCore's strict-monotonicity rule."""
    if column_azimuths_rad.size < 2:
        return
    rel = _relative_angle_rad(float(column_azimuths_rad[0]), column_azimuths_rad, direction)
    diffs = np.diff(rel)
    if not np.all(diffs > 0):
        bad = int(np.sum(diffs <= 0))
        raise ValueError(
            f"column_azimuths_rad is not strictly monotonic in {direction!r} direction "
            f"({bad} non-positive diffs of relative_angle out of {diffs.size}). "
            "The order-preserving dedup collapsed runs incorrectly, or the per-frame "
            "azimuth stream is not a single continuous rotation."
        )


def _circular_xcorr_shift(ref_az_rad: np.ndarray, query_az_rad: np.ndarray) -> int:
    """Return the integer cyclic shift ``k`` such that ``np.roll(ref, -k) ~= query``.

    Robust to non-uniform column spacing and dual-return run patterns:
    correlates the unit-circle embedding ``e^{i*theta}`` rather than raw
    azimuth values, so wrap-around at ``2*pi`` is handled implicitly.
    """
    if ref_az_rad.size != query_az_rad.size:
        raise ValueError(
            f"_circular_xcorr_shift requires equal-length inputs (got {ref_az_rad.size} vs {query_az_rad.size})."
        )
    n = ref_az_rad.size
    if n == 0:
        return 0
    a = np.exp(1j * ref_az_rad.astype(np.float64))
    b = np.exp(1j * query_az_rad.astype(np.float64))
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    # Circular cross-correlation: corr[k] = sum_i a[i] * conj(b[(i+k) mod n]).
    # FFT convention: corr = IFFT(FFT(a) * conj(FFT(b))) reverses the sign in the index;
    # peak position k means query is ref rolled by -k (i.e. query[i] ~ ref[(i+k) mod n]).
    corr = np.fft.ifft(fa * np.conj(fb)).real
    return int(np.argmax(corr))


def _nn_bin_circular(
    query_az_rad: np.ndarray,
    ref_az_rad: np.ndarray,
    ref_sort_idx: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor bin ``query_az_rad`` against a circular reference.

    Args:
        query_az_rad: Per-ray azimuth values, radians.  Caller must have
            wrapped them to ``[0, 2*pi)``.
        ref_az_rad: Reference per-column azimuths, radians, in the firing
            order used by ``column_azimuths_rad``.  Must already be
            wrapped to ``[0, 2*pi)``.
        ref_sort_idx: ``np.argsort(ref_az_rad)`` precomputed once.

    Returns:
        Per-ray column index into ``ref_az_rad``.
    """
    two_pi = 2.0 * np.pi
    ref_sorted = ref_az_rad[ref_sort_idx]
    pos = np.searchsorted(ref_sorted, query_az_rad)
    n = ref_sorted.size
    # Wrap-around neighbors: index n maps to index 0 (sentinel + 2*pi);
    # index -1 maps to index n-1 (sentinel - 2*pi).
    left = (pos - 1) % n
    right = pos % n
    left_az = ref_sorted[left]
    right_az = ref_sorted[right]
    # Distances along the circle (always non-negative, <= pi).
    d_left = np.abs(((query_az_rad - left_az + np.pi) % two_pi) - np.pi)
    d_right = np.abs(((right_az - query_az_rad + np.pi) % two_pi) - np.pi)
    pick_right = d_right < d_left
    sorted_choice = np.where(pick_right, right, left)
    # Map sorted index -> firing-order index.
    return ref_sort_idx[sorted_choice]


def build_lidar_model_from_per_frame_azimuth(
    per_frame_azimuth: Sequence[np.ndarray] | np.ndarray,
    spinning_direction: Literal["cw", "ccw"],
    per_beam_azimuth_offset: np.ndarray | None = None,
    per_ray_beam_index: np.ndarray | None = None,
    column_stride: int | None = None,
    dedup_tolerance_rad: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Extract a real per-column azimuth schedule + per-frame phase from raw streams.

    Implements a principled free-spinning-rotor extraction recipe.
    Dataset-agnostic: operates purely on per-frame absolute-azimuth
    arrays plus the rotor's ``spinning_direction``.

    The algorithm:

    1. Detect the column stride from frame 0 (rays sharing a column
       have a constant azimuth run; ``stride = n_rows`` for typical
       row-varies-fastest layouts).
    2. Reduce frame 0 to its per-column schedule and order-preserving
       dedup the dual-return / near-field run pattern.  The collapsed
       length is ``n_columns`` (the unique-azimuth count per rotation,
       not the raw column count).
    3. Verify NCore's strict-monotonicity rule
       (``np.diff(relative_angle(spinning_direction, ...)) > 0``).
    4. For every frame, recover the cyclic phase shift relative to the
       reference via FFT-based circular cross-correlation on the
       unit-circle embedding.  Combined with the residual fractional
       phase, this yields the per-frame absolute phase offset.
    5. Bin every per-ray azimuth (with optional per-beam offset ``Δ``
       subtracted up front when the caller plans to store
       ``row_azimuth_offsets_rad = Δ``) directly against the reference
       schedule using a wrap-around-aware nearest-neighbor search.

    Self-validation: the per-ray NN-bin residual must stay below
    :data:`MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG` on every frame; otherwise
    a :class:`ValueError` is raised so the converter can investigate
    rather than silently storing rays that violate NCore's 0.5 deg
    angular-error gate.

    **Precondition.** This helper requires per-frame azimuth that
    contains real per-ray ``Δ`` structure -- within-column rays vary
    by ``Δ[ring]``.  For sensors that publish rotor-phase-only streams
    (within-column rays share a single azimuth, so the per-frame
    azimuth carries only the rotor schedule), the helper's stride
    detection cannot recover ``Δ`` and the rotor-phase schedule it
    returns will not be sufficient on its own.  Route to a
    **synthetic-uniform + encode-in-column** strategy instead: the
    converter constructs a synthetic uniform ``column_azimuths_rad``
    fine enough to encode ``Δ`` via column index alone, NN-bins
    ``(rotor + Δ[ring])`` per ray against it (e.g. via
    :func:`bin_per_ray_to_columns` with ``per_beam_az_offset_rad=None``),
    stores ``row_azimuth_offsets_rad = zeros``, and stores
    ``direction = xyz / ||xyz||``.

    Args:
        per_frame_azimuth: ``F`` frames of per-ray (or per-column)
            absolute azimuths in radians.  Each frame may be wrapped to
            any branch (``[0, 2*pi)``, ``[-pi, pi)``, etc.); the helper
            normalizes internally.  Accepts either a list of 1-D arrays
            or a stacked ``(F, N)`` ndarray (then every frame must share
            the same ray count).
        spinning_direction: ``'cw'`` or ``'ccw'`` -- the rotor's
            firing-order convention.  Must match what will be stored on
            ``RowOffsetStructuredSpinningLidarModelParameters``.
        per_beam_azimuth_offset: Optional ``(n_rows,)`` per-beam offset
            ``Δ`` (radians) to subtract from each ray's measured
            azimuth before stride detection / NN-binning so the helper
            recovers the rotor-only schedule.  Pair with the
            **sensor-native + preserve-Δ** strategy: store
            ``row_azimuth_offsets_rad = Δ`` on
            ``RowOffsetStructuredSpinningLidarModelParameters``.  The
            helper returns the rotor-only ``column_azimuths_rad`` and
            the model recovers full per-ray geometry via
            ``β = column_azimuths_rad[col] + row_azimuth_offsets_rad[row]``.
            When ``None``, no per-beam subtraction is applied -- the
            full ``(rotor + Δ)`` per-ray azimuth is binned against the
            extracted schedule (the encode-in-column strategy: store
            ``row_azimuth_offsets_rad = 0`` and let the column index
            carry ``Δ``).
        per_ray_beam_index: Optional per-ray ``(N,)`` beam (row) index.
            Required when ``per_beam_azimuth_offset`` is provided.  When
            omitted but ``per_beam_azimuth_offset`` is set, the helper
            assumes a row-varies-fastest layout and uses
            ``np.arange(N) % column_stride``.
        column_stride: Optional explicit column stride; ``None`` means
            auto-detect from frame 0.
        dedup_tolerance_rad: Equality tolerance used by the
            order-preserving dedup and the column-stride detection.

    Returns:
        Tuple ``(column_azimuths_rad, per_frame_phase_offsets_rad,
        model_element_col_indices)`` where:

        * ``column_azimuths_rad`` -- ``(n_columns,)`` ``float32``,
          strictly monotonic in ``spinning_direction`` and ready to
          drop into ``RowOffsetStructuredSpinningLidarModelParameters``.
        * ``per_frame_phase_offsets_rad`` -- ``(F,)`` ``float32`` signed
          phase offset of each frame relative to frame 0, expressed in
          the spinning direction (``[0, 2*pi)``).  **Diagnostic only**:
          NCore's ``RowOffsetStructuredSpinningLidarModelParameters``
          has no per-frame phase field; this value is useful for
          converter logging and sanity-checking but is not stored.
        * ``model_element_col_indices`` -- list of ``F`` per-frame
          ``(N,) uint16`` arrays giving each ray's column index into
          ``column_azimuths_rad``.  Slot directly into
          ``model_element[:, 1]``.

    Raises:
        ValueError: If ``per_frame_azimuth`` violates the documented
            shape (ndarray must be 2-D ``(F, N)``; sequence entries
            must each be 1-D), or if the per-frame azimuth structure
            cannot be reconciled (e.g. unique-azimuth counts disagree
            across frames, monotonicity is violated, or NN-bin residual
            exceeds :data:`MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG`).
    """
    if isinstance(per_frame_azimuth, np.ndarray):
        if per_frame_azimuth.ndim != 2:
            # Silent corruption: a 1-D ndarray would fall through to the
            # sequence branch below, where iteration yields scalars and
            # ``np.asarray(scalar)`` produces 0-D arrays. Each "frame"
            # then has ``a.size == 1`` in ``_bare_frame``, collapsing the
            # column schedule to length 1 without raising.
            raise ValueError(
                f"per_frame_azimuth ndarray must be 2-D with shape (F, N); "
                f"got ndim={per_frame_azimuth.ndim}, shape {per_frame_azimuth.shape}."
            )
        frames = [per_frame_azimuth[f].astype(np.float64) for f in range(per_frame_azimuth.shape[0])]
    else:
        frames = [np.asarray(f, dtype=np.float64) for f in per_frame_azimuth]
        bad_ndims = sorted({int(frame.ndim) for frame in frames if frame.ndim != 1})
        if bad_ndims:
            # Silent corruption: 0-D entries (e.g. a sequence of scalars
            # or 0-D arrays) reach ``_bare_frame`` with ``a.size == 1``
            # and collapse the column schedule to length 1.
            raise ValueError(
                f"per_frame_azimuth sequence entries must be 1-D ray-azimuth arrays; "
                f"got non-1-D entries with ndim {bad_ndims}."
            )
    if not frames:
        raise ValueError("per_frame_azimuth must contain at least one frame.")

    two_pi = 2.0 * np.pi

    # Subtract per-beam offsets up front, before stride detection.  Once subtracted,
    # within-column rays share the same residual azimuth and the per-column reduction
    # works.  Caller pairs this with ``row_azimuth_offsets_rad = Δ`` on the stored
    # parameters so the model recovers full per-ray geometry.
    per_beam_offset_arr: np.ndarray | None = None
    per_ray_beam_idx_arr: np.ndarray | None = None
    if per_beam_azimuth_offset is not None:
        per_beam_offset_arr = np.asarray(per_beam_azimuth_offset, dtype=np.float64)
        if per_beam_offset_arr.ndim != 1:
            raise ValueError(f"per_beam_azimuth_offset must be 1-D (got shape {per_beam_offset_arr.shape}).")
        if per_ray_beam_index is not None:
            per_ray_beam_idx_arr = np.asarray(per_ray_beam_index, dtype=np.int64)
            if (per_ray_beam_idx_arr < 0).any():
                raise ValueError(
                    f"per_ray_beam_index must contain only non-negative entries; "
                    f"numpy would otherwise silently wrap a negative index into "
                    f"per_beam_azimuth_offset and subtract the wrong beam's offset "
                    f"(got min={int(per_ray_beam_idx_arr.min())})."
                )

    def _bare_frame(idx: int, ray_az: np.ndarray) -> np.ndarray:
        """Return ray azimuth with per-beam offset subtracted, wrapped to [0, 2*pi)."""
        a = ray_az % two_pi
        if per_beam_offset_arr is None:
            return a
        if per_ray_beam_idx_arr is None:
            n_rows_assumed = per_beam_offset_arr.size
            if a.size % n_rows_assumed != 0:
                raise ValueError(
                    f"Frame {idx} ray count ({a.size}) is not divisible by "
                    f"len(per_beam_azimuth_offset) ({n_rows_assumed}); "
                    "pass per_ray_beam_index explicitly for non-row-varies-fastest layouts."
                )
            beam_idx = np.arange(a.size) % n_rows_assumed
        else:
            if per_ray_beam_idx_arr.size != a.size:
                raise ValueError(
                    f"per_ray_beam_index length ({per_ray_beam_idx_arr.size}) != frame {idx} ray count ({a.size})."
                )
            beam_idx = per_ray_beam_idx_arr
        return (a - per_beam_offset_arr[beam_idx]) % two_pi

    a0_bare = _bare_frame(0, frames[0])
    if column_stride is None:
        stride = _detect_column_stride(a0_bare, tol_rad=dedup_tolerance_rad)
    else:
        stride = int(column_stride)
        if stride < 1:
            raise ValueError(f"column_stride must be >= 1 (got {stride}).")

    a0_per_col = a0_bare[::stride]
    column_azimuths_rad = _order_preserving_unique(a0_per_col, tol_rad=dedup_tolerance_rad)
    n_columns = column_azimuths_rad.size
    if n_columns < 2:
        raise ValueError(
            f"Reference frame collapsed to {n_columns} unique azimuth(s); need at least 2 to build a column schedule."
        )

    _assert_monotonic_in_direction(column_azimuths_rad, spinning_direction)

    ref_sort_idx = np.argsort(column_azimuths_rad)

    max_residual_rad = np.deg2rad(MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG)
    phase_offsets = np.zeros(len(frames), dtype=np.float32)
    bin_indices: list[np.ndarray] = []

    for f, ray_az in enumerate(frames):
        a_bare = _bare_frame(f, ray_az)
        a_per_col = a_bare[::stride]
        a_per_col_unique = _order_preserving_unique(a_per_col, tol_rad=dedup_tolerance_rad)
        if a_per_col_unique.size != n_columns:
            raise ValueError(
                f"Frame {f} has {a_per_col_unique.size} unique azimuths after "
                f"dedup; reference frame has {n_columns}.  The rotor's column "
                "schedule is not preserved across frames -- not a free-spinning "
                "rotor in the sense this helper assumes."
            )

        # Phase recovery: cyclic shift k such that np.roll(reference, -k) ~= a_per_col_unique.
        k = _circular_xcorr_shift(column_azimuths_rad, a_per_col_unique)
        # Fractional phase: average signed circular distance between rolled-ref and frame f.
        rolled = np.roll(column_azimuths_rad, -k)
        signed = ((a_per_col_unique - rolled + np.pi) % two_pi) - np.pi
        eps = float(np.mean(signed))
        # Composite phase = relative_angle(reference[0], reference[k]) + eps, taken in the spinning direction.
        cyclic_part = float(
            _relative_angle_rad(
                float(column_azimuths_rad[0]),
                np.asarray([column_azimuths_rad[k]], dtype=np.float64),
                spinning_direction,
            )[0]
        )
        eps_signed_dir = eps if spinning_direction == "ccw" else -eps
        phase_offsets[f] = float((cyclic_part + eps_signed_dir) % two_pi)

        # Per-ray NN-bin against the reference (per-beam offset already subtracted in a_bare).
        col_idx = _nn_bin_circular(a_bare, column_azimuths_rad, ref_sort_idx)
        residual = np.abs(((a_bare - column_azimuths_rad[col_idx] + np.pi) % two_pi) - np.pi)
        max_residual = float(residual.max())
        if max_residual > max_residual_rad:
            raise ValueError(
                f"Frame {f} max NN-bin residual {np.degrees(max_residual):.4f} deg "
                f"> MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG ({MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG} deg). "
                "Either the per-frame schedule has a larger fractional phase drift than "
                "expected, or the per-beam offset / stride detection is incorrect."
            )
        bin_indices.append(col_idx.astype(np.uint16))

    return column_azimuths_rad.astype(np.float32), phase_offsets, bin_indices


def bin_per_ray_to_columns(
    per_ray_az_rad: np.ndarray,
    column_azimuths_rad: np.ndarray,
    *,
    per_beam_az_offset_rad: np.ndarray | None = None,
    per_ray_beam_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Map per-ray azimuths to column indices via wrap-around-aware NN-binning.

    Public wrapper around :func:`_nn_bin_circular` that produces
    ``model_element[:, 1]`` (per-ray column indices into
    ``column_azimuths_rad``) for any of the three
    ``row_azimuth_offsets_rad`` strategies (preserve-Δ, encode-in-column
    sensor-native, encode-in-column synthetic-uniform) documented at the
    top of this module.

    The output is correct only when ``column_azimuths_rad`` resolves
    the relevant azimuth structure: a rotor-only schedule for the
    preserve-Δ strategy, or a dense (sensor-native or
    synthetic-uniform) schedule for the encode-in-column strategies.
    This helper produces grid indices only; it does **not** affect
    the stored ``direction`` (XYZ-derived is recommended across all
    three strategies).

    The binning itself is a 1-D circular nearest-neighbor search using
    ``np.searchsorted`` on the sorted reference and circular distance
    over modulo-2*pi.  This is O(N log K) with no scipy dependency and
    is the right primitive when the row index is already known
    per-ray (i.e. the column is the only unknown).  NCore's runtime
    ``_init_angles_to_columns_map`` solves a related but more general
    problem (forward angle->element lookup over the full FOV using a
    3-D-ray kdtree); converters that have per-ray ``(row, az)`` should
    use this helper instead.

    Args:
        per_ray_az_rad: ``(N,)`` per-ray absolute azimuths, radians.
            Any branch of wrapping is accepted; the helper normalizes
            to ``[0, 2*pi)`` internally.  When pairing with
            ``row_azimuth_offsets_rad = Δ`` (preserve strategy), this
            is the rotor azimuth + ``Δ`` and the helper subtracts
            ``Δ`` via ``per_beam_az_offset_rad``.  When pairing with
            ``row_azimuth_offsets_rad = 0`` (encode-in-column
            strategies, sensor-native or synthetic-uniform), this is
            also rotor + ``Δ`` and the bin lands on the column index
            that encodes ``Δ``.
        column_azimuths_rad: ``(K,)`` reference per-column azimuth
            schedule, radians, in firing order.  Must be in
            ``[0, 2*pi)``.  This is the same array that goes onto
            ``RowOffsetStructuredSpinningLidarModelParameters.column_azimuths_rad``.
        per_beam_az_offset_rad: Optional ``(n_rows,)`` per-beam azimuth
            offset ``Δ`` (radians) to subtract before binning.  Pass
            when storing ``row_azimuth_offsets_rad = Δ`` on the
            parameters (preserve strategy); leave ``None`` when
            storing ``row_azimuth_offsets_rad = 0`` (``Δ`` is encoded
            in the column index instead).  Must be paired with
            ``per_ray_beam_idx`` when set.
        per_ray_beam_idx: Optional ``(N,)`` per-ray beam (row) index.
            Required when ``per_beam_az_offset_rad`` is provided.

    Returns:
        ``(N,) uint16`` per-ray column index into
        ``column_azimuths_rad``.  Drop directly into
        ``model_element[:, 1]``.
    """
    a = np.asarray(per_ray_az_rad, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"per_ray_az_rad must be 1-D (got shape {a.shape}).")
    ref = np.asarray(column_azimuths_rad, dtype=np.float64)
    if ref.ndim != 1:
        raise ValueError(f"column_azimuths_rad must be 1-D (got shape {ref.shape}).")
    if ref.size < 2:
        raise ValueError(f"column_azimuths_rad must have at least 2 elements (got {ref.size}).")

    two_pi = 2.0 * np.pi

    if per_beam_az_offset_rad is not None:
        if per_ray_beam_idx is None:
            raise ValueError("per_ray_beam_idx is required when per_beam_az_offset_rad is provided.")
        offsets = np.asarray(per_beam_az_offset_rad, dtype=np.float64)
        beam_idx = np.asarray(per_ray_beam_idx, dtype=np.int64)
        if offsets.ndim != 1:
            raise ValueError(f"per_beam_az_offset_rad must be 1-D (got shape {offsets.shape}).")
        if beam_idx.size != a.size:
            raise ValueError(f"per_ray_beam_idx length ({beam_idx.size}) != per_ray_az_rad length ({a.size}).")
        if (beam_idx < 0).any():
            # Silent corruption: numpy negative indexing would wrap a
            # stray ``-1`` to ``offsets[-1]`` and subtract the *last*
            # beam's offset, producing wrong-but-plausible bare azimuths
            # that NN-bin to wrong columns. Mirrors the guard in
            # ``_bare_frame`` of ``build_lidar_model_from_per_frame_azimuth``.
            raise ValueError(
                f"per_ray_beam_idx must contain only non-negative entries (got min={int(beam_idx.min())})."
            )
        a = (a - offsets[beam_idx]) % two_pi
    else:
        if per_ray_beam_idx is not None:
            raise ValueError("per_ray_beam_idx provided but per_beam_az_offset_rad is None.")
        a = a % two_pi

    ref = ref % two_pi
    ref_sort_idx = np.argsort(ref)
    col_idx = _nn_bin_circular(a, ref, ref_sort_idx)
    return col_idx.astype(np.uint16)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def verify_camera_model_projection(
    params: OpenCVPinholeCameraModelParameters | OpenCVFisheyeCameraModelParameters | FThetaCameraModelParameters,
    cam_rays: np.ndarray,
    expected_image_points: np.ndarray,
    *,
    tol_px: float = MAX_CAMERA_MODEL_REPROJ_PX,
) -> float:
    """Verify NCore camera projection against source-derived image points.

    The comparison is against NCore *image points* -- the continuous
    coordinates returned by ``CameraModel.camera_rays_to_image_points`` -- not
    integer pixel indices. NCore's convention places the center of integer pixel
    ``(i, j)`` at image point ``(i + 0.5, j + 0.5)`` (see
    ``CameraModel.pixels_to_image_points``). Compute ``expected_image_points``
    from the source camera model in that same convention; passing pixel-corner
    indices instead introduces a constant 0.5 px bias that can hide a real error
    under ``tol_px`` or trip a spurious failure under a tight tolerance.

    Args:
        params: Built NCore camera model parameters to verify.
        cam_rays: ``(N, 3)`` camera-frame rays, including samples near the
            image edge where distortion coefficient errors are largest.
        expected_image_points: ``(N, 2)`` image-point coordinates computed from
            the source dataset's camera model, independently from ``params`` and
            in NCore's image-point convention (pixel center = index + 0.5).
        tol_px: Maximum allowed Euclidean image-point error, in pixel units.

    Returns:
        The maximum observed image-point error in pixels.

    Raises:
        ValueError: If shapes are invalid, NCore marks any sample invalid, or
            the maximum image-point error exceeds ``tol_px``.
    """
    rays = np.asarray(cam_rays, dtype=np.float32)
    expected = np.asarray(expected_image_points, dtype=np.float32)
    if rays.ndim != 2 or rays.shape[1] != 3:
        raise ValueError(f"cam_rays must have shape (N, 3); got {rays.shape}.")
    if expected.shape != (rays.shape[0], 2):
        raise ValueError(f"expected_image_points must have shape ({rays.shape[0]}, 2); got {expected.shape}.")
    if rays.shape[0] == 0:
        raise ValueError("cam_rays and expected_image_points must contain at least one sample.")
    if not np.isfinite(rays).all() or not np.isfinite(expected).all():
        raise ValueError("cam_rays and expected_image_points must contain only finite values.")
    if not np.isfinite(tol_px) or tol_px < 0.0:
        raise ValueError(f"tol_px must be finite and non-negative; got {tol_px}.")

    model = CameraModel.from_parameters(params, device="cpu")
    projected = model.camera_rays_to_image_points(rays)
    actual = projected.image_points.detach().cpu().numpy().astype(np.float64)
    valid = projected.valid_flag.detach().cpu().numpy().astype(bool)

    if not bool(np.all(valid)):
        invalid_indices = np.flatnonzero(~valid)
        preview = ", ".join(str(int(i)) for i in invalid_indices[:8])
        if invalid_indices.size > 8:
            preview += f", ... ({invalid_indices.size - 8} more)"
        raise ValueError(
            f"NCore camera projection marked {invalid_indices.size}/{rays.shape[0]} check samples invalid "
            f"(indices: {preview}). Verify the source intrinsics and NCore camera model mapping."
        )

    errors = np.linalg.norm(actual - expected.astype(np.float64), axis=1)
    max_idx = int(np.argmax(errors))
    max_err = float(errors[max_idx])
    if max_err > float(tol_px):
        z = rays[:, 2].astype(np.float64)
        radius = np.full(rays.shape[0], np.inf, dtype=np.float64)
        finite_z = np.abs(z) > 0.0
        radius[finite_z] = np.linalg.norm(rays[finite_z, :2].astype(np.float64) / z[finite_z, None], axis=1)
        center_idx = int(np.argmin(radius))
        edge_idx = int(np.argmax(np.where(np.isfinite(radius), radius, -np.inf)))
        raise ValueError(
            f"NCore camera projection mismatch: max reprojection error {max_err:.3f}px at sample {max_idx} "
            f"> tolerance {float(tol_px):.3f}px. "
            f"Center sample {center_idx}: {float(errors[center_idx]):.3f}px; "
            f"max-radius sample {edge_idx}: {float(errors[edge_idx]):.3f}px. "
            "Check the source camera model family, coefficient ordering, and source-to-NCore parameter mapping."
        )
    return max_err


def build_pinhole_model_from_opencv_K_D(
    K: np.ndarray,
    D: np.ndarray,
    resolution_wh: tuple[int, int],
    shutter_type: ShutterType = ShutterType.GLOBAL,
) -> OpenCVPinholeCameraModelParameters:
    """Build NCore pinhole parameters from an OpenCV K matrix and distortion vector.

    Handles the full OpenCV distortion layout (4, 5, 8, 12, or 14 elements):

    * ``[k1, k2, p1, p2]``                          — 4-element
    * ``[k1, k2, p1, p2, k3]``                      — 5-element
    * ``[k1, k2, p1, p2, k3, k4, k5, k6]``          — 8-element (rational)
    * ``[k1, k2, p1, p2, k3, k4, k5, k6, s1…s4]``  — 12-element (thin prism)
    * ``[..., tau_x, tau_y]``                        — 14-element (tilted-sensor)

    The 14-element OpenCV layout encodes Scheimpflug tilted-sensor distortion
    in ``tau_x`` / ``tau_y``.  NCore's pinhole model has no representation for
    tilted-sensor distortion, so this helper drops those two coefficients.
    Whenever ``D`` carries the tau components (``len(D) > 12``) a
    ``logging.WARNING`` is emitted regardless of the tau magnitudes -- passing
    a 14-element ``D`` is itself an explicit declaration that the source
    calibration used ``cv2.CALIB_TILTED_MODEL``, and the binary fact "this
    coefficient is unsupported" doesn't depend on its value.  Re-calibrate
    without the tilted-sensor model if the resulting precision loss matters
    for your downstream usage.

    NCore mapping:

    * ``radial_coeffs``     = ``[k1, k2, k3, k4, k5, k6]``  shape ``(6,)`` float32
    * ``tangential_coeffs`` = ``[p1, p2]``                   shape ``(2,)`` float32
    * ``thin_prism_coeffs`` = ``[s1, s2, s3, s4]``           shape ``(4,)`` float32
    * ``focal_length``      = ``[K[0,0], K[1,1]]``           shape ``(2,)`` float32
    * ``principal_point``   = ``[K[0,2], K[1,2]]``           shape ``(2,)`` float32

    See ``ncore/impl/data/types.py::OpenCVPinholeCameraModelParameters`` for
    field definitions and shape/dtype assertions.

    Args:
        K: 3×3 camera matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
        D: OpenCV distortion vector (4 to 14 elements; missing elements zero-padded).
        resolution_wh: ``(width, height)`` in pixels.
        shutter_type: NCore shutter type; defaults to ``ShutterType.GLOBAL``.

    Returns:
        A fully-populated ``OpenCVPinholeCameraModelParameters`` instance.

    Raises:
        ValueError: If ``K`` is not ``(3, 3)``, if ``resolution_wh``
            contains a non-positive element, or if ``D`` has a
            non-canonical OpenCV length. Each guard prevents a
            distinct class of silent corruption (see inline comments).
    """
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.shape != (3, 3):
        # Silent corruption: a transposed (3, 3) K silently picks the
        # wrong scalars (``K[0, 2]`` is the bottom-left, not ``cx``);
        # a (3, 4) projection matrix passed by mistake still satisfies
        # ``K[0, 0]`` / ``K[1, 1]`` / ``K[0, 2]`` / ``K[1, 2]`` index
        # access but the [1, 2] entry is a translation, not ``cy``.
        # Reject anything that isn't exactly the canonical 3x3 layout.
        raise ValueError(f"K must have shape (3, 3); got shape {K_arr.shape}.")

    if len(resolution_wh) != 2 or any(int(v) <= 0 for v in resolution_wh):
        # Silent corruption: ``np.array([-1, h], dtype=np.uint64)``
        # silently wraps the negative element to 2**64 - 1 without
        # warning. NCore's ``CameraModelParameters.__post_init__``
        # asserts ``resolution[0] > 0 and resolution[1] > 0``, but the
        # wrapped huge positive uint64 satisfies that check -- NCore's
        # own defense is bypassed. Reject at the boundary on signed
        # input. Zero would likewise produce a (0, w) / (h, 0) "valid"
        # resolution that NCore would accept.
        raise ValueError(f"resolution_wh must be (width, height) with both elements > 0; got {tuple(resolution_wh)}.")

    d = np.asarray(D, dtype=np.float32).ravel()
    if d.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"D must have a canonical OpenCV length (4, 5, 8, 12, or 14 elements); "
            f"got length {d.size}. See cv2.calibrateCamera flags."
        )

    def _d(i: int) -> float:
        return float(d[i]) if len(d) > i else 0.0

    # OpenCV layout: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y]
    k1, k2, p1, p2 = _d(0), _d(1), _d(2), _d(3)
    k3, k4, k5, k6 = _d(4), _d(5), _d(6), _d(7)
    s1, s2, s3, s4 = _d(8), _d(9), _d(10), _d(11)

    if len(d) > 12:
        _log.warning(
            "OpenCV tilted-sensor distortion (tau_x=%g, tau_y=%g) is not "
            "representable in NCore's OpenCVPinholeCameraModelParameters and "
            "will be silently dropped. Re-calibrate without the tilted-sensor "
            "model (cv2.CALIB_TILTED_MODEL) if the resulting precision loss "
            "matters.",
            _d(12),
            _d(13),
        )

    return OpenCVPinholeCameraModelParameters(
        resolution=np.array([int(resolution_wh[0]), int(resolution_wh[1])], dtype=np.uint64),
        shutter_type=shutter_type,
        principal_point=np.array([float(K_arr[0, 2]), float(K_arr[1, 2])], dtype=np.float32),
        focal_length=np.array([float(K_arr[0, 0]), float(K_arr[1, 1])], dtype=np.float32),
        radial_coeffs=np.array([k1, k2, k3, k4, k5, k6], dtype=np.float32),
        tangential_coeffs=np.array([p1, p2], dtype=np.float32),
        thin_prism_coeffs=np.array([s1, s2, s3, s4], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------


def solve_static_extrinsic_mean(
    rig_interpolator: PoseInterpolator,
    sensor_world_poses: np.ndarray,
    sensor_timestamps_us: np.ndarray,
) -> np.ndarray:
    """Solve a static ``T_sensor_rig`` from per-frame world-frame sensor poses.

    Canonical derivation for sensors whose native calibration is published
    only as world-frame per-image / per-sweep poses (rather than a direct
    sensor-to-vehicle extrinsic):

      1. Interpolate the dense rig trajectory at each ``sensor_timestamps_us``
         using NCore's ``PoseInterpolator`` (SLERP + linear).
      2. Solve ``T_sensor_rig_i = se3_inverse(T_rig_world_i) @ T_sensor_world_i``
         per frame.
      3. Average rotations via SVD (chordal L2 / quaternion mean) and
         translations via arithmetic mean to get a single static extrinsic.

    This avoids the "single-frame extrinsic" anti-pattern (using only
    frame 0's pair), which freezes any instantaneous timing/calibration
    noise into the extrinsic.

    Args:
        rig_interpolator: ``PoseInterpolator`` built from the dense
            ``T_rig_world`` trajectory.
        sensor_world_poses: ``[N, 4, 4]`` float64 array of ``T_sensor_world``
            poses (one per sensor image / sweep).
        sensor_timestamps_us: ``[N]`` uint64 array of per-frame timestamps
            (microseconds, same time base as the rig trajectory).

    Returns:
        ``[4, 4]`` float64 array -- the averaged static ``T_sensor_rig``.

    Raises:
        ValueError: If ``sensor_timestamps_us`` is empty. Reduction
            over an empty per-frame stack emits a ``RuntimeWarning``
            and returns identity rotation + ``NaN`` translation
            without raising; reject at the boundary instead so the
            failure surfaces at the converter-author call site.
    """
    if sensor_timestamps_us.size == 0:
        raise ValueError("sensor_timestamps_us must be non-empty.")
    T_rig_world = rig_interpolator.interpolate_to_timestamps(sensor_timestamps_us, dtype=np.float64)
    T_sensor_rig_per_frame = np.einsum("nij,njk->nik", se3_inverse(T_rig_world), sensor_world_poses.astype(np.float64))
    # Chordal L2 rotation mean via SVD on stacked rotation matrices.
    R_sum = T_sensor_rig_per_frame[:, :3, :3].sum(axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R_mean = U @ D @ Vt
    t_mean = T_sensor_rig_per_frame[:, :3, 3].mean(axis=0)
    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean
    return T_mean
