# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers and threshold constants for the validate_* modules."""

from __future__ import annotations

import logging
import os
import re
import tempfile

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ncore_validation.utils import MAX_LIDAR_MODEL_ANGULAR_ERROR_DEG  # noqa: F401 -- re-exported for validate_* modules


log = logging.getLogger("validate")

PASS_ICON = "\u2713"  # checkmark
FAIL_ICON = "\u2717"  # cross
WARN_ICON = "!"

# ---------------------------------------------------------------------------
# Validation threshold constants
# ---------------------------------------------------------------------------

MAX_STATIC_POSE_TRANSLATION_M = 20.0
"""Sanity bound on static sensor->rig translation norms (meters)."""

MAX_LIDAR_DISTANCE_M = 300.0
"""Maximum plausible LiDAR range return (meters)."""

LIDAR_DIRECTION_TAUTOLOGY_THRESHOLD_DEG = 0.01
"""Tautology gate for ``warn_lidar_direction_collapsed_to_model``.

Stored ``direction`` agreeing with the model-derived direction
(``elements_to_sensor_rays(model_element)``) below this angular tolerance
*and* with ``row_azimuth_offsets_rad`` all-zero indicates the converter is
storing model-derived directions on a rotor-phase-only schedule -- the
LiDAR-smearing failure mode where per-beam offsets ``Δ`` are silently
dropped.  Threshold is well above the ~5e-4 deg float32 quantization noise
of XYZ-derived directions and well below the ~5e-7 deg agreement between
stored and model when both come from the same trig formula, so it cleanly
separates "stored = model" from "stored = xyz / ||xyz||"."""

MIN_EGO_DISPLACEMENT_FOR_CLOCK_CHECK_M = 1.0
"""Minimum ego displacement (meters) before the camera-follows-ego clock-sync
warning is relevant.  Below this the vehicle is effectively stationary."""

MIN_RIG_FORWARD_ALIGNMENT = 0.95
"""Minimum dot-product between the rig's +x axis and the travel direction.
Values below this indicate the rig frame convention may have +x pointing
somewhere other than forward, which can affect NuRec priors and ncore_vis.
This is the threshold for the `warn_rig_forward_not_x` soft warning."""

MIN_RIG_FORWARD_ALIGNMENT_ERROR = 0.9
"""Hard-error threshold for ``check_rig_forward_alignment``.

Derived from NCore's rig-frame convention
(`ncore/docs/data/conventions.rst`: +X forward, +Y left, +Z up, origin at
middle of rear axle). When the median of
``dot(R_rig_world @ [1, 0, 0], travel_direction)`` drops below 0.9 the
converter has swapped a rig axis (e.g. stored a LiDAR-frame extrinsic in
place of a rig-frame one, or rotated the rig by ~90 deg). The threshold
is lower than the warning threshold of 0.95 to tolerate brief
reverse-driving or sharp-turn segments without false positives."""

MIN_RIG_FORWARD_DISPLACEMENT_M = 0.05
"""Minimum per-segment displacement (meters) to include a consecutive-pose
pair in the rig-forward alignment median. Filters out noise and rotations
in place. 50 mm is well above typical pose/GPS/SLAM jitter (sub-cm) yet
low enough to cover dense-waypoint low-speed scenes (e.g. stopped in
traffic: a 20 s sequence with ~1200 waypoints and 7.6 m total travel has
zero segments above 100 mm but 40 above 50 mm)."""

MIN_RIG_FORWARD_SAMPLES = 8
"""Minimum number of moving-segment samples required for the rig-forward
alignment hard-error check to fire. Parking-only, static-rig, and pivot-
dominant sequences produce degenerate medians below this count; the check
skips cleanly (no warning, no error) rather than false-positive."""

MIN_RIG_FORWARD_ARCLENGTH_M = 1.0
"""Minimum cumulative translation (meters) across the sequence for the
rig-forward alignment hard-error check to fire. Non-AV converters (SfM
reconstructions of a static scene, static-rig captures, etc.)
legitimately produce near-stationary rig trajectories; the check skips
them cleanly."""

CUBOID_NEAR_RANGE_M = 50.0
"""Maximum distance (rig-frame centroid norm) for a cuboid to be considered
"near-range" for the LiDAR overlap check.  Cuboids beyond this distance
may have sparse or no LiDAR coverage and are excluded."""

CUBOID_GROUND_CLEARANCE_PROBE_M = 1.0
"""Height (meters) of the probe volume placed directly below each cuboid for
the ground-clearance check.  The probe shares the cuboid's oriented XY
footprint but extends this far below the cuboid bottom face.  LiDAR points
in this volume indicate the cuboid is floating above its object."""

CUBOID_FLOATING_MEDIAN_RATIO = 0.25
"""Warn when the median *below-ratio* across near-range cuboids exceeds this
value.  The below-ratio for a single cuboid is
``n_below / (n_inside + n_below)`` where *n_inside* counts LiDAR points
inside the cuboid and *n_below* counts points in the below-cuboid probe.
A correctly placed cuboid has near-zero below-ratio; a cuboid shifted up by
``height / 2`` (≈ 0.7 m for a car) pushes lower-body LiDAR returns into the
probe, raising the median well above 0.25."""

MIN_NEAR_RANGE_CUBOID_HIT_RATE = 0.5
"""Minimum fraction of near-range cuboids (within ``CUBOID_NEAR_RANGE_M``)
that must contain at least one LiDAR point.  Falling below this threshold
indicates a likely extrinsic or transform-chain error (e.g. rotated
``T_lidar_rig``).  Correct transforms typically yield 80%+ for near-range
objects; 50% is a conservative lower bound."""

MIN_NEAR_RANGE_CUBOIDS_FOR_CHECK = 5
"""Minimum number of near-range cuboids required before the overlap check
triggers.  Below this the sample is too small to be meaningful."""

MIN_CUBOID_OBB_INLIER_RATIO = 0.2
"""Minimum fraction of LiDAR points inside a cuboid's rig-frame
axis-aligned bounding box (AABB) that must also fall inside the actual
oriented bounding box (OBB).  Detects cuboid geometry bugs that
``check_cuboid_lidar_overlap`` misses because its threshold is merely
"at least one point hits the OBB":

  * A correctly-populated OBB has a high AABB->OBB ratio because the
    physical object that generated the LiDAR returns lives inside the
    OBB itself.
  * An OBB rotated 90 deg around Z (e.g. dim.x <-> dim.y swap without a
    compensating rot change) has roughly the same AABB but almost no
    LiDAR points inside the rotated OBB, so the ratio collapses.
  * Wrong centroid / wrong reference frame / wrong dim magnitude also
    shift the OBB off the real object and drop the ratio.

Dataset-agnostic: no class-label filter (label vocabularies differ
across datasets) and no geometric filter. Cubic cuboids trivially pass
because their AABB equals their OBB at rotation 0; that is correct
behavior, not a false positive."""

MIN_CUBOIDS_FOR_OBB_CHECK = 5
"""Minimum number of cuboids with LiDAR coverage (AABB contains at least one
point) required before the OBB inlier ratio check triggers."""

CUBOID_POSE_GAP_MAX_US = 1000
"""Fail when a pose-dependent cuboid's ``reference_frame_timestamp_us`` is
more than this many microseconds from the nearest pose waypoint.  A large
gap means NuRec must interpolate the ego pose instead of using the exact
pose the labels were defined relative to, causing spatial offset on dynamic
objects (vehicle speed x time gap).  See ``check_cuboid_pose_gap`` for which
reference frames are pose-dependent."""

MAX_POSE_GAP_SWEEP_RATIO = 1.0
"""Warn if the maximum gap between consecutive rig->world pose timestamps
is >= this fraction of the median LiDAR sweep duration.  A ratio >= 1.0
means the trajectory has at least one gap larger than a full sweep, causing
staircase interpolation during NuRec motion compensation (rowing)."""

MIN_LIDAR_CENTROID_DRIFT_M = 10.0
"""Minimum absolute drift (meters) between first-frame and last-frame LiDAR
point cloud centroids (in sensor coordinates) to trigger the world-frame
warning.  Guards against false positives on stationary vehicles."""

MIN_CENTROID_DRIFT_EGO_RATIO = 0.5
"""Warn if centroid drift / ego displacement exceeds this ratio.  World-frame
data stored as sensor-frame will have drift ~= ego displacement (ratio ~1.0);
true sensor-frame data stays near the origin (ratio ~0)."""

POSE_TRANSLATION_MAX_M = 10_000.0
"""Warn when any rig->world translation component exceeds this magnitude.
Poses with large absolute coordinates have not been re-referenced to the
first ego pose, causing float32 precision loss in NuRec motion compensation."""

MAX_POSE_ANGULAR_RATE_DEG_S = 180.0
"""Maximum physically plausible angular rate between consecutive rig->world
pose waypoints (degrees per second).  Values above this indicate corrupted
pose entries (e.g. sampled from an invalid per-pixel grid).  180 deg/s is
well beyond what any road vehicle can sustain."""

SWEEP_CENTER_OFFSET_RATIO = 0.35
"""If the median offset between LiDAR sweep start timestamps and the nearest
pose waypoint is >= this fraction of the sweep duration, warn that sweeps
may be centered rather than starting at the frame timestamp.  A perfectly
centered sweep would produce a ratio of ~0.5; 0.35 catches moderate cases
while allowing for normal jitter."""

LIDAR_CAMERA_SWEEP_OFFSET_RATIO = 0.35
"""Hard-fail when LiDAR sweep midpoints are systematically *earlier* than camera
exposure midpoints (same frame index) by >= this fraction of the sweep duration.
In synchronized AV capture with rolling-shutter cameras, the LiDAR sweep
midpoint and camera exposure midpoint should be within a few ms of each other.
A large negative offset (>= 35% of sweep) indicates the converter treated the
frame timestamp as sweep center instead of sweep start, shifting every per-ray
timestamp by ~half a sweep duration.  This causes severe motion-compensation
blur across the entire scene.

Note: with global-shutter cameras, exposure start ≈ end (< 1 ms window), so
the camera midpoint sits near the LiDAR sweep *start*.  A correctly aligned
sweep produces a *positive* ~50 ms offset (LiDAR midpoint after camera
midpoint), while a centered sweep produces ~0 ms — neither triggers this
negative-offset check.  The pose-based ``warn_sweep_timestamps_centered``
may catch that case instead."""

LIDAR_CAMERA_POSITIVE_OFFSET_RATIO = 0.75
"""Hard-fail when LiDAR sweep midpoints are systematically *later* than camera
exposure midpoints by >= this fraction of the sweep duration.  A positive offset
up to ~50% is normal for global-shutter cameras triggered at sweep start (camera
midpoint near sweep start, LiDAR midpoint near sweep center).  Only offsets
>= 75% indicate a clear timestamp interpretation error."""

MAX_CUBOID_OBSERVATIONS = 200
"""Maximum observations sampled by each cuboid-LiDAR check."""

MIN_LIDAR_FORWARD_RAY_ALIGNMENT = 0.85
"""Hard threshold for ``check_lidar_forward_ray_in_rig``: best
``dot(T_lidar_rig.R @ d_model, [1, 0, 0])`` across columns (~32 deg
tolerance).  Below this, no column anywhere lands near rig +X (e.g.
a near-90 deg ``T_lidar_rig`` rotation, or a totally-wrong extrinsic).
Wholesale residual yaw between ``T_lidar_rig.R`` and ``column_azimuths_rad``
is **not** caught here -- on a 360 deg spinner some column always lands
near rig +X regardless of how the cloud is rotated; see
``warn_lidar_forward_ray_rig_axis_residual`` for the cardinal-spread
sub-hard warning."""

MIN_LIDAR_FORWARD_RAY_AXIS_RESIDUAL = 0.97
"""Warning threshold for ``warn_lidar_forward_ray_rig_axis_residual``
(~14 deg), applied to the *worst* cardinal dot product across the four
target model azimuths ``{0, pi/2, pi, 3*pi/2}``.  Catches both small
mount-bracket double-counts AND wholesale residual yaw between
``T_lidar_rig.R`` and ``column_azimuths_rad``; legitimate Option-B
converters (column-0 ray aligned with sensor-native +X rather than rig +X)
also trip this warning, so cross-confirm with
``check_cuboid_lidar_overlap`` before classifying as a converter bug."""

MIN_LIDAR_AZIMUTH_COVERAGE_RAD = np.pi
"""Minimum azimuth span (180 deg) required before forward-ray
alignment checks apply.  Skips legitimate sector-coverage side mounts."""

FULL_ROTATION_COVERAGE_MIN_RAD = 0.95 * 2.0 * np.pi
"""Minimum ``column_azimuths_rad`` span (342 deg) required by
``warn_lidar_spin_frequency_mismatch``, whose estimator assumes one stored
frame is one revolution.  A sector sensor's frame covers ``coverage / 2 pi``
of a turn, so a correct declared rate deviates from the window-implied rate by
``1 - coverage / 2 pi`` -- at this bound 5%, inside
``SPIN_FREQUENCY_MISMATCH_RATIO``, so coverage alone cannot trigger the
warning.  A wrap-aware span also reads one column step short of
``2 pi`` on a full-rotation sensor, which this bound absorbs."""

SPIN_FREQUENCY_MISMATCH_RATIO = 0.1
"""Warn when ``spinning_frequency_hz`` deviates from the rate implied by the
stored frame window by at least this fraction.  ``1 / frame duration`` is the
definition NCore's reference converters use when they derive the field from
data (``tools/data_converter/nuscenes/converter.py``).

The band is wide because a declared nominal rate legitimately differs from the
measured window by a percent or two (rounded datasheet value, rotor servo
error), while the derivations this check targets -- frame cadence on a
decimated stream, a fit spanning frames, a datasheet default for another
variant of the sensor -- miss by tens of percent.  It is also what keeps
``FULL_ROTATION_COVERAGE_MIN_RAD`` from warning by construction on a sensor
that sits right at the coverage bound."""

MAX_CUBOID_RIG_DISTANCE_P95_M = 300.0
"""Hard threshold (meters) for ``check_cuboid_rig_proximity``: p95
of ``|| c_rig ||_2`` across cuboid observations.  Aligned with
``MAX_LIDAR_DISTANCE_M`` so long-range automotive annotations stay
inside the envelope, while frame-skew bugs (UTM/ECEF origins) land
in the thousands.  Tune for non-AV scene scales."""

MIN_CUBOID_RIG_PROXIMITY_OBSERVATIONS = 5
"""Minimum observations before ``check_cuboid_rig_proximity`` fires
(p95 unreliable below this)."""

MODEL_ELEMENT_ZERO_FRACTION_WARN = 0.99
"""Warn when this fraction or more of per-ray ``model_element`` entries
are ``(0, 0)``.  A common converter mistake is leaving ``model_element``
as an ``np.zeros((N, 2), uint16)`` placeholder, which slips through
``check_lidar_model_accuracy`` when every stored direction happens to
align with the ``(row=0, col=0)`` cell, and is undetectable downstream
once per-point (row, col) information is lost.  Threshold is a near-1.0
fraction rather than exactly 1.0 so the warning still fires when a
handful of valid rays coincidentally land at (0, 0)."""

ROW_ELEVATIONS_UNIFORM_STRIDE_RAD = 1e-3
"""Warn when ``np.std(np.diff(row_elevations_rad))`` falls below this
tolerance (radians), indicating the array is (nearly) uniformly spaced
-- typical of a ``np.linspace(vfov_min, vfov_max, n_rows)`` fallback.
Real spinning LiDARs have per-beam elevation tables with non-uniform
spacing.  A uniform grid is mathematically valid but usually means the
per-beam calibration was not consulted; the converter should re-read
the raw calibration before keeping a synthetic grid."""

COLUMN_AZIMUTHS_UNIFORM_STRIDE_RAD = 1e-4
"""Warn when ``np.std(np.diff(column_azimuths_rad))`` falls below this
tolerance (radians), indicating the array is (nearly) uniformly spaced
-- typical of a ``np.linspace(0, 2*pi, n_columns, endpoint=False)``
fallback.  Real free-spinning LiDARs ship per-frame azimuth schedules
that, when collapsed to the unique-azimuth set, have a non-uniform
per-column stride (dual-return run patterns, near-field channel bursts,
non-uniform schedules).  A uniform grid is a synthesis artifact in
exactly the same sense as a synthetic ``row_elevations_rad``: it
discards the sensor's actual per-column firing structure and replaces
it with a fiction that happens to share an average angular resolution.
Threshold is one order of magnitude tighter than the row-elevations
threshold because column-azimuth strides are inherently smaller (a
typical 3600-column rotor has a ``2*pi / 3600 ~= 1.7e-3 rad`` step)."""

MAX_FISHEYE_MAX_ANGLE_DEVIATION_DEG = 5.0
"""Warn when a stored OpenCV-fisheye ``max_angle`` deviates from the
calibration-derived value (``OpenCVFisheyeCameraModelParameters.compute_max_angle()``)
by more than this many degrees.  Too large pushes NuRec to evaluate the
Kannala-Brandt polynomial outside its calibration domain (geometric
distortion / edge blur); too small silently drops image-edge pixels from
reconstruction."""

EGO_MASK_KEY_PATTERN = re.compile(r"ego", re.IGNORECASE)
"""Matches mask key names that look ego-related (e.g. ``"ego_vehicle"``).
NuRec expects the ego-vehicle mask under the exact key ``"ego"``;
any other ego-like name is silently ignored."""

NEAR_ZERO_SENSOR_TRANSLATION_M = 0.01
"""Warn when **all** static sensor-to-rig translations have Euclidean magnitude
below this threshold (1 cm).

A converter that missed the dataset's calibration file commonly leaves all
sensor poses as identity (zero translation).  A single sensor being near the
rig origin is plausible, but all sensors simultaneously at (0, 0, 0) is almost
always a calibration-load bug: real vehicle rigs have sensors at different
mount heights and fore-aft positions."""

MAX_CAMERA_OPTICAL_AXIS_TILT_Z = 0.7
"""Warning bound on ``|T_camera_rig.R[:, 2].z|`` after unit-norm renormalization
(~45 deg from horizontal).  Catches a missing source-frame to NCore-camera-frame
basis change in ``T_camera_rig``: NCore camera convention is +Z optical axis,
so a converter that forgets the basis change for a +X- or +Y-principal source
camera leaves the camera looking straight up or down.  Real AV cameras sit
within ~20 deg of horizontal, so 0.7 leaves headroom for roof-rack or
up-tilted mounts while still catching the failure mode."""

MAX_CAMERA_OPTICAL_AXIS_AZIMUTH_DEG = 60.0
"""Warning bound (degrees) on the gap between a camera's id-implied mounting
direction and the actual optical-axis azimuth in the rig frame.  Catches
whole-rig yaw bugs in ``T_camera_rig`` (the converter built the static
extrinsic against a frame whose +X is not aligned with rig +X = travel
direction, validated independently by ``check_rig_forward_alignment``).
60 deg is intentionally wide because real AV cameras with overlapping FOV
are routinely 20-40 deg off their nominal direction (e.g. fish-eye corners);
the threshold still catches the 90 / 180 deg whole-rig rotation bugs."""


# ---------------------------------------------------------------------------
# validate_result.yaml schema and collector
# ---------------------------------------------------------------------------


@dataclass
class FailedCheck:
    """One hard-fail check recorded during validation.

    Attributes:
        tier: Section header active when the check ran (e.g. ``"V4 Structure"``,
            ``"NuRec Readiness"``, ``"Data Quality"``).
        group: Sub-group header within the tier (e.g. ``"Cameras"``,
            ``"LiDAR"``, ``"Cross-Sensor"``), or ``None`` outside any group.
        name: The ``label`` argument originally passed to :func:`_check`.
        message: Verbatim failure text (the ``error_msg`` argument, or
            ``label`` when no custom message was supplied). The downstream
            validator enforces verbatim quoting of this string in the
            diagnosis YAML's ``unresolved_issues.converter[*].error_message``.
    """

    tier: str
    group: str | None
    name: str
    message: str


@dataclass
class WarningResult:
    """One non-fatal warning emitted during validation.

    Attributes:
        tier: Section header active when the warning fired.
        group: Sub-group header within the tier, or ``None``.
        message: Verbatim warning text (the ``label`` argument passed to
            :func:`_warn`).
    """

    tier: str
    group: str | None
    message: str


@dataclass
class ValidateResult:
    """Typed contents of the ``validate_result.yaml`` sidecar.

    Attributes:
        schema_version: :data:`VALIDATE_RESULT_SCHEMA_VERSION` at write time.
        program: :data:`VALIDATE_RESULT_PROGRAM`; self-identifies the
            producing tool so the validator can reject foreign YAMLs.
        program_version: ``nvidia-ncore`` package version captured via
            :mod:`importlib.metadata`, or a fallback when lookup fails.
        started_at: ISO-8601 UTC timestamp at which validation began.
        ended_at: ISO-8601 UTC timestamp at which validation finished
            (including any clean exit after a hard failure).
        duration_sec: Wall-clock validation duration in seconds.
        exit_code: Mirror of the process exit code: ``0`` on a clean
            pass, any non-zero value on failure. Cross-checked against
            the agent-reported ``validate_run.exit_code`` in the
            diagnosis.
        passed: ``exit_code == 0``. Carried as a typed field so the
            validator does not need to infer pass/fail from the exit
            code alone.
        component_groups: The ``--component-group`` arguments that were
            validated (as plain strings for YAML serialization).
        sample_frames: Value of the ``--sample-frames`` flag, for
            provenance. ``None`` when the flag was not set.
        failed_checks: One entry per failed :func:`_check`, in emission
            order. The validator quotes ``message`` verbatim against the
            diagnosis's ``unresolved_issues.converter[*].error_message``.
        warnings: One entry per :func:`_warn` call, in emission order.
    """

    schema_version: int
    program: str
    program_version: str
    started_at: str
    ended_at: str
    duration_sec: float
    exit_code: int
    passed: bool
    component_groups: list[str]
    sample_frames: int | None
    failed_checks: list[FailedCheck] = field(default_factory=list)
    warnings: list[WarningResult] = field(default_factory=list)


class ResultCollector:
    """In-memory recorder for tier/group/check/warning emissions.

    A single active collector at a time is held in the module-global
    :data:`_collector`. Activation and deactivation flow through
    :func:`start_collection` and :func:`stop_collection`; the
    :func:`_section`, :func:`_group`, :func:`_check`, and :func:`_warn`
    helpers append to the active collector when one exists and are a
    pure log-only no-op otherwise. Keeping the wiring inside the helpers
    means the seven ``validate_*.py`` modules need no changes to gain
    structured output.

    **Concurrency contract**: not thread-safe, not re-entrant. The
    module-global is sized for ``validate.py``'s single-shot CLI: one
    process, one ``start_collection`` / ``stop_collection`` pair around
    one validation run. A second ``start_collection`` overwrites the
    global without merging, and concurrent helper calls from multiple
    threads would tag entries with whichever ``current_tier`` /
    ``current_group`` happened to be set most recently. Wrap the
    collector in per-call instances and pass it through call stacks if
    a future caller needs parallel or nested validation runs.

    Attributes:
        current_tier: Most recent :func:`_section` title, used to tag
            subsequent checks and warnings. Empty string before the
            first ``_section`` call.
        current_group: Most recent :func:`_group` title under the
            current tier, or ``None`` after a ``_section`` resets it.
        failed_checks: Captured failed checks in emission order.
        warnings: Captured warnings in emission order.
    """

    def __init__(self) -> None:
        """Initialize an empty collector with no tier or group context."""
        self.current_tier: str = ""
        self.current_group: str | None = None
        self.failed_checks: list[FailedCheck] = []
        self.warnings: list[WarningResult] = []


_collector: ResultCollector | None = None


def start_collection() -> ResultCollector:
    """Begin collecting check and warning records.

    Returns:
        The newly active :class:`ResultCollector` (also stored in the
        module-global :data:`_collector`).
    """
    global _collector
    _collector = ResultCollector()
    return _collector


def stop_collection() -> None:
    """Stop collecting records. Subsequent helper calls are log-only."""
    global _collector
    _collector = None


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix.

    Returns a second-precision timestamp matching the format already used
    by step-run entries elsewhere in the validators (e.g.
    ``"2026-05-06T22:00:00Z"``). Sub-second granularity lives in the
    sibling ``duration_sec`` field, not in these timestamps.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump_validate_result(path: Path | str, result: ValidateResult) -> None:
    """Serialize a :class:`ValidateResult` to YAML at ``path`` atomically.

    Writes through a same-directory temp file then ``os.replace`` so
    readers see either the prior file or the complete new one, never a
    truncated intermediate. ``sort_keys=False`` keeps the on-disk field
    order matching the dataclass declaration.

    Args:
        path: Output file path. Parent directories must already exist.
        result: Result record to serialize. Nested dataclasses
            (:class:`FailedCheck`, :class:`WarningResult`) are recursively
            converted via :func:`dataclasses.asdict`.
    """
    p = Path(path)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=p.parent,
        prefix=f".{p.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            yaml.safe_dump(asdict(result), tmp, default_flow_style=False, sort_keys=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, p)
    except BaseException:
        # Cleanup on any failure (incl. KeyboardInterrupt / SystemExit) so
        # we don't leave .tmp residues next to the sidecar.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Section / check / warn helpers (wired to the active ResultCollector)
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    """Log a validation tier header (e.g. ``V4 Structure`` or ``NuRec Readiness``).

    When a collector is active, also resets the current group to ``None``
    so warnings and checks emitted before the next :func:`_group` call
    are tagged with the tier but no sub-group.
    """
    log.info(f"\n{title}:")
    if _collector is not None:
        _collector.current_tier = title
        _collector.current_group = None


def _group(title: str) -> None:
    """Log a sub-group header under a tier (e.g. ``Cameras`` or ``LiDAR``).

    When a collector is active, updates the current group so subsequent
    checks and warnings inherit the tier/group context.
    """
    log.info(f"  {title}:")
    if _collector is not None:
        _collector.current_group = title


def _check(label: str, ok: bool, errors: list[str], error_msg: str | None = None) -> bool:
    """Log a check result and, if failed, append the error message.

    Args:
        label: Human-readable check description shown in output.
        ok: Whether the check passed.
        errors: Mutable list to which a failure message is appended on ``not ok``.
        error_msg: Custom failure message; defaults to *label* when ``None``.

    Returns:
        The *ok* value, for convenience in further branching.
    """
    log.info(f"    {PASS_ICON if ok else FAIL_ICON} {label}")
    if not ok:
        msg = error_msg or label
        errors.append(msg)
        if _collector is not None:
            _collector.failed_checks.append(
                FailedCheck(
                    tier=_collector.current_tier,
                    group=_collector.current_group,
                    name=label,
                    message=msg,
                )
            )
    return ok


def _warn(label: str) -> None:
    """Log a non-fatal warning line (does not append to ``errors``).

    When a collector is active, also records a :class:`WarningResult` so
    the warning becomes a first-class entry in ``validate_result.yaml``.
    """
    log.info(f"    {WARN_ICON} {label}")
    if _collector is not None:
        _collector.warnings.append(
            WarningResult(
                tier=_collector.current_tier,
                group=_collector.current_group,
                message=label,
            )
        )


def _check_sensor_frame_ts(readers: dict, sensor_type: str, errors: list[str]) -> None:
    """Validate NCore frame timestamp invariants for a set of sensor readers.

    NCore's writer enforces: (1) per-frame start <= end, (2) starts and ends
    strictly increasing across frames, (3) end timestamps unique (frame key).

    Args:
        readers: Mapping of sensor id to reader (camera or LiDAR).  Each
            reader must expose a ``frames_timestamps_us`` attribute with
            shape ``(F, 2)``.
        sensor_type: Label used in error messages (e.g. ``"camera"``,
            ``"LiDAR"``).
        errors: Mutable list to which failure messages are appended.
    """
    bad_order: list[str] = []
    bad_monotonic: list[str] = []
    bad_unique: list[str] = []
    for sensor_id, reader in readers.items():
        ts = reader.frames_timestamps_us
        if ts.size == 0:
            continue
        if not (ts[:, 0] <= ts[:, 1]).all():
            n_bad = int((ts[:, 0] > ts[:, 1]).sum())
            bad_order.append(f"{sensor_id} ({n_bad} frames with start > end)")
        if ts.shape[0] > 1:
            if not np.all(ts[:-1, 0] < ts[1:, 0]) or not np.all(ts[:-1, 1] < ts[1:, 1]):
                bad_monotonic.append(sensor_id)
            if len(np.unique(ts[:, 1])) != ts.shape[0]:
                bad_unique.append(sensor_id)
    _check(
        "frame timestamps ordered (start <= end)",
        not bad_order,
        errors,
        f"{sensor_type} frame timestamp ordering violated: {'; '.join(bad_order)}",
    )
    _check(
        "frame timestamps strictly increasing across frames",
        not bad_monotonic,
        errors,
        f"{sensor_type} frame starts/ends not strictly increasing: {'; '.join(bad_monotonic)}",
    )
    _check(
        "frame end timestamps unique",
        not bad_unique,
        errors,
        f"duplicate {sensor_type} frame end timestamps (NCore keys frames by end ts): {'; '.join(bad_unique)}",
    )


def _sample_frame_indices(n_frames: int, sample_frames: int | None) -> np.ndarray:
    """Return frame indices to evaluate, optionally evenly-spaced subsampled.

    When ``sample_frames`` is ``None`` or ``>= n_frames``, returns
    ``np.arange(n_frames)`` (full sweep).  Otherwise returns
    ``sample_frames`` evenly-spaced indices in ``[0, n_frames - 1]``.
    Used by the iteratively-expensive LiDAR sweeps
    (:func:`check_lidar_frame_data`, :func:`check_lidar_model_accuracy`)
    so iteration cycles don't pay the full cost during early
    converter development.  Cheap structural checks always run on the
    full set and are unaffected by this option.
    """
    if n_frames <= 0:
        return np.empty(0, dtype=int)
    if sample_frames is None or sample_frames <= 0 or sample_frames >= n_frames:
        return np.arange(n_frames, dtype=int)
    return np.linspace(0, n_frames - 1, sample_frames, dtype=int)
