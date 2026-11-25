# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side OpenCV fisheye domain policy shared by the GPU camera backends."""

import math

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from numpy.typing import ArrayLike

from ncore.impl.data.util import compute_max_angle_with_monotonicity


# At least the GPU solvers' absolute derivative floor. A relative margin also
# protects against cancellation in float32 Horner evaluation of the derivative.
_DERIVATIVE_RELATIVE_MARGIN = 1e-6


@dataclass(frozen=True)
class FisheyeInverseDomain:
    """Configured FOV and the separately computed, conditioned inverse domain."""

    configured_max_angle: float
    inverse_monotonic_limit: float
    max_angle: float


def opencv_fisheye_domain(radial_coeffs: ArrayLike, max_angle: float) -> FisheyeInverseDomain:
    """Validate float32 coefficients and limit the configured FOV to a safe inverse.

    NCore supplies the shared polynomial root finder. Its inverse deliberately
    extrapolates beyond the configured FOV; NuRec instead preserves its own FOV
    and finite-sentinel policy. This metadata must be rebuilt if calibration changes.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        coeffs = np.asarray(radial_coeffs, dtype=np.float32)
    if coeffs.shape != (4,) or not np.isfinite(coeffs).all():
        raise ValueError("OpenCV fisheye requires four finite float32 distortion coefficients")
    if not math.isfinite(max_angle) or not 0.0 < max_angle <= math.pi:
        raise ValueError("OpenCV fisheye max_angle must be finite and in (0, pi]")
    return _inverse_domain(tuple(float(k) for k in coeffs), float(max_angle))


def opencv_fisheye_max_angle(radial_coeffs: ArrayLike, max_angle: float) -> float:
    """Return the configured FOV restricted to the conditioned increasing branch."""
    return opencv_fisheye_domain(radial_coeffs, max_angle).max_angle


@lru_cache(maxsize=256)
def _inverse_domain(coeffs: tuple[float, ...], max_angle: float) -> FisheyeInverseDomain:
    # Find the first point where g'(theta) reaches a positive relative margin:
    # g'(theta) > eps * (1 + sum(abs(d_i) * theta**i)). This is the derivative
    # of the polynomial below. The actual projection coefficients are unchanged.
    # Moving the boundary before g'=0 also separates repeated stationary roots
    # that otherwise can acquire tiny imaginary parts and evade NCore's real-root
    # filter. No second polynomial root implementation is needed here.
    safety_poly = np.zeros(10, dtype=np.float64)
    safety_poly[1::2] = (1.0, *coeffs)
    safety_poly -= _DERIVATIVE_RELATIVE_MARGIN * np.abs(safety_poly)
    monotonic_limit = compute_max_angle_with_monotonicity(safety_poly, np.inf)
    inverse_limit = min(math.pi, monotonic_limit)
    bound = min(max_angle, inverse_limit)
    if monotonic_limit <= max_angle or float(np.float32(bound)) >= inverse_limit:
        # A float32 cast may round above the calculated boundary. Stay strictly
        # inside it, even when the configured FOV is just below the inverse limit
        # or when pi rounds upward. Preserve the original configured FOV separately.
        bound = float(np.nextafter(np.float32(bound), np.float32(0.0)))
    radius = bound * np.polynomial.polynomial.polyval(bound * bound, [1.0, *coeffs])
    with np.errstate(over="ignore", invalid="ignore"):
        finite_radius = np.isfinite(np.float32(radius))
    if bound <= 0.0 or not finite_radius or radius <= 0.0:
        raise ValueError("OpenCV fisheye calibration has no finite increasing projection domain")
    return FisheyeInverseDomain(max_angle, inverse_limit, bound)
