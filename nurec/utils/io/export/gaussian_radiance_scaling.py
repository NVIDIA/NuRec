# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gaussian Radiance Scaling

Apply a scalar radiance scale to a Gaussian model's Spherical Harmonic
coefficients in place. This is the asset-level brightness adjustment used
during USD export to match Gaussians to the radiance level of an external
USD scene; it is the permanent, asset-baked counterpart to the runtime
PPISP responsivity multiplier.

Forward SH evaluation of a Gaussian's RGB output is::

    rgb = features_albedo * C0 + 0.5 + sum_k Y_k(view) * features_specular_k

To produce ``s * rgb`` from an unmodified forward eval, both the
view-dependent specular bands and the DC (albedo) band must be scaled.
The specular bands are linear in their coefficients, but the DC band
carries an implicit ``+0.5`` constant from the standard SH<->RGB convention
(``RGB2SH(rgb) = (rgb - 0.5) / C0``), so it needs an extra constant offset
to compensate. The closed-form rewrite is::

    features_specular -> s * features_specular
    features_albedo   -> s * features_albedo + (s - 1) * 0.5 / C0

This module implements that rewrite with fail-loud validation.
"""

from __future__ import annotations

import logging
import math

from typing import Iterable, Protocol, Tuple, runtime_checkable

import torch

from nurec.models.gaussians.utils import C0


log = logging.getLogger(__name__)


@runtime_checkable
class SHCoefficientsBearer(Protocol):
    """Structural type for any object that exposes SH coefficient tensors.

    The radiance scaling and warm-start logic only need to read these
    two tensors and call in-place methods on them (``mul_``, ``add_``,
    ``copy_``); they never reassign ``model.features_albedo`` itself.
    The Protocol therefore declares both attributes as **read-only
    properties** returning ``torch.Tensor``. This is intentional and
    matters for type-checking:

    * As a regular attribute annotation (``features_albedo: torch.Tensor``),
      mypy treats the slot as invariant (readable *and* writable), so a
      class whose instance attribute is the narrower ``nn.Parameter``
      fails the structural check despite ``nn.Parameter`` being a
      ``torch.Tensor`` subclass.
    * As a read-only property, the slot is covariant on reads, and any
      class with an instance attribute, method, or property whose
      observed type is ``torch.Tensor`` (or a subclass like
      ``nn.Parameter``) satisfies the Protocol cleanly.

    Production: satisfied by ``SHGaussianModel`` and its subclasses
    (``RigidGaussianModel``, ``DeformableGaussianModel``,
    ``DeformableRigidAssetModel``), which expose these as
    ``nn.Parameter`` instance attributes.
    """

    @property
    def features_albedo(self) -> torch.Tensor: ...

    @property
    def features_specular(self) -> torch.Tensor: ...


def apply_radiance_scale_to_sh(model: SHCoefficientsBearer, scale: float) -> None:
    """In-place scale the SH-evaluated RGB output of ``model`` by ``scale``.

    The view-dependent specular bands are scaled directly. The DC band is
    scaled and offset to compensate for the implicit ``+0.5`` constant in
    the SH<->RGB conversion, so that a subsequent forward SH evaluation
    reproduces ``scale * rgb_original`` exactly.

    This is the single-node primitive. Composite-aware exporters should
    use :func:`apply_radiance_scale_to_sh_nodes` so every Gaussian node is
    scaled consistently.

    Args:
        model: Any object exposing ``features_albedo`` (shape ``[N, 3]``)
            and ``features_specular`` (shape ``[N, num_specular_dims]``)
            as ``torch.Tensor`` attributes. Typically an
            ``SHGaussianModel`` or a subclass.
        scale: Multiplicative radiance scale. Must be a strictly positive
            finite number. ``scale == 1.0`` is a documented no-op.

    Raises:
        TypeError: If ``scale`` is not numeric or if either SH parameter is
            not a ``torch.Tensor``.
        ValueError: If ``scale`` is non-finite or non-positive.
        AttributeError: If ``model`` is missing ``features_albedo`` or
            ``features_specular``.
    """
    _validate_radiance_scaling_inputs(model, scale)

    if scale == 1.0:
        log.info("Radiance scale = 1.0: no-op (SH coefficients unchanged)")
        return

    s = float(scale)
    dc_offset_compensation = _compute_dc_offset_compensation(s)
    _apply_validated_radiance_scale_to_sh(model, s, dc_offset_compensation)
    log.info(
        "Applied radiance scale %.6f to SH coefficients (DC offset compensated by %.6f)",
        s,
        dc_offset_compensation,
    )


def apply_radiance_scale_to_sh_nodes(nodes: Iterable[Tuple[str, SHCoefficientsBearer]], scale: float) -> None:
    """In-place scale SH coefficients for all Gaussian nodes in a model.

    This is the composite-aware variant of :func:`apply_radiance_scale_to_sh`.
    It validates every node before mutating any tensor, so a malformed node
    does not leave earlier nodes partially scaled.
    """
    _validate_radiance_scale(scale)

    node_list = list(nodes)
    if not node_list:
        raise ValueError("radiance scaling requires at least one Gaussian node")

    for node_id, node in node_list:
        try:
            _validate_sh_coefficients_bearer(node)
        except (AttributeError, TypeError) as exc:
            raise type(exc)(f"gaussian node {node_id!r}: {exc}") from exc

    if scale == 1.0:
        log.info("Radiance scale = 1.0: no-op (SH coefficients unchanged)")
        return

    s = float(scale)
    dc_offset_compensation = _compute_dc_offset_compensation(s)
    for _, node in node_list:
        _apply_validated_radiance_scale_to_sh(node, s, dc_offset_compensation)
    log.info(
        "Applied radiance scale %.6f to SH coefficients for %d Gaussian node(s) (DC offset compensated by %.6f)",
        s,
        len(node_list),
        dc_offset_compensation,
    )


def _validate_radiance_scaling_inputs(model: object, scale: object) -> None:
    """Fail-loud validation for :func:`apply_radiance_scale_to_sh` inputs.

    Validates the scalar first so callers receive a clear error before any
    model attribute access can fail.
    """
    _validate_radiance_scale(scale)
    _validate_sh_coefficients_bearer(model)


def _validate_radiance_scale(scale: object) -> None:
    """Validate the scalar radiance multiplier."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(f"scale must be a real number (int or float), got {type(scale).__name__}")
    if not math.isfinite(float(scale)):
        raise ValueError(f"scale must be finite, got {scale!r}")
    if float(scale) <= 0.0:
        raise ValueError(f"scale must be strictly positive, got {scale!r}")


def _validate_sh_coefficients_bearer(model: object) -> None:
    """Validate that ``model`` exposes the SH tensors required for scaling."""
    for attr_name in ("features_albedo", "features_specular"):
        if not hasattr(model, attr_name):
            raise AttributeError(
                f"model is missing required SH parameter '{attr_name}'. "
                "apply_radiance_scale_to_sh requires an SHGaussianModel-like "
                "model exposing features_albedo and features_specular."
            )
        param = getattr(model, attr_name)
        if not isinstance(param, torch.Tensor):
            raise TypeError(f"model.{attr_name} must be a torch.Tensor, got {type(param).__name__}")


def _apply_validated_radiance_scale_to_sh(
    model: SHCoefficientsBearer,
    scale: float,
    dc_offset_compensation: float,
) -> None:
    """Apply radiance scaling after caller-side validation."""
    with torch.no_grad():
        model.features_specular.mul_(scale)
        model.features_albedo.mul_(scale).add_(dc_offset_compensation)


def _compute_dc_offset_compensation(scale: float) -> float:
    """Return the constant to add to ``features_albedo`` when scaling by ``scale``.

    Derivation: the DC term of an SH-encoded RGB is ``features_albedo * C0 + 0.5``.
    Requiring ``features_albedo' * C0 + 0.5 == scale * (features_albedo * C0 + 0.5)``
    yields ``features_albedo' = scale * features_albedo + (scale - 1) * 0.5 / C0``.
    The trailing constant is what this helper returns.
    """
    return (float(scale) - 1.0) * 0.5 / C0
