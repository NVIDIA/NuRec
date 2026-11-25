# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-step update contract of the Gaussian layer classes and the Fourier-freeze gradient hook.

Every layer class must advance the Fourier-freeze counter (``_global_step_for_freeze``) in
``update_step_train_batch_start``; otherwise the backward hook on ``features_albedo`` keeps the
temporal Fourier bands (``features_albedo[..., 1:, :]``) masked for the whole run. Each override must
also keep its own per-step work and update every owned module exactly once per step.

The models are built with ``__new__`` plus ``nn.Module.__init__`` and only the attributes the per-step
update reads, the same convention as the other tests in this package; the full constructors need a
data source and cuboid tracks, which the per-step contract does not depend on.
"""

from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
import torch

from torch import nn

from nurec.models.gaussians.construction_barriers_model import ConstructionBarriersGaussianModel
from nurec.models.gaussians.gaussians_model import (
    DeformableGaussianModel,
    DeformableRigidAssetModel,
    RigidGaussianModel,
    SHGaussianModel,
)
from nurec.models.gaussians.interpolated_table_gaussians import InterpolatedTableDeformableGaussianModel
from nurec.models.gaussians.vehicle_light import VehicleLightRigidGaussianModel


_ModelT = TypeVar("_ModelT", bound=SHGaussianModel)

_SYSTEM = object()  # The per-step updates under test only forward ``system``; they never read it.
_EPOCH = 3


class _RecordingStepModule:
    """Stands in for a per-step-updated child module (tracks calibration, time embedding, encoding)."""

    def __init__(self, returned: dict[str, torch.Tensor] | None = None) -> None:
        self.calls: list[tuple[int, int, Any]] = []
        self._returned = returned if returned is not None else {}

    def update_step_train_batch_start(self, epoch: int, global_step: int, system: Any, **kwargs: Any) -> dict:
        self.calls.append((epoch, global_step, system))
        return dict(self._returned)


def _bare(model_class: type[_ModelT], *, fourier_freeze_until_step: int = 0) -> _ModelT:
    """A layer instance with only the state ``update_step_train_batch_start`` reads."""
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.fourier_features_dim = 1
    model.config = cast(Any, SimpleNamespace(fourier_freeze_until_step=fourier_freeze_until_step))
    model._global_step_for_freeze = 0
    model.time_embed = None
    if isinstance(model, RigidGaussianModel):
        model.global_step = 0
        model.tracks_calib = cast(Any, _RecordingStepModule({"tracks_calib/marker": torch.tensor(1.0)}))
    if isinstance(model, InterpolatedTableDeformableGaussianModel):
        model.config = cast(
            Any, SimpleNamespace(fourier_freeze_until_step=fourier_freeze_until_step, deformation_start_iteration=100)
        )
        model.use_deformation = False
    if isinstance(model, DeformableGaussianModel):
        model.config = cast(
            Any, SimpleNamespace(fourier_freeze_until_step=fourier_freeze_until_step, deformnet_start_iteration=100)
        )
        model.use_deform_network = False
        model.deform_network = cast(
            Any, SimpleNamespace(feature_volume=SimpleNamespace(encoding=_RecordingStepModule()))
        )
    if isinstance(model, DeformableRigidAssetModel):
        model.config = cast(
            Any,
            SimpleNamespace(
                fourier_freeze_until_step=fourier_freeze_until_step,
                restriction_policy="vis_frequency_drop",
                masking_end_step=3000,
            ),
        )
        model._first_update_done = True
        model._resumed_from_ckpt = False
        model.log_every_n_steps = 100
        model.optimizers = []
        model._masking_started_logged = True
        model._masking_ended_logged = False
        model._deform_started_logged = True
        model._deform_ended_logged = False
        model.use_deformation = True
        model.gaussians_hit_counter = torch.tensor([0, 1, 5, 7], dtype=torch.int32)
        model.gaussians_hit_mask = torch.zeros(4, dtype=torch.bool)
    return model


def _tracks_calib(model: RigidGaussianModel) -> _RecordingStepModule:
    return cast(_RecordingStepModule, model.tracks_calib)


# Every concrete layer class in the hierarchy rooted at SHGaussianModel.
_LAYER_CLASSES: list[type[SHGaussianModel]] = [
    SHGaussianModel,
    ConstructionBarriersGaussianModel,
    RigidGaussianModel,
    DeformableGaussianModel,
    DeformableRigidAssetModel,
    InterpolatedTableDeformableGaussianModel,
    VehicleLightRigidGaussianModel,
]


def test_layer_class_list_covers_every_subclass_of_the_sh_model() -> None:
    """The class list below must follow the hierarchy; a new subclass has to be added here deliberately."""

    def _descendants(cls: type) -> set[type]:
        direct = set(cls.__subclasses__())
        return direct.union(*(_descendants(sub) for sub in direct)) if direct else set()

    production = {
        cls
        for cls in _descendants(SHGaussianModel)
        if cls.__module__.startswith("nurec.models.gaussians.") and not cls.__module__.endswith("_test")
    }
    assert production | {SHGaussianModel} == set(_LAYER_CLASSES)


@pytest.mark.parametrize("model_class", _LAYER_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("global_step", [1, 2_999, 10_000, 29_999])
def test_freeze_counter_follows_the_global_step(model_class: type[SHGaussianModel], global_step: int) -> None:
    model = _bare(model_class, fourier_freeze_until_step=10_000)

    model.update_step_train_batch_start(_EPOCH, global_step, _SYSTEM)

    assert model._global_step_for_freeze == global_step


@pytest.mark.parametrize(
    "model_class",
    [
        RigidGaussianModel,
        DeformableGaussianModel,
        DeformableRigidAssetModel,
        InterpolatedTableDeformableGaussianModel,
        VehicleLightRigidGaussianModel,
    ],
    ids=lambda cls: cls.__name__,
)
def test_tracks_calib_and_time_embed_are_updated_exactly_once_per_step(model_class: type[RigidGaussianModel]) -> None:
    model = _bare(model_class)
    time_embed = _RecordingStepModule({"time_embed/marker": torch.tensor(2.0)})
    model.time_embed = cast(Any, time_embed)

    returned = model.update_step_train_batch_start(_EPOCH, 4_321, _SYSTEM)

    assert _tracks_calib(model).calls == [(_EPOCH, 4_321, _SYSTEM)]
    assert time_embed.calls == [(_EPOCH, 4_321, _SYSTEM)]
    # Both modules' additional parameters reach the caller.
    assert set(returned) == {"tracks_calib/marker", "time_embed/marker"}


def test_sh_model_updates_time_embed_once_and_returns_its_parameters() -> None:
    model = _bare(SHGaussianModel)
    time_embed = _RecordingStepModule({"time_embed/marker": torch.tensor(2.0)})
    model.time_embed = cast(Any, time_embed)

    returned = model.update_step_train_batch_start(_EPOCH, 7, _SYSTEM)

    assert time_embed.calls == [(_EPOCH, 7, _SYSTEM)]
    assert set(returned) == {"time_embed/marker"}


def test_rigid_model_still_records_its_global_step() -> None:
    model = _bare(RigidGaussianModel)

    model.update_step_train_batch_start(_EPOCH, 1_234, _SYSTEM)

    assert model.global_step == 1_234


@pytest.mark.parametrize(
    ("global_step", "expected_encoding_steps", "expected_use_deform_network"),
    [(99, [], False), (100, [0], True), (250, [150], True)],
)
def test_deformable_model_keeps_its_deformation_schedule(
    global_step: int, expected_encoding_steps: list[int], expected_use_deform_network: bool
) -> None:
    model = _bare(DeformableGaussianModel)
    encoding = cast(_RecordingStepModule, model.deform_network.feature_volume.encoding)

    model.update_step_train_batch_start(_EPOCH, global_step, _SYSTEM)

    assert [step for _, step, _ in encoding.calls] == expected_encoding_steps
    assert model.use_deform_network is expected_use_deform_network
    assert _tracks_calib(model).calls == [(_EPOCH, global_step, _SYSTEM)]


@pytest.mark.parametrize(("global_step", "expected"), [(99, False), (100, True)])
def test_interpolated_table_model_keeps_its_deformation_switch(global_step: int, expected: bool) -> None:
    model = _bare(InterpolatedTableDeformableGaussianModel, fourier_freeze_until_step=10_000)

    model.update_step_train_batch_start(_EPOCH, global_step, _SYSTEM)

    assert model.use_deformation is expected
    assert model._global_step_for_freeze == global_step
    assert _tracks_calib(model).calls == [(_EPOCH, global_step, _SYSTEM)]


def test_deformable_rigid_asset_model_keeps_its_hit_mask_update() -> None:
    model = _bare(DeformableRigidAssetModel)

    model.update_step_train_batch_start(_EPOCH, 10, _SYSTEM)

    # hit frequency = counter / step; masked where below 0.2 (counter < 2 at step 10).
    assert model.gaussians_hit_mask.tolist() == [True, True, False, False]
    assert model._current_step == 10
    assert _tracks_calib(model).calls == [(_EPOCH, 10, _SYSTEM)]


_N_GAUSSIANS = 4
_FOURIER_BANDS = 20
_FREEZE_UNTIL = 10_000


def _albedo_gradient_after_step(model: RigidGaussianModel, global_step: int) -> torch.Tensor:
    """Run the per-step update, then one backward pass through ``features_albedo``; return its gradient."""
    model.update_step_train_batch_start(_EPOCH, global_step, _SYSTEM)
    albedo = model.features_albedo
    albedo.grad = None
    weights = torch.arange(1, albedo.numel() + 1, dtype=albedo.dtype).reshape(albedo.shape)
    (albedo * weights).sum().backward()
    assert albedo.grad is not None
    return albedo.grad


@pytest.mark.parametrize("model_class", [RigidGaussianModel, DeformableRigidAssetModel], ids=lambda cls: cls.__name__)
def test_fourier_bands_are_masked_before_the_freeze_step_and_train_from_it(
    model_class: type[RigidGaussianModel],
) -> None:
    model = _bare(model_class, fourier_freeze_until_step=_FREEZE_UNTIL)
    model.fourier_features_dim = _FOURIER_BANDS
    # Assigning the parameter attaches the freeze hook (``SHGaussianModel.__setattr__``).
    model.features_albedo = nn.Parameter(torch.zeros(_N_GAUSSIANS, _FOURIER_BANDS, 3))
    unmasked = torch.arange(1, _N_GAUSSIANS * _FOURIER_BANDS * 3 + 1, dtype=torch.float32).reshape(
        _N_GAUSSIANS, _FOURIER_BANDS, 3
    )

    for global_step in (0, 1, _FREEZE_UNTIL - 1):
        grad = _albedo_gradient_after_step(model, global_step)
        torch.testing.assert_close(grad[:, 0, :], unmasked[:, 0, :])
        assert torch.count_nonzero(grad[:, 1:, :]) == 0, f"bands 1+ must be masked at step {global_step}"

    for global_step in (_FREEZE_UNTIL, _FREEZE_UNTIL + 1, 3 * _FREEZE_UNTIL - 1):
        grad = _albedo_gradient_after_step(model, global_step)
        torch.testing.assert_close(grad, unmasked, msg=f"all bands must train at step {global_step}")


def test_no_freeze_hook_without_a_freeze_step() -> None:
    model = _bare(RigidGaussianModel, fourier_freeze_until_step=0)
    model.fourier_features_dim = _FOURIER_BANDS
    model.features_albedo = nn.Parameter(torch.zeros(_N_GAUSSIANS, _FOURIER_BANDS, 3))

    grad = _albedo_gradient_after_step(model, 0)

    assert torch.count_nonzero(grad) == grad.numel()
