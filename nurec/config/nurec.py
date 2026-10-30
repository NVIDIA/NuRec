# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os

from functools import reduce
from math import gcd
from typing import Any, Literal

from pydantic import model_validator

from nurec.config.base_schema import BaseConfigSchema, Field, resolve_per_sensor, validate_per_sensor_coverage
from nurec.config.checkpoint import CheckpointConfig
from nurec.config.component_only import ComponentOnlyConfig, validate_component_only
from nurec.config.datamodule import SODatamoduleConfig
from nurec.config.dataset import (
    NCoreDatasetConfig,
)
from nurec.config.fixed_camera_render import FixedCameraRenderConfig
from nurec.config.harmonizer import HarmonizerConfig
from nurec.config.logger import LoggerConfigType
from nurec.config.model import ActiveTileMaskStrategy, GSplatRendererConfig, ModelConfig
from nurec.config.prober import ProberConfig
from nurec.config.profiling import ProfilingConfig
from nurec.config.scopedtimer import ScopedTimerConfig
from nurec.config.sensor import SensorConfig
from nurec.config.systems import (
    GaussiansSystemConfig,
)
from nurec.config.trainer import TrainerConfig
from nurec.config.version import Version, get_version
from nurec.config.viewer import ViewerConfig
from nurec.instant_nurec.config.dataset import NCoreInstantNuRecDatasetConfig
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.libs.losses.orchestration.config import LossConfig


# Cycle boundary: instant_nurec schemas may depend on the leaf config targets only, never
# the full //nurec/config:pylib that owns this NuRecConfig schema.


SENTINEL = "<sentinel>"

current_version = get_version()

cmd_logger = logging.getLogger(__name__)


class NuRecConfig(BaseConfigSchema):
    seed: int
    mode: Literal["train", "val", "test", "trainval"]

    resume: str | None
    resume_weights_only: bool
    verbose: bool
    force_validate: bool = Field(default=False, description="Force validation of the config.")

    out_dir: str
    logger: LoggerConfigType = Field(discriminator="name")

    checkpoint: CheckpointConfig
    harmonizer: HarmonizerConfig

    viewer: ViewerConfig = Field(default_factory=ViewerConfig, description="Web-based viewer configuration.")
    prober: ProberConfig = Field(
        default_factory=ProberConfig, description="Tensor prober configuration for debugging and testing."
    )
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig, description="Profiling configuration.")
    fixed_camera_render: FixedCameraRenderConfig = Field(
        default_factory=FixedCameraRenderConfig,
        description="Periodic fixed-camera render callback configuration.",
    )
    scopedtimer: ScopedTimerConfig = Field(default_factory=ScopedTimerConfig, description="Scoped timer configuration.")

    trainer: TrainerConfig
    dataset: NCoreDatasetConfig = Field(discriminator="name")
    datamodule: SODatamoduleConfig
    model: ModelConfig
    in_process_instant_nurec: InstantNuRecConfig | None = None
    component_only: ComponentOnlyConfig = Field(
        default_factory=ComponentOnlyConfig,
        description=(
            "Component-only optimization: train a subset of the composite's components with "
            "the loss masked to their responsible pixels. Inert unless component_only.enabled."
        ),
    )
    system: GaussiansSystemConfig = Field(discriminator="name")
    sensor: SensorConfig
    loss: LossConfig | None

    version: Version | None = Field(
        default=current_version,
        description="Not to be set by the user. Used to detect NuRec version mismatch when loading old configs. Not available in sandboxed test executions",
    )

    save_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where images are saved during validation phase. If left unchanged, defaults to `out_dir/save`"
        ),
    )
    ckpt_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where model checkpoints are saved during training. If left "
            "unchanged, defaults to `out_dir/checkpoints`"
        ),
    )
    config_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where the config (with all the auto-generated and default "
            "fields) will be stored. If left unchanged, defaults to `out_dir/config`"
        ),
    )
    run_id: str = Field(
        default=SENTINEL,
        description=(
            "A unique identifier of the training run. If left unchanged, will be auto-generated. "
            "If resuming training from a checkpoint, the previous run_id will be restored."
        ),
    )
    train_config_name: str = Field(
        default=SENTINEL,
        description=(
            "The name of the training config file. If empty, the config name will be the one provided in the command line. "
        ),
    )
    matmul_precision: Literal["highest", "high", "medium"] = Field(
        default="highest",
        description=(
            "PyTorch matmul precision setting. Controls the internal precision of float32 matrix multiplications. "
            "'highest' uses float32 datatype (default), 'high' uses TensorFloat32 or bfloat16-based algorithm, "
            "'medium' uses bfloat16 datatype for better performance with some precision trade-off."
        ),
    )

    def _setup_resume_path(self) -> None:
        if self.resume is None:
            return

        # When user use resume=auto, we will try to resume from the last checkpoint with the same run_id.
        # If the checkpoint does not exist, we will train from scratch.
        # This is useful for long-running training jobs on clusters where the job is constantly interrupted & restarted.
        if self.resume == "auto":
            is_auto_resume = True
            self.resume = "last"
        else:
            is_auto_resume = False

        # Add .ckpt if config.resume doesn't include it. Useful to run resume=last or resume=best
        if not self.resume.endswith(".ckpt"):
            self.resume += ".ckpt"
        if not os.path.exists(self.resume):
            if self.ckpt_dir == SENTINEL:
                if is_auto_resume:
                    # Automatically find out the checkpoint directory
                    self.ckpt_dir = os.path.join(self.out_dir, self.logger.run_id, "checkpoints")
                    cmd_logger.info(f"Automatically resuming from {self.ckpt_dir}")
                else:
                    raise FileNotFoundError("config.ckpt_dir is not provided, can't guess the resume path.")

            self.resume = os.path.join(self.ckpt_dir, self.resume)
            if not os.path.exists(self.resume):
                if is_auto_resume:
                    # If checkpoint not found, training from scratch!
                    self.resume = None
                    cmd_logger.info("Checkpoint not found, training from scratch.")
                # Do not check for checkpoint existence here.
                # It's only used by the trainer, and can be safely ignored when it's not being used.
                # This check will eventually be done on training.

    def _setup_run_id(self) -> None:
        """propagate run_id set in self.logger to other fields"""
        run_id = self.logger.run_id
        self.run_id = run_id
        self.save_dir = os.path.join(self.out_dir, run_id, "save")
        self.ckpt_dir = os.path.join(self.out_dir, run_id, "checkpoints")
        self.config_dir = os.path.join(self.out_dir, run_id, "config")

    def _setup_update_n_epochs(self) -> None:
        # Check if any of the samplers set the update_n_epochs parameter otherwise use the global value
        if isinstance(self.dataset, NCoreDatasetConfig) and self.dataset.samplers:
            update_n_epochs = []
            for batch_sampler, sampler_config in self.dataset.samplers.items():
                if not hasattr(sampler_config, "update_n_epochs"):
                    raise ValueError(f"Required key 'update_n_epochs' is missing from {batch_sampler} config.")
                elif type(update_interval := sampler_config.get("update_n_epochs")) is not int or (update_interval < 0):
                    raise TypeError(f"Key 'update_n_epochs' in {batch_sampler} must be a non-negative integer.")

                if update_interval > 0:
                    update_n_epochs.append(update_interval)

            if update_n_epochs:
                gcd_all = reduce(gcd, update_n_epochs)
                if gcd_all != min(update_n_epochs):
                    raise ValueError(
                        "Update intervals update_n_epochs of individual samplers must be multiples of each other."
                    )
                self.datamodule.update_n_epochs = gcd_all

    def _validate_in_process_instant_nurec_datasource(self) -> None:
        if self.in_process_instant_nurec is None:
            return

        predict_dataset = self.in_process_instant_nurec.dataset.predict
        if not isinstance(predict_dataset, NCoreInstantNuRecDatasetConfig):
            raise ValueError("in_process_instant_nurec.dataset.predict must be an NCoreInstantNuRecDatasetConfig")

        if predict_dataset.ncore_json_paths != [self.dataset.path]:
            raise ValueError("in_process_instant_nurec.dataset.predict must use only the outer NuRec dataset.path")

        # The shared standalone instant_nurec preset carries a legacy base path for list
        # files. An absolute or universal outer path cannot be redirected by it.
        if "://" not in self.dataset.path and not os.path.isabs(self.dataset.path):
            raise ValueError(
                "in_process_instant_nurec requires the outer NuRec dataset.path to be absolute or universal"
            )

    def model_post_init(self, __context) -> None:
        self._validate_in_process_instant_nurec_datasource()
        validate_component_only(self.component_only, self.model.layers)
        self._setup_update_n_epochs()
        self._setup_resume_path()
        self._setup_run_id()  # modifies self.ckpt_dir so needs to be called AFTER _setup_resume_path

        # Handle train_config_name setup
        if self.train_config_name == SENTINEL and __context is not None and "config_name" in __context:
            self.train_config_name = __context.get("config_name")

    @model_validator(mode="after")
    def _validate_lidar_ray_mode_consistency(self) -> NuRecConfig:
        """Fail loudly on an inconsistent model-free LiDAR configuration.

        The model-free path spans two configs that MUST agree, PER SENSOR:
          - dataset.lidar_ray_mode ("structured" grid vs "unstructured" (N, 1) list)
          - model.renderer.lidar_unstructured (routes to render_unstructured)

        Every lidar_unstructured_* / lidar_ray_mode / use_real_lidar_rays field is
        PerSensor[...] (a scalar broadcast, or a {lidar_id: value} dict listing every
        active lidar_id). We resolve each field for every active lidar_id and
        enforce the pairing/consistency rules per sensor.
        """
        # Active lidar ids (dataset config lists all sensors this run uses).
        lidar_ids = list(self.dataset.lidar_ids)

        renderer = self.model.renderer
        gsplat_renderer = renderer if isinstance(renderer, GSplatRendererConfig) else None

        # PerSensor coverage: every per-sensor dict must list all active ids.
        # Validate the full set up-front for clear errors.
        _per_sensor_fields: list[tuple[str, Any]] = [
            ("dataset.lidar_ray_mode", self.dataset.lidar_ray_mode),
            ("dataset.use_real_lidar_rays", self.dataset.use_real_lidar_rays),
            (
                "dataset.lidar_unstructured_raydrop_use_stored_no_returns",
                self.dataset.lidar_unstructured_raydrop_use_stored_no_returns,
            ),
            (
                "dataset.lidar_unstructured_raydrop_virtual_rays",
                self.dataset.lidar_unstructured_raydrop_virtual_rays,
            ),
            (
                "dataset.lidar_unstructured_raydrop_rays_per_tile",
                self.dataset.lidar_unstructured_raydrop_rays_per_tile,
            ),
            (
                "dataset.lidar_unstructured_raydrop_lattice_time_modes",
                self.dataset.lidar_unstructured_raydrop_lattice_time_modes,
            ),
            (
                "dataset.lidar_unstructured_raydrop_lattice_max_frames",
                self.dataset.lidar_unstructured_raydrop_lattice_max_frames,
            ),
            (
                "dataset.lidar_unstructured_raydrop_lattice_n_bins_azimuth",
                self.dataset.lidar_unstructured_raydrop_lattice_n_bins_azimuth,
            ),
            (
                "dataset.lidar_unstructured_raydrop_lattice_n_bins_elevation",
                self.dataset.lidar_unstructured_raydrop_lattice_n_bins_elevation,
            ),
        ]
        # NOTE: two groups of optional-override fields are intentionally NOT
        # coverage-checked, because for both a partial dict is meaningful:
        #   - the FOV fields (lidar_unstructured_fov_vert_deg/_horiz_deg): a dict may
        #     list only the sensors needing an explicit FOV (e.g. a model-less MEMS
        #     unit); omitted sensors fall back to their model FOV (None).
        #   - the occupancy-grid resolutions
        #     (lidar_unstructured_raydrop_occupancy_n_bins_azimuth/_elevation): both
        #     are Optional[int] where None means "derive from the sensor grid", so
        #     omitting a sensor selects that default rather than leaving a hole.
        # Requiring full coverage on either would force callers to spell out values
        # they explicitly want defaulted.
        if gsplat_renderer is not None:
            _per_sensor_fields += [
                ("model.renderer.lidar_unstructured", gsplat_renderer.lidar_unstructured),
                (
                    "model.renderer.culling.cull_dead_pairs_lidar_unstructured",
                    gsplat_renderer.culling.cull_dead_pairs_lidar_unstructured,
                ),
                (
                    "model.renderer.culling.box_method_lidar_unstructured",
                    gsplat_renderer.culling.box_method_lidar_unstructured,
                ),
                (
                    "model.renderer.culling.pbf_near_range_lidar_unstructured",
                    gsplat_renderer.culling.pbf_near_range_lidar_unstructured,
                ),
                (
                    "model.renderer.culling.rs_single_cover_lidar_unstructured",
                    gsplat_renderer.culling.rs_single_cover_lidar_unstructured,
                ),
            ]
        for _name, _val in _per_sensor_fields:
            validate_per_sensor_coverage(_val, lidar_ids, field_name=_name)

        raydrop_loss = None if self.loss is None else self.loss.get("raydrop")
        raydrop_active = raydrop_loss is not None and raydrop_loss.lambda_ > 0

        any_unstructured = False
        for lidar_id in lidar_ids:
            ray_mode = resolve_per_sensor(self.dataset.lidar_ray_mode, lidar_id)

            if ray_mode == "structured":
                # A structured sensor must NOT be routed through the unstructured renderer.
                if gsplat_renderer is not None and bool(
                    resolve_per_sensor(gsplat_renderer.lidar_unstructured, lidar_id)
                ):
                    raise ValueError(
                        f"model.renderer.lidar_unstructured=True for sensor '{lidar_id}' requires "
                        "dataset.lidar_ray_mode='unstructured' (the model-free renderer needs the "
                        "(N, 1) measured/at-element ray list, not the structured sensor-model grid)."
                    )
                # NOTE: lidar_unstructured_raydrop_use_stored_no_returns is deliberately
                # NOT validated here, and is a silent no-op on a structured sensor. The
                # range-view grid already represents non-returning beams as cells that no
                # return scatters into, and keeps its prefilled DROPPED at those cells, so
                # there is nothing for the flag to classify (see get_lidar_data_batch,
                # which gates it on `unstructured`).
                #
                # This is a foot-gun on a mixed rig: the field is per-sensor, so it can be
                # set on a structured sensor and will be ignored with no signal. Warning
                # here is the right fix, but this validator currently only ever raises, so
                # introducing a warn path is a separate change. Left as a follow-up.
                continue

            # ---- unstructured sensor ----
            any_unstructured = True
            if gsplat_renderer is None:
                raise ValueError(
                    f"dataset.lidar_ray_mode='unstructured' (sensor '{lidar_id}') requires a GSplat "
                    f"renderer (model.renderer), but got {type(self.model.renderer).__name__}. The "
                    "model-free unstructured LiDAR path is only implemented for GSplatRendererConfig."
                )
            if not bool(resolve_per_sensor(gsplat_renderer.lidar_unstructured, lidar_id)):
                raise ValueError(
                    f"dataset.lidar_ray_mode='unstructured' (sensor '{lidar_id}') requires "
                    "model.renderer.lidar_unstructured=True for that sensor (route the LiDAR branch "
                    "through render_unstructured on the model-free (N, 1) ray list)."
                )

            measured = bool(resolve_per_sensor(self.dataset.use_real_lidar_rays, lidar_id))
            mode = resolve_per_sensor(self.dataset.lidar_unstructured_raydrop_virtual_rays, lidar_id)
            stored = bool(resolve_per_sensor(self.dataset.lidar_unstructured_raydrop_use_stored_no_returns, lidar_id))
            # Negatives may come from stored no-return emissions, from synthesized
            # virtual rays, or both -- the two sources are independent. Only the case
            # where NEITHER is enabled leaves the raydrop head with no negatives at all.
            if raydrop_active and mode == "off" and not stored:
                raise ValueError(
                    f"loss.raydrop.lambda_>0 with dataset.lidar_ray_mode='unstructured' (sensor "
                    f"'{lidar_id}') requires a source of raydrop negatives: set "
                    "dataset.lidar_unstructured_raydrop_use_stored_no_returns=True (only if the "
                    "dataset stores no-return emissions, NCore V4+), or set "
                    "dataset.lidar_unstructured_raydrop_virtual_rays to 'model_elements' or "
                    "'aggregate_lattice', or set loss.raydrop.lambda_=0.0."
                )
            # aggregate_lattice yields sampled sensor-frame directions with no grid
            # element indices, so it needs measured world rays (use_real_lidar_rays=True).
            if mode == "aggregate_lattice" and not measured:
                raise ValueError(
                    f"dataset.lidar_unstructured_raydrop_virtual_rays='aggregate_lattice' (sensor "
                    f"'{lidar_id}') requires dataset.use_real_lidar_rays=True (measured world rays): "
                    "the aggregate-lattice sampler produces sensor-frame directions without grid "
                    "element indices, which the modeled-at-elements ray path cannot generate from. "
                    "Use 'model_elements' for the modeled (use_real_lidar_rays=False) path."
                )
            # model_elements derives virtual rays from the sensor grid, so it is the
            # MODELED path (use_real_lidar_rays=False); pairing with measured rays is
            # contradictory.
            if mode == "model_elements" and measured:
                raise ValueError(
                    f"dataset.lidar_unstructured_raydrop_virtual_rays='model_elements' (sensor "
                    f"'{lidar_id}') requires dataset.use_real_lidar_rays=False (the modeled-at-"
                    "elements path): it derives virtual rays from the sensor grid, which contradicts "
                    "measured (model-free) rays. Use 'aggregate_lattice' for the measured "
                    "(use_real_lidar_rays=True) path."
                )
            # Firing-lattice grid resolution is all-or-nothing per sensor.
            _n_az = resolve_per_sensor(self.dataset.lidar_unstructured_raydrop_lattice_n_bins_azimuth, lidar_id)
            _n_el = resolve_per_sensor(self.dataset.lidar_unstructured_raydrop_lattice_n_bins_elevation, lidar_id)
            if (_n_az is None) != (_n_el is None):
                raise ValueError(
                    f"dataset.lidar_unstructured_raydrop_lattice_n_bins_azimuth and "
                    f"lidar_unstructured_raydrop_lattice_n_bins_elevation (sensor '{lidar_id}') must be "
                    f"set together (or both None to use the built-in defaults); got azimuth={_n_az}, "
                    f"elevation={_n_el}."
                )
            _occ_az = resolve_per_sensor(self.dataset.lidar_unstructured_raydrop_occupancy_n_bins_azimuth, lidar_id)
            _occ_el = resolve_per_sensor(self.dataset.lidar_unstructured_raydrop_occupancy_n_bins_elevation, lidar_id)
            if (_occ_az is None) != (_occ_el is None):
                raise ValueError(
                    f"dataset.lidar_unstructured_raydrop_occupancy_n_bins_azimuth and "
                    f"lidar_unstructured_raydrop_occupancy_n_bins_elevation (sensor '{lidar_id}') must be "
                    f"set together (or both None to use the sensor-grid / auto default); got azimuth={_occ_az}, "
                    f"elevation={_occ_el}."
                )
            # Positive-integer sanity (ge=1 can't be enforced on a PerSensor union field).
            for _fname, _fval in (
                ("lidar_unstructured_raydrop_rays_per_tile", self.dataset.lidar_unstructured_raydrop_rays_per_tile),
                (
                    "lidar_unstructured_raydrop_lattice_time_modes",
                    self.dataset.lidar_unstructured_raydrop_lattice_time_modes,
                ),
                (
                    "lidar_unstructured_raydrop_lattice_max_frames",
                    self.dataset.lidar_unstructured_raydrop_lattice_max_frames,
                ),
            ):
                _v = resolve_per_sensor(_fval, lidar_id)
                if int(_v) < 1:
                    raise ValueError(f"dataset.{_fname} (sensor '{lidar_id}') must be >= 1; got {_v}.")
            for _n_bins in (_n_az, _n_el):
                if _n_bins is not None and int(_n_bins) < 1:
                    raise ValueError(
                        f"dataset.lidar_unstructured_raydrop_lattice n_bins (sensor '{lidar_id}') "
                        f"must be >= 1; got {_n_bins}."
                    )
            # Same bound for the occupancy grid: the all-or-nothing check above only
            # pairs them, so 0 / negative would reach the raydrop tiling and fail there.
            for _occ_bins in (_occ_az, _occ_el):
                if _occ_bins is not None and int(_occ_bins) < 1:
                    raise ValueError(
                        f"dataset.lidar_unstructured_raydrop_occupancy n_bins (sensor '{lidar_id}') "
                        f"must be >= 1; got {_occ_bins}."
                    )
            # FOV overrides: both bounds, lo < hi.
            for _fname, _fov in (
                ("lidar_unstructured_fov_vert_deg", self.dataset.lidar_unstructured_fov_vert_deg),
                ("lidar_unstructured_fov_horiz_deg", self.dataset.lidar_unstructured_fov_horiz_deg),
            ):
                _fov_value = resolve_per_sensor(_fov, lidar_id, default=None)
                if _fov_value is not None:
                    if len(_fov_value) != 2:
                        raise ValueError(
                            f"dataset.{_fname} (sensor '{lidar_id}') must be [lo, hi] (2 values); got {_fov_value}."
                        )
                    if float(_fov_value[0]) >= float(_fov_value[1]):
                        raise ValueError(
                            f"dataset.{_fname} (sensor '{lidar_id}') must be [lo, hi] with lo < hi (degrees); got {_fov_value}."
                        )

        # floating_point_filter with neighbor_source="rangeview" applies a range-DISCONTINUITY
        # filter defined on the 2D (n_rows, n_columns) range-view image; it has no meaning for
        # the model-free (N, 1) return list. Forbid ONLY the rangeview source when ANY sensor is
        # unstructured; the sensor-agnostic "angle" source works from ray directions alone and is
        # the intended way to run the filter on unstructured lidar, so it is allowed.
        _fpf = self.system.test.lidar.floating_point_filter
        if any_unstructured and _fpf.enabled and _fpf.neighbor_source == "rangeview":
            raise ValueError(
                "test.lidar.floating_point_filter with neighbor_source='rangeview' is not supported when "
                "any lidar sensor uses dataset.lidar_ray_mode='unstructured': the range-view discontinuity "
                "filter is defined on the 2D range-view grid, which does not exist for the (N, 1) return "
                "list. Use neighbor_source='angle' for unstructured runs."
            )

        # The lidar half of active tile masking needs a STRUCTURED sensor model.
        # VALID_CAMERA_LIDAR_OVERLAP derives its lidar mask through
        # GSplatRenderer.build_camera_to_lidar_tile_mapping ->
        # compute_lidar_element_to_tile, which requires a
        # RowOffsetStructuredSpinningLidarModelParameters and reshapes to
        # (n_rows, n_columns). The model-free path has neither, and
        # render_unstructured takes no mask, so the combination would crash (or at
        # best silently drop the mask). CAMERA_INVALID_ONLY is fine: its lidar half
        # returns early, so it masks camera tiles only and leaves lidar rendering
        # and supervision untouched.
        if (
            any_unstructured
            and gsplat_renderer is not None
            and gsplat_renderer.active_tile_mask.enabled
            and gsplat_renderer.active_tile_mask.strategy is ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP
        ):
            raise ValueError(
                "model.renderer.active_tile_mask.strategy='valid_camera_lidar_overlap' is not supported "
                "when any lidar sensor uses dataset.lidar_ray_mode='unstructured': deriving the lidar "
                "tile mask needs a structured (n_rows, n_columns) spinning-lidar model to build the "
                "camera->lidar tile mapping, which the model-free (N, 1) path does not have. Use "
                "strategy='camera_invalid_only' (camera tiles only; lidar rendering and supervision "
                "untouched, so it is safe here) or set active_tile_mask.enabled=False."
            )

        # Unstructured tiling is collated one frame at a time
        # (UnstructuredLidarTilingData.collate_fn asserts len(seq) == 1), because each
        # frame's tiling has its own bin counts and offsets and they cannot be stacked
        # into a batch. Fail at config time rather than after the run has started and
        # the first DataLoader batch reaches collation.
        if any_unstructured and self.datamodule.train_batch_size != 1:
            raise ValueError(
                f"datamodule.train_batch_size={self.datamodule.train_batch_size} is not supported when "
                "any lidar sensor uses dataset.lidar_ray_mode='unstructured': the per-frame unstructured "
                "tiling is variable-size (its own azimuth/elevation bin counts and tile offsets per "
                "frame) and cannot be collated across frames, so only train_batch_size=1 is valid."
            )
        return self
