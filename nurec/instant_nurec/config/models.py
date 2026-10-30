# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from nurec.config.base_schema import BaseConfigSchema, Field
from nurec.config.harmonizer import HarmonizerConfig
from nurec.config.model import RendererConfigType


class BaseModelConfig(BaseConfigSchema):
    pass


class PrimitiveExportPreprocessConfig(BaseConfigSchema):
    """
    Config for per-chunk primitive preprocessing before export (and before merge).
    Used by preprocess_for_export(); not part of merge logic.
    """

    enabled: bool = Field(
        default=True,
        description="Whether to run per-chunk preprocess (density/sky/road filtering) before export or merge.",
    )
    density_prune_threshold: float = Field(
        default=0.01, description="Density threshold for pruning Gaussians in each chunk."
    )


class CelsiusExportPreprocessConfig(PrimitiveExportPreprocessConfig):
    """Celsius-specific export preprocess options (e.g. sky handling)."""

    keep_sky_gaussians: bool = Field(
        default=False,
        description="Whether to keep sky Gaussians when filtering. If False, sky Gaussians are removed per chunk.",
    )
    project_to_z_offset: bool = Field(
        default=False,
        description="If True, project each Gaussian onto the road plane (z_offset in rig space) along the ray "
        "that spawned it (from context). Requires context_rig to be passed to preprocess_for_export.",
    )
    z_offset: float = Field(
        default=0.0,
        description="Z height of the road plane in rig space; used when project_to_z_offset is True.",
    )
    projection: Literal["ray"] = Field(
        default="ray",
        description="Projection method: 'ray' projects along the spawning ray.",
    )


class KelvinExportPreprocessConfig(PrimitiveExportPreprocessConfig):
    """Kelvin-specific export preprocess options."""

    infill_road_color: bool = Field(
        default=True,
        description=(
            "If True, replace the non-sky region of the per-chunk sky cubemap with the chunk's "
            "median road RGB (blended via the soft sky mask). Avoids garbage MLP outputs in "
            "directions that never observed sky during training."
        ),
    )


class GaussiansActivationConfig(BaseConfigSchema):
    """
    Configuration for activation functions used in neural reconstruction models.
    """

    # Opacity activation parameters
    opacity_shift: float = Field(default=-2.0, description="Shift parameter for opacity sigmoid activation")

    # Scale activation parameters
    scale_type: Literal["world", "pixel"] = Field(default="world", description="Type of scale activation")
    scale_shift_log_ratio: float = Field(default=-1.0, description="Shift parameter for scale activation")
    scale_max: float = Field(default=0.3, description="Maximum scale value")
    scale_min: float = Field(
        default=0.0,
        description="Minimum scale value (clamp applied after exp). Use 0.01 when using 3DGUT renderer to avoid NaN gradients.",
    )

    # Distance activation parameters
    distance_type: Literal["sigmoid", "none"] = Field(default="sigmoid", description="Type of distance activation")
    distance_min: float = Field(default=0.1, description="Minimum distance value")
    distance_max: float = Field(default=500.0, description="Maximum distance value")
    distance_shift: float = Field(default=-1.65, description="Shift parameter for distance sigmoid")

    # XYZ activation parameters
    xyz_type: Literal["exp", "none"] = Field(default="exp", description="Type of XYZ activation")
    z_offset: float = Field(default=1.0, description="Offset for XYZ activation")

    # Sky mask activation parameters
    sky_mask_clamp_min: float = Field(default=-10.0, description="Minimum clamp value for sky mask")
    sky_mask_clamp_max: float = Field(default=10.0, description="Maximum clamp value for sky mask")

    # Forward flow activation parameters
    forward_flow_scale: float = Field(default=30.0, description="Scale factor for forward flow")

    # Falloff sigma activation parameters
    falloff_sigma_min: float = Field(
        default=0.25, description="Minimum falloff sigma value. If set to -1, it will be inferred from frame gap."
    )
    falloff_sigma_max: float = Field(default=2.0, description="Maximum falloff sigma value")
    falloff_sigma_clamp_min: float = Field(default=-5.0, description="Minimum clamp value for falloff sigma")
    falloff_sigma_clamp_max: float = Field(default=5.0, description="Maximum clamp value for falloff sigma")


class _CelsiusModelEncoderConfig(BaseConfigSchema):
    depth: int
    embed_dim: int
    n_heads: int
    mlp_ratio: float
    layer_scale_init_values: Optional[float] = Field(
        default=None, description="The initial values for the layer scale. If None, no layer scale is used."
    )
    qk_norm: bool = Field(
        default=False, description="Whether to use layer normalization after the query and key projections"
    )
    block_pattern: str = Field(
        default="T", description="Pattern of blocks in the encoder. 'T' for Transformer, 'M' for Mamba2Block."
    )

    def model_post_init(self, __context) -> None:
        # Expand block pattern to full length, for example if depth == 8, then
        # - block_pattern = "T" -> "TTTTTTTT"
        # - block_pattern = "TMMT" -> "TMMTTMMT"
        assert self.depth % len(self.block_pattern) == 0, "Encoder depth must be divisible by block pattern length"
        self.block_pattern = self.block_pattern * (self.depth // len(self.block_pattern))


class _CelsiusModelSkyModuleConfig(BaseConfigSchema):
    enabled: bool


class _CelsiusModelMotionModuleConfig(BaseConfigSchema):
    enabled: bool
    n_motion_tokens: int = Field(
        description="Set to 0 to disable motion basis (velocity will just be predicted from upscaled backbone feature)"
    )
    unpatch_dim: int = Field(default=128, description="Dimension of the progressive unpatching (very memory intensive)")
    motion_qkv_dim: int
    falloff: bool

    bidirectional_flow: bool = Field(
        default=False, description="Whether to use bidirectional flow (i.e. velocity will be 6-D)"
    )

    # [-1, 0] means disabled, [100000, 200000] means always enabled
    context_replace_start_global_step: int = Field(
        default=-1, description="Before this global step, the context will be replaced with the predicted velocity."
    )
    context_replace_end_global_step: int = Field(
        default=0, description="After this global step, the context will not be replaced with the predicted velocity."
    )

    def model_post_init(self, __context) -> None:
        assert self.context_replace_start_global_step < self.context_replace_end_global_step


class _CelsiusModelAffineModuleConfig(BaseConfigSchema):
    enabled: bool
    n_affine_tokens: int = Field(
        default=-1, description="Number of affine tokens. Set to -1 to allow dynamic number of tokens."
    )
    cross_attend: bool = Field(default=True, description="How the affine token interact with the other tokens")
    optimization_start_global_step: int = Field(
        default=0, description="Start optimizing the affine module from this global step."
    )

    def model_post_init(self, __context) -> None:
        if self.n_affine_tokens == -1:
            assert self.cross_attend, "Dynamic number of affine tokens requires cross_attend=True."


class _CelsiusModelVelocityFromLidarConfig(BaseConfigSchema):
    enabled: bool = Field(default=False, description="Whether to use velocity from lidar if available")
    gap_from_image_us: int = Field(default=0, description="Max gap from image in microseconds")
    near_mask_threshold_m: float = Field(default=3.0, description="Near mask threshold in meters")
    box_filter_voxel_size_m: float = Field(default=0.1, description="Box filter voxel size in meters")
    box_filter_max_count: int = Field(default=3, description="Box filter max count")
    gaussians_scale: float = Field(default=0.01, description="Scale for the gaussians")


class CelsiusModelConfig(BaseModelConfig):
    """
    Configuration for the Celsius model.
    """

    name: Literal["celsius"]

    renderer: RendererConfigType

    track_padding_m: List[float] = Field(
        default=[0.1, 0.1, 0.1],
        description=(
            "Padding in meters for the track bounding box in the x, y, z dimensions (left/right, front/back, up/down). "
            "This is used when masking dynamic objects is needed in static models."
        ),
        min_length=3,
        max_length=3,
    )

    init_weights_path: Optional[str] = Field(
        default=None,
        description="Path to the initial weights for the Celsius model. If not provided, the model will be initialized randomly.",
    )

    init_token_scale: float = Field(default=0.02, description="Scale for the sky/motion tokens for initialization")

    patch_shape: Tuple[int, int] = Field(default=(8, 8))

    encoder: _CelsiusModelEncoderConfig
    sky_module: _CelsiusModelSkyModuleConfig
    motion_module: _CelsiusModelMotionModuleConfig
    affine_module: _CelsiusModelAffineModuleConfig
    activations: GaussiansActivationConfig = Field(
        default_factory=GaussiansActivationConfig, description="Activation functions configuration."
    )
    velocity_from_lidar: _CelsiusModelVelocityFromLidarConfig = Field(
        default_factory=_CelsiusModelVelocityFromLidarConfig, description="Velocity from lidar configuration."
    )

    use_deferred_bp: bool = Field(default=True)
    activation_checkpointing: bool = Field(default=False)

    use_patch_embed_norm: bool = Field(
        default=True, description="Whether to use layer normalization after the patch embedding"
    )
    use_encoder_norm: bool = Field(default=True, description="Whether to use layer normalization after the encoder")
    centroid_prediction: Literal["distance", "xyz"] = Field(
        default="distance", description="The type of centroid prediction"
    )

    harmonizer: HarmonizerConfig | None = Field(
        default=None,
        description="Harmonizer model to be applied to each rendered frame as part of the model. None to disable.",
    )

    scene_rescale: float = Field(default=1.0, description="Rescale scenes for model input and output")

    legacy_mask_input: bool = Field(default=False, description="Whether to use the legacy mask input")

    export_preprocess: CelsiusExportPreprocessConfig = Field(
        default_factory=CelsiusExportPreprocessConfig,
        description="Per-chunk preprocess options for predict/export (filtering before merge or export).",
    )


class KelvinTokenGSEncoderConfig(BaseConfigSchema):
    name: Literal["token-gs-encoder"]
    depth: int
    n_heads: int
    embed_dim: int
    use_qk_norm: bool = Field(default=True)
    layer_scale_init_values: Optional[float] = Field(
        default=1e-4, description="The initial values for the layer scale. If None, no layer scale is used."
    )


class KelvinDAv3EncoderConfig(BaseConfigSchema):
    name: Literal["dav3-encoder"]
    depth: int
    n_heads: int
    embed_dim: int
    take_block_indices: List[int]
    aa_start_block_idx: int
    ffn_type: Literal["mlp", "swiglu"]
    checkpointing: Literal["all", "local", "none"] = Field(
        default="none", description="Whether to checkpoint the encoder"
    )


class KelvinVGGTEncoderConfig(BaseConfigSchema):
    """Config for the VGGT/VGGT geometry encoder behind the Kelvin latent seam.

    Defaults match the ``vggt_v2905`` checkpoint's constructor config: encoder
    embed_dim 1024 produces a concatenated latent width of 2048, with depth 18,
    taps (3, 8, 12, 17), attention_order [CAMERA, TIME], a COND camera-extrinsic
    encoder with decoupled rig/ego pose, and per-camera-name embeddings.
    Encoder freezing is controlled independently by the model configuration.
    """

    name: Literal["vggt-encoder"]

    # Concatenated frame+global attention-block channel width (2 * encoder embed_dim).
    # The ported VGGTEncoder is built with embed_dim // 2 (=1024 for v2905).
    embed_dim: int = Field(default=2048, description="Latent channel width = 2 * VGGTEncoder embed_dim.")
    n_heads: int = Field(default=16, description="Number of attention heads in the VGGT aggregator.")
    depth: int = Field(default=18, description="Number of alternating-attention layers (v2905 aggregator depth).")
    return_attn_block_idx: Tuple[int, ...] = Field(
        default=(3, 8, 12, 17), description="Aggregator layer indices whose (frame|attn) outputs form the taps."
    )
    # DEVIATION from brief sketch (img_size=None): the dinov2 patch-embed's
    # pos_embed interpolation needs the checkpoint's training img_size (518,518)
    # to reshape the 37x37 grid; None would break interpolate_pos_encoding for a
    # non-518 Kelvin input. See report decision notes.
    img_size: Tuple[int, int] = Field(
        default=(518, 518), description="Training image size; needed for dinov2 pos_embed interpolation."
    )
    img_features_stride: int = Field(default=14, description="Patch stride (patch-14 backbone).")
    patch_embed: str = Field(
        default="dinov2_vitl14_reg", description="Patch-embed backbone variant (v2905 uses dinov2 ViT-L/14 + reg)."
    )
    attention_order: Optional[List[str]] = Field(
        default_factory=lambda: ["camera", "time"],
        description="Per-layer second-attention pattern (tiled to depth). v2905 = [camera, time].",
    )
    use_camera_extrinsic_encoder: bool = Field(
        default=True, description="Use the COND CameraExtrinsicEncoder (v2905 conditions on pose)."
    )
    decouple_rig_ego_pose: bool = Field(
        default=True, description="Condition on decoupled rig + ego pose tokens (v2905=True)."
    )
    learn_embedding_per_camera_name: bool = Field(
        default=True, description="Add a learnable per-camera-name embedding (v2905 has 11 camera-name params)."
    )
    camera_names: Optional[List[str]] = Field(
        default_factory=lambda: [
            "camera_frontwide",
            "camera_crossleft",
            "camera_crossright",
            "camera_rearleft",
            "camera_rearright",
            "camera_front_wide_120fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
            "camera_rear_left_70fov",
            "camera_rear_right_70fov",
        ],
        description="Camera-name embedding table keys (verbatim from the v2905 checkpoint, see NOTES.md).",
    )
    camera_name_override: Optional[str] = Field(
        default=None,
        description=(
            "Camera-name key used by the legacy single-camera context path when adding per-camera-name "
            "embeddings. Set this (e.g. 'camera_front_wide_120fov') when camera_name_order is unset and "
            "learn_embedding_per_camera_name is True; encode() raises if neither routing mechanism is present."
        ),
    )
    camera_name_order: Optional[List[str]] = Field(
        default=None,
        description=(
            "Ordered active camera names for a multi-camera item. Input frames must be camera-major "
            "(all timestamps for camera 0, then camera 1, and so on), and FrameMeta.logical_sensor_id "
            "must match this list exactly. Leave unset for the legacy single-camera path, which uses "
            "camera_name_override."
        ),
    )
    cross_camera_ego_translation_tolerance_m: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "Maximum allowed metric translation disagreement between ego motion derived from different "
            "cameras at one timestamp. This guards synchronization/layout errors before the first-camera "
            "ego reference is broadcast to every camera."
        ),
    )
    cross_camera_ego_rotation_tolerance_deg: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Maximum allowed rotation disagreement, in degrees, between ego motion derived from different "
            "cameras at one timestamp. Tune together with the translation tolerance from a real-data audit."
        ),
    )
    pose_translation_scale_factor: float = Field(
        default=50.0,
        description=(
            "First-frame-relative pose translations are divided by this before conditioning, matching the "
            "vggt training default (TransformBatchConfig.pose_translation_scale_factor=50.0, "
            "pose_normalize_translation=True)."
        ),
    )
    use_vggt_depth: bool = Field(
        default=False,
        description=(
            "If True, construct + run the ported VGGT DPT depth head over the encoder taps and "
            "populate KelvinVGGTFeaturesLatent.encoder_depth with metric z-depth."
        ),
    )
    depth_scale: float = Field(
        default=40.0,
        description=(
            "Multiplicative conversion from the vggt depth head's raw (exp-activated) output units to "
            "meters. v2905 was supervised with GT depth divided by loss_config.fixed_gt_scale_value=40.0 "
            "(recorded in the checkpoint's own hyper_parameters; vggt's geometry_loss.py "
            "normalize_groundtruth_data does depth_gt /= fixed_gt_scale_value, and vggt/tasks/gaussians.py "
            "DenseGaussianPostProcessConfig.depth_scale documents the same 40.0 for v2905). Applied inside "
            "_run_depth so encoder_depth is true meters."
        ),
    )
    grad_checkpointing: bool = Field(
        default=False, description="Enable gradient checkpointing in the VGGT aggregator blocks."
    )
    grad_checkpointing_reentrant: bool = Field(
        default=False,
        description=(
            "Use reentrant gradient checkpointing for the VGGT aggregator blocks. Default (False, "
            "non-reentrant) is the working/expected setting under DDP. The 8/8 multi-GPU 'CUDA "
            "invalid access of peer GPU memory over nvlink' failures observed with non-reentrant "
            "checkpointing were traced to grad_checkpointing.py's wrapper capturing `training` "
            "statically at model-construction time, so checkpointing ran unconditionally (even "
            "during eval/no-grad sanity-validation passes); the wrapper now reads `module.training` "
            "and `torch.is_grad_enabled()` live on every call (see grad_checkpointing.py), matching "
            "the known-good aa_vit encoder pattern. Setting this True instead hits the classic DDP "
            "'Expected to mark a variable ready only once' error and is not a supported workaround."
        ),
    )
    ckpt_key: str = Field(
        default="vggt",
        description="Key in KelvinModelConfig.init_weights_paths whose loaded state dict holds the vggt weights.",
    )


class KelvinTokenGSDecoderConfig(BaseConfigSchema):
    name: Literal["token-gs-decoder"]
    depth: int
    use_qk_norm: bool = Field(default=True)
    layer_scale_init_values: Optional[float] = Field(
        default=1e-4, description="The initial values for the layer scale. If None, no layer scale is used."
    )
    num_gaussian_tokens: int = Field(default=1024, description="Number of Gaussian tokens")
    gaussian_token_init_std: float = Field(
        default=0.02, description="Standard deviation for the Gaussian token initialization"
    )
    use_decoder_norm: bool = Field(default=False, description="Whether to use layer normalization after the decoder")


class KelvinDPTDecoderConfig(BaseConfigSchema):
    name: Literal["dpt-decoder"]
    dpt_dim: int
    dpt_reassemble_hidden_dims: List[int]

    depth_offset: bool = Field(default=False, description="Whether to predict a depth offset (in world space)")
    uv_offset: bool = Field(default=False, description="Whether to predict a UV offset (in pixel space)")

    checkpointing: bool = Field(default=False, description="Whether to use checkpointing for the DPT decoder")
    dpt_chunk_size: int = Field(
        default=-1, description="Chunk size for the DPT decoder. Used for saving memory. -1 to disable."
    )

    fusion_for_gs_motion: bool = Field(
        default=False, description="Whether to use fusion for the GS head and the motion head"
    )

    # Motion-related:
    time_encoding_dim: int = Field(default=256, description="Dimension of the time sinusoidal encoding")
    motion_depth: int = Field(default=4, description="Depth of the motion head (V-DPM setup is equivalent to 8)")

    use_encoder_depth: bool = Field(
        default=False,
        description=(
            "Pretrained-depth ablation: if True, skip the DPT depth head entirely (it is "
            "`del`eted in __init__) and use the encoder's own KelvinVGGTFeaturesLatent.encoder_depth "
            "as `pred_depth` instead (with pred_depth_conf = ones_like). Requires a KelvinVGGTEncoder "
            "with use_vggt_depth=True upstream. Downstream unprojection (gs_distance = pred_depth / "
            "distance_to_depth_scale, etc.) is unchanged."
        ),
    )

    def model_post_init(self, __context) -> None:
        assert self.dpt_dim > 0, "DPT dimension must be positive"


class KelvinPointQueryCADecoderConfig(BaseConfigSchema):
    """Decoder that keeps DPT depth/context/motion heads but replaces the DPT gaussian head
    with a cross-attention head using depth-derived xyz positions (2D grid tokenization) as queries."""

    name: Literal["point-query-ca-decoder"]

    # Shared DPT fields (for depth, context, motion heads)
    dpt_dim: int
    dpt_reassemble_hidden_dims: List[int]
    checkpointing: bool = Field(default=False, description="Whether to use checkpointing for the DPT heads")
    dpt_chunk_size: int = Field(default=-1, description="Chunk size for the DPT heads. -1 to disable.")
    time_encoding_dim: int = Field(default=256, description="Dimension of the time sinusoidal encoding")
    motion_depth: int = Field(default=4, description="Depth of the motion head")

    # Cross-attention gaussian head
    ca_depth: int = Field(default=1, description="Number of cross-attention decoder blocks")
    use_qk_norm: bool = Field(default=True, description="Whether to use QK normalization in cross-attention")
    layer_scale_init_values: Optional[float] = Field(default=1e-4, description="Layer scale init values for CA blocks")
    xyz_downsample_stride: int = Field(default=4, description="Stride for downsampling depth map")
    grid_center_stride: int = Field(
        default=4,
        description="Grouping stride relative to the candidate grid. "
        "Each token covers a grid_center_stride x grid_center_stride block, so gs_per_token = grid_center_stride^2.",
    )
    use_gt_semantic_mask: bool = Field(
        default=False,
        description="If True, use ground-truth semantic labels instead of predicted semantics for the sky/ego opacity mask.",
    )
    voxel_size_m: Optional[float] = Field(
        default=None,
        description=(
            "Metric voxel edge for geometry-only static point deduplication. When set, one real candidate nearest "
            "each voxel center is retained, static representatives are Morton-grouped, dynamic grid tokens are "
            "preserved, and max_total_tokens is required. None keeps the historical uniform-grid tokenizer."
        ),
    )
    max_total_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Static-aware query-token budget across dynamic and deduplicated static tokens. Dynamic tokens have "
            "priority and are never truncated; if they alone exceed the budget, no static tokens are emitted and "
            "the dynamic overrun is logged. Otherwise, remaining slots are filled by Morton-ordered static tokens. "
            "Samples are padded only to the largest retained count in the local batch, not to this budget."
        ),
    )

    # Road-area extra point queries (oversampling on ROAD pixels).
    road_xyz_downsample_stride: Optional[int] = Field(
        default=None,
        description=(
            "When set, enables an extra batch of point queries restricted to ROAD pixels, sampled at this stride "
            "on the full-resolution image grid (e.g. 1 = per-pixel, 2 = every other pixel). Tokens still group "
            "grid_center_stride^2 sub-points so K (and therefore xyz_embed / gs_linear shapes) is unchanged. "
            "None disables the road branch."
        ),
    )
    road_token_threshold: float = Field(
        default=0.5,
        description=(
            "Fraction of the grid_center_stride x grid_center_stride sub-cells inside a candidate road token that "
            "must be classified as ROAD for the token to be kept."
        ),
    )
    road_max_tokens_per_view: Optional[int] = Field(
        default=None,
        description=(
            "Per-(batch element, view) cap on the number of extra road tokens. If more candidates exist they are "
            "randomly subsampled; if fewer, the slot is padded with zero-mask tokens so M_road is fixed per batch. "
            "Required when road_xyz_downsample_stride is set."
        ),
    )
    road_use_gt_semantic_mask: bool = Field(
        default=True,
        description=(
            "If True, use the GT ROAD ray-flag semantics to pick road pixels; if False, use the predicted "
            "argmax of semantic logits' ROAD channel."
        ),
    )

    def model_post_init(self, __context) -> None:
        assert self.dpt_dim > 0, "DPT dimension must be positive"
        assert self.xyz_downsample_stride >= 1, "xyz_downsample_stride must be a positive int"
        assert self.grid_center_stride >= 1, "grid_center_stride must be a positive int"
        if self.voxel_size_m is None:
            assert self.max_total_tokens is None, "max_total_tokens requires voxel_size_m"
        else:
            assert self.voxel_size_m > 0.0, "voxel_size_m must be positive"
            assert self.max_total_tokens is not None and self.max_total_tokens > 0, (
                "max_total_tokens must be a positive int when voxel_size_m is set"
            )
            assert self.grid_center_stride**2 == 64, (
                "voxel dedup requires grid_center_stride=8 so each query groups 64 points"
            )
            assert self.road_xyz_downsample_stride is None, (
                "the separate road oversampling branch must be disabled when voxel dedup is enabled"
            )
        if self.road_xyz_downsample_stride is not None:
            assert self.road_xyz_downsample_stride >= 1, "road_xyz_downsample_stride must be a positive int"
            assert self.road_max_tokens_per_view is not None and self.road_max_tokens_per_view > 0, (
                "road_max_tokens_per_view must be a positive int when road_xyz_downsample_stride is set"
            )
            assert 0.0 < self.road_token_threshold <= 1.0, "road_token_threshold must be in (0, 1]"


class KelvinPointQueryDeformAttnDecoderConfig(KelvinPointQueryCADecoderConfig):
    """Point-query decoder with explicit camera projection and local deformable attention.

    The inherited tokenizer/depth/context/motion fields intentionally match the
    established PointQuery control. Confidence is not an input to deduplication,
    projection, sampling, or attention.
    """

    name: Literal["point-query-deform-attn-decoder"]  # type: ignore[assignment]
    deform_feature_dim: int = Field(default=128, description="Feature width after projecting the VGGT tap.")
    deform_num_sampling_points: int = Field(
        default=4,
        description="Samples per projected query/view; the first is always the exact geometric pixel.",
    )
    deform_offset_radius_feature_px: float = Field(
        default=2.0,
        description="Maximum learned displacement around fixed unit-ring offsets, in feature pixels.",
    )
    deform_query_chunk_size: int = Field(
        default=1024,
        description="Point-query chunk size for checkpointed multi-view feature sampling.",
    )
    deform_depth_residual_scale_m: float = Field(
        default=1.0,
        description="Metric scale used by the query-vs-view depth residual attention cue.",
    )
    deform_occlusion_abs_tolerance_m: float = Field(
        default=0.5,
        description="Absolute depth margin before a projected query is classified as occluded.",
    )
    deform_occlusion_rel_tolerance: float = Field(
        default=0.05,
        description="Scene-depth-relative margin before a projected query is classified as occluded.",
    )
    deform_same_time_tolerance_us: int = Field(
        default=5000,
        description="Timestamp tolerance for cross-camera fusion of dynamic point queries.",
    )
    deform_view_fusion_mode: Literal["scalar", "depth-aware-interaction"] = Field(
        default="scalar",
        description=(
            "Across-view scorer. 'scalar' preserves the established cosine/depth/time scorer; "
            "'depth-aware-interaction' embeds sampled observations with metric depth residuals and "
            "scores a low-dimensional query-observation interaction plus time cues."
        ),
    )
    deform_view_interaction_dim: int = Field(
        default=32,
        description="Low-dimensional query/observation width for depth-aware interaction scoring.",
    )
    deform_predict_rgb_residual: bool = Field(
        default=False,
        description="Predict a zero-initialized bounded RGB-logit residual from the fused point feature.",
    )
    deform_rgb_residual_max_logit_delta: float = Field(
        default=2.0,
        description="Maximum absolute RGB-logit correction emitted by the fused point feature.",
    )
    deform_checkpointing: bool = Field(
        default=True,
        description="Activation-checkpoint geometric attention chunks to bound memory as context grows.",
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        assert self.deform_feature_dim > 0
        assert self.deform_num_sampling_points > 0
        assert self.deform_offset_radius_feature_px >= 0.0
        assert self.deform_query_chunk_size > 0
        assert self.deform_depth_residual_scale_m > 0.0
        assert self.deform_occlusion_abs_tolerance_m >= 0.0
        assert self.deform_occlusion_rel_tolerance >= 0.0
        assert self.deform_same_time_tolerance_us >= 0
        assert self.deform_view_interaction_dim > 0
        assert self.deform_rgb_residual_max_logit_delta > 0.0


class KelvinSkyCubemapDecoderConfig(BaseConfigSchema):
    name: Literal["cubemap-decoder"]
    cubemap_size: int
    embed_dim: int
    depth: int
    fusion_dim: int | None = Field(default=None, description="Dimension of the fusion layer")
    checkpointing: bool = Field(default=False, description="Whether to use checkpointing for the cubemap decoder")


class KelvinSkySolidColorConfig(BaseConfigSchema):
    name: Literal["solid-color"]
    color: tuple[float, float, float]
    cubemap_size: int


class KelvinPostProcessingConfig(BaseConfigSchema):
    enabled: bool = Field(
        default=True,
        description="If False, skip the per-camera affine RGB module and use an identity transform at render time.",
    )
    optimization_start_global_step: int = Field(
        default=0, description="Start optimizing the post processing module from this global step."
    )


class KelvinModelConfig(BaseModelConfig):
    """
    Configuration for the Kelvin model.
    """

    name: Literal["kelvin"]
    renderer: RendererConfigType

    track_padding_m: List[float] = Field(
        default=[1.0, 1.0, 1.0],
        description=(
            "Padding in meters for cuboid track bounding boxes when warping world points for motion supervision "
            "(x, y, z)."
        ),
        min_length=3,
        max_length=3,
    )

    scene_rescale: float = Field(default=1.0, description="Rescale scenes for model input and output")
    prepare_normal_supervision: bool = Field(default=True, description="Whether to prepare normal supervision")
    use_2dgs: bool = Field(default=False, description="Whether to use 2DGS rendering")
    freeze_encoder: bool = Field(
        default=False,
        description="If True, freeze encoder parameters (requires_grad=False); train decoder/sky/post-processing only.",
    )

    sky: KelvinSkyCubemapDecoderConfig | KelvinSkySolidColorConfig = Field(discriminator="name")

    patch_shape: Tuple[int, int] = Field(default=(8, 8))

    encoder: KelvinTokenGSEncoderConfig | KelvinDAv3EncoderConfig | KelvinVGGTEncoderConfig = Field(
        discriminator="name"
    )
    decoder: (
        KelvinTokenGSDecoderConfig
        | KelvinDPTDecoderConfig
        | KelvinPointQueryCADecoderConfig
        | KelvinPointQueryDeformAttnDecoderConfig
    ) = Field(discriminator="name")
    post_processing: KelvinPostProcessingConfig = Field(default_factory=KelvinPostProcessingConfig)
    init_weights_paths: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Paths to initial weights by component or full model. "
            'Single entry with key "tokengs" or "full" loads a full-model checkpoint. '
            "Multiple entries are passed to encoder/decoder/sky initialize_weights by name."
        ),
    )

    activations: GaussiansActivationConfig = Field(
        default_factory=GaussiansActivationConfig, description="Activation functions configuration."
    )

    # Augmentations
    voxel_size: float | None = Field(default=None, description="Voxel size for the voxelization")

    export_preprocess: KelvinExportPreprocessConfig = Field(
        default_factory=KelvinExportPreprocessConfig,
        description="Per-chunk preprocess options for predict/export (filtering before merge or export).",
    )
