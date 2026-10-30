# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import pickle
import tempfile

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, cast

import numpy as np
import torch
import torch.utils.checkpoint

from einops import einsum, rearrange, repeat
from torch import nn

from ncore.data import ConcreteCameraModelParametersUnion
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.instant_nurec.config.models import (
    KelvinDAv3EncoderConfig,
    KelvinDPTDecoderConfig,
    KelvinModelConfig,
    KelvinPointQueryCADecoderConfig,
    KelvinPointQueryDeformAttnDecoderConfig,
    KelvinTokenGSDecoderConfig,
    KelvinVGGTEncoderConfig,
)
from nurec.instant_nurec.models.activations import GaussianActivations, GaussianParams
from nurec.instant_nurec.models.blocks.aa_vit import AlternateAttentionVisionTransformer
from nurec.instant_nurec.models.blocks.attention import CrossAttentionBlock, KVProjector
from nurec.instant_nurec.models.blocks.dav3 import convert_dav3_state_dict_to_instant_nurec
from nurec.instant_nurec.models.blocks.dpt import DPTFullHead
from nurec.instant_nurec.models.blocks.embeds import ContinuousTimeEmbed
from nurec.instant_nurec.models.kelvin_backbone.base import (
    KelvinInstantNuRecSupervisionPack,
    KelvinLatent,
    KelvinMotionSupervision,
    KelvinMultiscaleFeaturesLatent,
    KelvinVGGTFeaturesLatent,
    _tokengs_init_weights,
)
from nurec.instant_nurec.models.kelvin_backbone.grid_tokens import (
    FullResInputs,
    GridTokens,
    VoxelDedupDiagnostics,
    build_grid_tokens,
    build_voxel_dedup_tokens,
    concat_grid_tokens,
)
from nurec.instant_nurec.models.kelvin_backbone.point_query_deform_attn import (
    PointQueryDeformAttn,
    apply_bounded_rgb_logit_residual,
    build_context_image_atlas,
    project_points_to_context_views,
)
from nurec.instant_nurec.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinSemanticClass,
    KelvinStaticLayer,
)
from nurec.instant_nurec.utils.motion import (
    TimeRemapping,
    associate_points_with_cuboid_tracks,
    warp_points_with_cuboid_tracks,
)
from nurec.models.gaussians.utils import sh_degree_to_specular_dim
from nurec.models.nn_extensions import TypedModuleList
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.log import BatchMediaLogger
from nurec.utils.misc import unpack_optional
from nurec.utils.profiling import ScopedTimer
from nurec.utils.visualize import make_image_grid, scalar2img


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class KelvinDecoderReturn:
    # Allowing all dynamic layers
    static_layer: KelvinStaticLayer | None
    dynamic_layers: list[KelvinDynamicLayer]
    supervision_pack: KelvinInstantNuRecSupervisionPack


class KelvinDecoderBase(nn.Module, ABC):
    @abstractmethod
    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        """
        Initialize the weights of the model from the loaded state dicts.
        """

    @abstractmethod
    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> list[KelvinDecoderReturn]:
        """
        Decode from the encoded latent into Gaussian parameters.
        Batches are used to pass in useful information about the raw context (e.g. timestamps, etc.)
        """

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please call decode() method directly.")

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs):
        # Do nothing by default
        pass

    def get_potential_unused_parameters(self) -> Iterator[nn.Parameter]:
        return iter([])


class KelvinTokenGSDecoder(KelvinDecoderBase):
    def __init__(self, config: KelvinTokenGSDecoderConfig, model_config: KelvinModelConfig):
        super().__init__()

        dim = model_config.encoder.embed_dim
        num_latent_heads = model_config.encoder.n_heads
        self.feature_norm = nn.LayerNorm(dim)
        self.kv_projector = KVProjector(
            dim=dim,
            n_heads=num_latent_heads,
            kv_bias=True,
            k_norm=config.use_qk_norm,
        )
        self.blocks = TypedModuleList(
            [
                CrossAttentionBlock(
                    dim,
                    num_latent_heads,
                    qkv_bias=True,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    qk_norm=config.use_qk_norm,
                    mlp_ratio=4.0,
                    dropout=0.0,
                    layer_scale_init_values=config.layer_scale_init_values,
                    kv_projector=None,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(dim) if config.use_decoder_norm else nn.Identity()
        self.gaussian_tokens = nn.Parameter(
            torch.randn(config.num_gaussian_tokens, dim) * config.gaussian_token_init_std
        )
        self.n_gaussians_per_token = model_config.patch_shape[0] * model_config.patch_shape[1]  # for now
        n_gaussian_params = 3 + 3 + 3 + 4 + 1
        self.token_to_gs_linear = nn.Linear(dim, n_gaussian_params * self.n_gaussians_per_token)
        self.gaussian_activations = GaussianActivations(model_config.activations)

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        self.apply(_tokengs_init_weights)
        torch.nn.init.trunc_normal_(self.token_to_gs_linear.weight, std=0.002)

    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> list[KelvinDecoderReturn]:
        """
        The returned GaussianParams will have shape (B, N, C)
        """
        batch_size = encoded_latent.batch_size
        deepest = encoded_latent.deepest
        encoder_feature = rearrange(deepest, "B V h w C -> B (V h w) C")
        encoder_feature = self.feature_norm(encoder_feature)

        k, v = self.kv_projector(k=encoder_feature, v=encoder_feature)
        gaussian_tokens = repeat(self.gaussian_tokens, "N C -> B N C", B=batch_size)
        for block in self.blocks:
            gaussian_tokens = block(gaussian_tokens, k, v)

        gaussian_tokens = self.norm(gaussian_tokens)

        gs_params_pre_activation = rearrange(
            self.token_to_gs_linear(gaussian_tokens),
            "B N (gs_per_token gs_params) -> B (N gs_per_token) gs_params",
            gs_per_token=self.n_gaussians_per_token,
        )
        gs_xyz, gs_rgb, gs_scale, gs_rotation, gs_opacity = gs_params_pre_activation.split([3, 3, 3, 4, 1], dim=-1)
        gs_params = GaussianParams(
            rgb=gs_rgb, scale=gs_scale, rotation=gs_rotation, opacity=gs_opacity, xyz=gs_xyz, activated=False
        )
        gs_params = self.gaussian_activations.forward(gs_params, scene_rescale=scene_rescale)

        return_values: list[KelvinDecoderReturn] = []
        for bidx in range(batch_size):
            gs_params_bidx = gs_params[bidx].flatten()
            return_values.append(
                KelvinDecoderReturn(
                    static_layer=KelvinStaticLayer(
                        rotations=gs_params_bidx.rotation,
                        scales=gs_params_bidx.scale,
                        rgb=gs_params_bidx.rgb,
                        positions=unpack_optional(gs_params_bidx.xyz),
                        densities=gs_params_bidx.opacity,
                    ),
                    dynamic_layers=[],
                    supervision_pack=KelvinInstantNuRecSupervisionPack(),
                )
            )
        return return_values


class KelvinDPTDecoder(KelvinDecoderBase):
    """
    DPT Head (compared to corresponding encoder this is ~5-10% of parameters & FLOPS)
    TODO: Compare SDT head from https://aigeeksgroup.github.io/AnyDepth/

    See schematic plot here:
    https://excalidraw.com/#json=e8F-fbXBIoMoihwIZxxwe,U5OH-X-P6QjCUh6i8xb5ag
    """

    class TimeModulatedMotionHead(nn.Module):
        """
        Takes in image tokens, source time, and target time, output motion offset from each frame to the target time.
        Currently the API jointly predicts forward & backward motion, as well as dynamic probability.
        - Pros: save compute by predicting two motion offsets at once.
        - Cons: somewhat introduces unncessary dependencies between the two motion offsets which should be independent.
            However, the original V-DPM style modulation does not work well if we de-couple them, where forward & backward
            motion offsets cancel each other out (i.e. the network does not seem to take src-time nor tgt-time into account).
            Two solutions might be possible: (1) Predict XYZ instead of offset, matching the exact V-DPM setting,
            (2) Adding attention mask (as in StreetForward / 4RC) may help, but haven't tried it yet.
        """

        def __init__(self, config: KelvinDPTDecoderConfig, model_config: KelvinModelConfig):
            super().__init__()
            ffn_type = (
                "mlp"
                if not isinstance(model_config.encoder, KelvinDAv3EncoderConfig)
                else model_config.encoder.ffn_type
            )
            self.embed_dim = model_config.encoder.embed_dim // 2
            self.vit = AlternateAttentionVisionTransformer(
                depth=config.motion_depth,
                embed_dim=self.embed_dim,  # Match encoder ViT
                n_heads=model_config.encoder.n_heads,
                mlp_ratio=4.0,
                aa_start_block_idx=0,
                img_pos_embed_shape=518 // model_config.patch_shape[0],
                n_cls_tokens=0,
                with_default_global_cls_tokens=False,
                rope_frequency=100.0,
                ffn_type=ffn_type,
                checkpointing="all" if config.checkpointing else "none",
                n_cls_tokens_aa=2,  # [CLS + SRC-Time]
                use_modulated_attention=True,
            )
            self.source_time_embed = ContinuousTimeEmbed(
                patch_shape=(1, 1),
                embed_dim=self.embed_dim,
                frequency_embedding_dim=config.time_encoding_dim,
                max_period=500.0,
            )
            self.source_time_norm = nn.LayerNorm(self.embed_dim)
            self.target_time_embed = ContinuousTimeEmbed(
                patch_shape=(1, 1),
                embed_dim=self.embed_dim // 2,
                frequency_embedding_dim=config.time_encoding_dim,
                max_period=500.0,
            )
            self.final_motion_head = DPTFullHead(
                input_dim=self.embed_dim * 2,
                reassemble_hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
                reassemble_dim=config.dpt_dim,
                output_dim=3 + 3,
                n_blocks=len(config.dpt_reassemble_hidden_dims),
                head_before_conv="1-layer",
                head_after_conv="2-layers",
                head_after_conv_dim=32,
                pos_embed_strength=0.1,
                checkpointing=config.checkpointing,
            )
            self.cls_token_norm = nn.LayerNorm(self.embed_dim)

        def _encode_timestamps_us(
            self, time_remappings: list[TimeRemapping], timestamps_us: torch.Tensor, embed_block: ContinuousTimeEmbed
        ) -> torch.Tensor:
            """
            Encode timestamps into continuous time embeddings.
            Input:
                time_remappings: (B, )
                timestamps_us: (B, V, H, W, 1)
                embed_block: ContinuousTimeEmbed
            Output:
                t_embed: (B, V, C)
            """
            B, V, H, W, _ = timestamps_us.shape
            frame_timestamp_us = timestamps_us[:, :, H // 2, W // 2, 0]
            t_float = torch.stack(
                [
                    time_remappings[bidx].timestamps_us_to_continuous_times(frame_timestamp_us[bidx])
                    for bidx in range(B)
                ],
                dim=0,
            )  # (B, V)
            t_embed = embed_block(rearrange(t_float, "B V -> (B V) 1 1"))
            t_embed = rearrange(t_embed, "(B V) 1 1 C -> B V C", B=B, V=V)
            return t_embed

        @ScopedTimer("KelvinDPTDecoder.motion_head")
        def forward(
            self,
            encoded_latent: KelvinMultiscaleFeaturesLatent,
            output_shape: tuple[int, int],
            fusion_features: torch.Tensor | None,
            chunk_size: int,
            *,  # Force keyword-only for timing to be clear
            time_remappings: list[TimeRemapping],
            source_timestamps_us: torch.Tensor,
            prev_target_timestamps_us: torch.Tensor,
            next_target_timestamps_us: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Input:
                encoded_latent: (B, V, h, w, C) & (B, V, n_cls_tokens, C)
                time_remappings: (B, )
                source_timestamps_us: (B, V, H, W, 1)
                prev_target_timestamps_us: (B, V, H, W, 1)
                next_target_timestamps_us: (B, V, H, W, 1)
            Output:
                Flow to be added on XYZ to reach prev timestamp: (B, V, H, W, 3)
                Flow to be added on XYZ to reach next timestamp: (B, V, H, W, 3)
            """
            H, W = output_shape
            B, V, _, _, _ = prev_target_timestamps_us.shape
            assert prev_target_timestamps_us.shape == (B, V, H, W, 1), (
                f"Expected (B, V, H, W, 1), got {prev_target_timestamps_us.shape}"
            )
            assert prev_target_timestamps_us.shape == next_target_timestamps_us.shape

            prev_target_t_embed = self._encode_timestamps_us(
                time_remappings, prev_target_timestamps_us, self.target_time_embed
            )
            next_target_t_embed = self._encode_timestamps_us(
                time_remappings, next_target_timestamps_us, self.target_time_embed
            )
            source_t_embed = self._encode_timestamps_us(time_remappings, source_timestamps_us, self.source_time_embed)
            source_t_embed = self.source_time_norm(source_t_embed)

            multiscale_features: list[torch.Tensor] = []
            for feat, src_cls_token in zip(encoded_latent.features, unpack_optional(encoded_latent.cls_tokens)):
                with torch.autocast("cuda", enabled=True):
                    src_cls_token = self.cls_token_norm(src_cls_token[..., self.embed_dim :])
                    img_feat, _ = self.vit.get_intermediate_features(
                        # Last the last half (x) and remove local_x part.
                        img_tokens=feat[..., self.embed_dim :],
                        block_indices=[len(self.vit.blocks) - 1],
                        global_cls_token=torch.cat([src_cls_token, source_t_embed.unsqueeze(-2)], dim=-2),
                        modulation_cond=torch.cat([prev_target_t_embed, next_target_t_embed], dim=-1),
                    )
                multiscale_features.append(rearrange(img_feat[-1], "B V h w C -> (B V) h w C"))

            x = self.final_motion_head(
                multiscale_features, output_shape=output_shape, fusion_features=fusion_features, chunk_size=chunk_size
            )
            x = rearrange(x, "(B V) C H W -> B V H W C", B=B, V=V)
            flow_prev, flow_next = x.split([3, 3], dim=-1)
            return flow_prev, flow_next

    def __init__(self, config: KelvinDPTDecoderConfig, model_config: KelvinModelConfig):
        super().__init__()
        self.config = config
        embed_dim = model_config.encoder.embed_dim
        self.n_blocks = 1
        if isinstance(model_config.encoder, KelvinDAv3EncoderConfig):
            self.n_blocks = len(model_config.encoder.take_block_indices)
        elif isinstance(model_config.encoder, KelvinVGGTEncoderConfig):
            self.n_blocks = len(model_config.encoder.return_attn_block_idx)
        assert self.n_blocks == len(config.dpt_reassemble_hidden_dims), "Number of blocks must match"

        # Pre-training heads.
        # Output depth and depth confidence
        # or alternatively, world-points and its confidence (need the 5-layers before-conv setting)
        self.depth_head = DPTFullHead(
            input_dim=embed_dim,
            reassemble_hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
            reassemble_dim=config.dpt_dim,
            output_dim=1 + 1,
            n_blocks=self.n_blocks,
            head_before_conv="1-layer",
            head_after_conv="2-layers",
            head_after_conv_dim=32,
            pos_embed_strength=0.1,
            checkpointing=config.checkpointing,
        )
        if config.use_encoder_depth:
            # Pretrained-depth ablation: the encoder's own KelvinVGGTFeaturesLatent
            # .encoder_depth drives unprojection instead, so this head would be dead weight.
            del self.depth_head

        # Up-scale RGB features for fusion (helps with HD output).
        rgb_fusion_dim = config.dpt_dim // 2
        self.rgb_fusion = nn.Sequential(
            nn.Conv2d(3, rgb_fusion_dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(rgb_fusion_dim // 4, rgb_fusion_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(rgb_fusion_dim // 2, rgb_fusion_dim, 3, 1, 1),
            nn.GELU(),
        )

        # Context-training heads (RGB, world-normal, semantic-Logits).
        self.n_semantic_classes = len(KelvinSemanticClass)
        self.context_head = DPTFullHead(
            input_dim=embed_dim,
            reassemble_hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
            reassemble_dim=config.dpt_dim,
            output_dim=3 + 3 + self.n_semantic_classes,
            n_blocks=self.n_blocks,
            head_before_conv="1-layer",
            head_after_conv="2-layers",
            head_after_conv_dim=32,
            pos_embed_strength=0.1,
            checkpointing=config.checkpointing,
        )
        self.context_head_init_values = [0.0] * 3 + [0.0, -1.0, 0.0] + [float("nan")] * self.n_semantic_classes
        # Time-conditioned training heads (motion-offset for now)
        self.context_motion_head: KelvinDPTDecoder.TimeModulatedMotionHead | nn.Identity = nn.Identity()
        if config.motion_depth > 0:
            self.context_motion_head = self.TimeModulatedMotionHead(config, model_config)

        # GS-training heads (scale, world-quaternion, opacity, higher-order SHs, depth-offset, uv-offset)
        self.sh_degree = 0
        self.depth_offset_dim = 1 if config.depth_offset else 0
        self.uv_offset_dim = 2 if config.uv_offset else 0
        gs_output_dim = (
            3 + 4 + 1 + sh_degree_to_specular_dim(self.sh_degree) + self.depth_offset_dim + self.uv_offset_dim
        )
        self.gaussians_head = DPTFullHead(
            input_dim=embed_dim,
            reassemble_hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
            reassemble_dim=config.dpt_dim,
            output_dim=gs_output_dim,
            n_blocks=self.n_blocks,
            head_before_conv="1-layer",
            head_after_conv="2-layers",
            head_after_conv_dim=32,
            pos_embed_strength=0.1,
            checkpointing=config.checkpointing,
        )
        self.gaussians_head_init_values = (
            [float("nan")] * 3
            + [float("nan")] * 4
            + [float("nan")] * (1 + sh_degree_to_specular_dim(self.sh_degree))
            + [0.0] * (self.depth_offset_dim + self.uv_offset_dim)
        )

        self.cuboids_dims_padding = nn.Buffer(torch.tensor(model_config.track_padding_m, dtype=torch.float32))
        self.gaussian_activations = GaussianActivations(model_config.activations)

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        if "dav3" not in loaded_state_dicts:
            logger.warning("[KelvinDPTDecoder] No dav3 state dict found, skipping weight initialization.")
            return

        state_dict = convert_dav3_state_dict_to_instant_nurec(
            loaded_state_dicts["dav3"],
            patch_embed_name=None,
            backbone_name=None,
            dpt_reassemble_name=None if self.config.use_encoder_depth else "depth_head.reassemble",
            dpt_depth_head_name=None if self.config.use_encoder_depth else "depth_head.fusion_head",
            dpt_rays_head_name=None,
            camera_encoder_name=None,
        )
        self.context_head.zero_init(init_values=self.context_head_init_values)
        self.gaussians_head.zero_init(init_values=self.gaussians_head_init_values)
        for prefix, module_state_dict in (
            ("rgb_fusion", self.rgb_fusion.state_dict()),
            ("context_head", self.context_head.state_dict()),
            ("context_motion_head", self.context_motion_head.state_dict()),
            ("gaussians_head", self.gaussians_head.state_dict()),
            ("gaussian_activations", self.gaussian_activations.state_dict()),
        ):
            state_dict |= {f"{prefix}.{k}": v for k, v in module_state_dict.items()}

        state_dict["cuboids_dims_padding"] = self.cuboids_dims_padding.data
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def inverse_log_transform(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.expm1(torch.abs(x)))

    def _log_gaussian_statistics(
        self,
        media_logger: BatchMediaLogger,
        gaussian_params: GaussianParams,
        supervision_pack: KelvinInstantNuRecSupervisionPack,
        context: DataAndRenderingBatch,
        grid_width: int,
    ) -> None:
        """
        Log statistics of the gaussian primitives.
        """
        if supervision_pack.context_rgb is not None:
            media_logger.log_image(
                "Context RGB",
                make_image_grid(
                    [t for t in (supervision_pack.context_rgb.detach().cpu().numpy() * 255).astype(np.uint8)],
                    grid_width=grid_width,
                ),
            )
        if (
            supervision_pack.context_depth is not None
            and supervision_pack.context_xyz is not None
            and context.rendering is not None
        ):
            rays = unpack_optional(unpack_optional(context.rendering.camera).rays)
            ref_xyz = rays[..., :3] + rays[..., 3:] * supervision_pack.context_depth
            diff_norm = torch.linalg.norm(ref_xyz - supervision_pack.context_xyz, dim=-1).detach().cpu().numpy()
            diff_norm_img = make_image_grid(
                [scalar2img(t, vmin=0.0, vmax=1.0) for t in diff_norm],
                grid_width=grid_width,
            )
            media_logger.log_image("Context XYZ Diff", diff_norm_img)
        if supervision_pack.context_world_normal is not None:
            pred_normal_img = ((supervision_pack.context_world_normal.detach().cpu().numpy() + 1.0) * 0.5 * 255).astype(
                np.uint8
            )
            normal_imgs = [t for t in pred_normal_img]
            if context.data.camera is not None and (gt_normal := context.data.camera.labels.normals) is not None:
                normal_imgs += [t for t in ((gt_normal.detach().cpu().numpy() + 1.0) * 0.5 * 255).astype(np.uint8)]
            media_logger.log_image(
                "Context World Normal",
                make_image_grid(normal_imgs, grid_width=grid_width),
            )
        if supervision_pack.context_semantic_logits is not None and context.data.camera is not None:
            pred_sem_img = KelvinSemanticClass.semantics_to_rgb(
                torch.argmax(supervision_pack.context_semantic_logits.detach(), dim=-1, keepdim=True).to(torch.uint8)
            )
            gt_sem_img = KelvinSemanticClass.semantics_to_rgb(
                KelvinSemanticClass.get_target_from_frame_labels(context.data.camera.labels)
            )
            media_logger.log_image(
                "Context Semantics",
                make_image_grid(
                    [p.cpu().numpy() for p in pred_sem_img] + [g.cpu().numpy() for g in gt_sem_img],
                    grid_width=grid_width,
                ),
            )

        opacity = gaussian_params.opacity.detach()[..., 0].cpu().numpy()
        media_logger.log_image(
            "Context Opacity",
            make_image_grid(
                [t for t in (opacity * 255).astype(np.uint8)],
                grid_width=grid_width,
            ),
        )

    def _dump_meshing_data(
        self,
        gaussian_params: GaussianParams,
        supervision_pack: KelvinInstantNuRecSupervisionPack,
        context: DataAndRenderingBatch,
    ):
        """
        Dump useful data for meshing using meshing tool.
        """
        rendering = unpack_optional(unpack_optional(context.rendering).camera)
        with tempfile.NamedTemporaryFile(mode="wb", prefix="meshing_data_", suffix=".pkl", delete=False) as f:
            save_path = f.name
            pickle.dump(
                {
                    "xyz": unpack_optional(gaussian_params.xyz).detach().cpu().numpy(),  # [V, H, W, 3]
                    "depth": unpack_optional(supervision_pack.context_depth).detach().cpu().numpy(),  # [V, H, W, 1]
                    "rgb": unpack_optional(supervision_pack.context_rgb).detach().cpu().numpy(),  # [V, H, W, 3]
                    "rays": unpack_optional(rendering.rays).detach().cpu().numpy(),  # [V, H, W, 6]
                    "intrinsics": [
                        cast(ConcreteCameraModelParametersUnion, unpack_optional(i)).to_dict()
                        for i in rendering.sensor_model_parameters
                    ],
                },
                f,
            )
        logger.info(f"Dumped meshing information to {save_path}...")

    def get_potential_unused_parameters(self) -> Iterator[torch.nn.Parameter]:
        # Note -- if we shard model parameters need to reconsider this design.
        return self.gaussians_head.parameters()

    @ScopedTimer("KelvinDPTDecoder.decode")
    @torch.autocast("cuda", enabled=False)
    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> list[KelvinDecoderReturn]:
        """
        The returned GaussianParams will have shape (B, V, H, W, C)
        """
        assert isinstance(encoded_latent, KelvinMultiscaleFeaturesLatent), (
            "Encoded latent must be a KelvinMultiscaleFeaturesLatent"
        )
        renderings = [unpack_optional(unpack_optional(batch.rendering).camera) for batch in batches]
        data = [unpack_optional(batch.data.camera) for batch in batches]

        img_rgb = torch.stack([unpack_optional(d.labels.rgb) for d in data], dim=0)
        B, V, H, W, _ = img_rgb.shape
        img_feats = [rearrange(feat, "B V h w C -> (B V) h w C") for feat in encoded_latent.features]

        # Forward and activate depth
        if self.config.use_encoder_depth:
            # Pretrained-depth ablation: the encoder's pretrained depth head output
            # drives unprojection; self.depth_head was deleted in __init__. The encoder depth is
            # already metric z-depth (same distance-along-optical-axis convention as this branch's
            # own pred_depth -- see KelvinVGGTEncoder._run_depth's docstring for the evidence),
            # so NO scene_rescale correction is applied: the -log(scene_rescale) shift in the
            # else-branch exists because Kelvin's own head predicts in scene_rescale-scaled
            # units, which the vggt head (metric-trained, metric-conditioned) does not.
            if not isinstance(encoded_latent, KelvinVGGTFeaturesLatent) or encoded_latent.encoder_depth is None:
                raise ValueError(
                    "decoder.use_encoder_depth=True requires a KelvinVGGTFeaturesLatent with a populated "
                    "encoder_depth (i.e. encoder.use_vggt_depth=true), but got "
                    f"{type(encoded_latent).__name__} with encoder_depth="
                    f"{getattr(encoded_latent, 'encoder_depth', None)!r}. "
                    "Set model.encoder.use_vggt_depth=true or disable model.decoder.use_encoder_depth."
                )
            pred_depth = encoded_latent.encoder_depth.to(dtype=img_rgb.dtype)  # (B, V, H, W, 1), metric
            assert pred_depth.shape == (B, V, H, W, 1), (
                f"encoder_depth shape {tuple(pred_depth.shape)} != expected {(B, V, H, W, 1)}"
            )
            pred_depth_conf = torch.ones_like(pred_depth)  # (B, V, H, W, 1)
        else:
            depth_and_dconf = self.depth_head(img_feats, output_shape=(H, W), chunk_size=self.config.dpt_chunk_size)
            depth_and_dconf = rearrange(depth_and_dconf, "(B V) C H W -> B V C H W", B=B, V=V)
            pred_depth = torch.exp(depth_and_dconf[:, :, 0].unsqueeze(-1) - math.log(scene_rescale))  # (B, V, H, W, 1)
            pred_depth_conf = torch.exp(depth_and_dconf[:, :, 1].unsqueeze(-1)) + 1.0  # (B, V, H, W, 1)

        # Forward and activate context
        img_rgb = rearrange(img_rgb, "B V H W C -> (B V) C H W")
        if self.config.checkpointing:
            rgb_fusion_features = torch.utils.checkpoint.checkpoint(self.rgb_fusion, img_rgb, use_reentrant=False)
        else:
            rgb_fusion_features = self.rgb_fusion(img_rgb)
        context_features_tensor = self.context_head(
            img_feats, output_shape=(H, W), fusion_features=rgb_fusion_features, chunk_size=self.config.dpt_chunk_size
        )
        context_features_tensor = rearrange(context_features_tensor, "(B V) C H W -> B V H W C", B=B, V=V)
        (
            context_rgb,
            context_world_normal,
            context_semantic_logits,
        ) = context_features_tensor.split(
            [3, 3, self.n_semantic_classes],
            dim=-1,
        )
        context_rgb = self.gaussian_activations.rgb(context_rgb)
        context_world_normal = torch.nn.functional.normalize(context_world_normal, dim=-1)

        # For motion, determine the gap based on the time remappings for now
        source_timestamps_us = torch.stack(
            [unpack_optional(renderings[bidx].rays_timestamps_us) for bidx in range(B)], dim=0
        )
        frame_gap_timestamps_us = torch.stack(
            [time_remappings[bidx].frame_gap_timestamps_us for bidx in range(B)], dim=0
        ).to(img_rgb.device)
        prev_target_timestamps_us = source_timestamps_us - frame_gap_timestamps_us[..., 0][..., None, None, None]
        next_target_timestamps_us = source_timestamps_us + frame_gap_timestamps_us[..., 1][..., None, None, None]
        # This typically gives sharp motion boundary.
        context_dynamic_mask = torch.argmax(context_semantic_logits, dim=-1) == KelvinSemanticClass.MOVABLE.value

        context_prev_flow: torch.Tensor | None = None
        context_next_flow: torch.Tensor | None = None
        motion_supervisions: list[list[KelvinMotionSupervision]] = [[] for _ in range(B)]
        if isinstance(self.context_motion_head, self.TimeModulatedMotionHead):
            context_prev_flow, context_next_flow = self.context_motion_head.forward(
                encoded_latent,
                output_shape=(H, W),
                fusion_features=rgb_fusion_features if self.config.fusion_for_gs_motion else None,
                chunk_size=self.config.dpt_chunk_size,
                time_remappings=time_remappings,
                source_timestamps_us=source_timestamps_us,
                prev_target_timestamps_us=prev_target_timestamps_us,
                next_target_timestamps_us=next_target_timestamps_us,
            )
            context_prev_flow = context_prev_flow / scene_rescale
            context_next_flow = context_next_flow / scene_rescale
            motion_supervisions = [
                [
                    KelvinMotionSupervision(
                        source_timestamps_us=source_timestamps_us[bidx],
                        target_timestamps_us=prev_target_timestamps_us[bidx],
                        context_flow=context_prev_flow[bidx],
                    ),
                    KelvinMotionSupervision(
                        source_timestamps_us=source_timestamps_us[bidx],
                        target_timestamps_us=next_target_timestamps_us[bidx],
                        context_flow=context_next_flow[bidx],
                    ),
                ]
                for bidx in range(B)
            ]

        # If cuboid tracks are provided, use them instead.
        if cuboid_tracks is not None:
            # No need to re-scale points here since both cuboids and pred_depth are already scaled.
            context_prev_flow_list: list[torch.Tensor] = []
            context_next_flow_list: list[torch.Tensor] = []
            context_dynamic_mask_list: list[torch.Tensor] = []
            for bidx in range(B):
                batch_cuboid_tracks = cuboid_tracks[bidx]
                context_xyz = (
                    pred_depth[bidx].detach()
                    / renderings[bidx].distance_to_depth_scale
                    * renderings[bidx].rays[..., 3:]
                    + renderings[bidx].rays[..., :3]
                )
                tracks_idx = associate_points_with_cuboid_tracks(
                    points=context_xyz,
                    points_timestamps_us=source_timestamps_us[bidx],
                    points_dynamic_mask=context_dynamic_mask[bidx].unsqueeze(-1),
                    cuboid_tracks=batch_cuboid_tracks,
                    cuboids_dims_padding=self.cuboids_dims_padding,
                )
                # Unassign points associated to static cuboids.
                dynamic_track_mask = (batch_cuboid_tracks.tracks_flags & TrackFlags.DYNAMIC) != 0
                if dynamic_track_mask.numel() > 0:
                    is_dynamic_track = dynamic_track_mask[tracks_idx.clamp_min(0).to(torch.long)]
                    tracks_idx.masked_fill_(~is_dynamic_track, -1)
                dynamic_mask, (prev_world_points, next_world_points) = warp_points_with_cuboid_tracks(
                    points=context_xyz,
                    source_timestamps_us=source_timestamps_us[bidx],
                    target_timestamps_us_list=[prev_target_timestamps_us[bidx], next_target_timestamps_us[bidx]],
                    cuboid_tracks=batch_cuboid_tracks,
                    tracks_idx=tracks_idx,
                )
                context_prev_flow_list.append(prev_world_points - context_xyz)
                context_next_flow_list.append(next_world_points - context_xyz)
                context_dynamic_mask_list.append(dynamic_mask.squeeze(-1))

            # Replace with ones from gt cuboids.
            context_prev_flow = torch.stack(context_prev_flow_list, dim=0)
            context_next_flow = torch.stack(context_next_flow_list, dim=0)
            context_dynamic_mask = torch.stack(context_dynamic_mask_list, dim=0)

        if context_prev_flow is None or context_next_flow is None:
            raise RuntimeError(
                "No motion head found in the model, and cuboid tracks are not provided. Dynamic actors cannot be inferred."
            )

        # Forward and activate gaussian parameters
        gs_params_tensor = self.gaussians_head(
            img_feats,
            output_shape=(H, W),
            fusion_features=rgb_fusion_features if self.config.fusion_for_gs_motion else None,
            chunk_size=self.config.dpt_chunk_size,
        )
        gs_params_tensor = rearrange(gs_params_tensor, "(B V) C H W -> B V H W C", B=B, V=V)
        (
            gs_scale,
            gs_world_quaternion,
            gs_opacity,
            gs_specular,  # TODO: wire gs_specular through to GaussianParams when sh_degree > 0
            gs_depth_offset,
            gs_uv_offset,
        ) = gs_params_tensor.split(
            [3, 4, 1, sh_degree_to_specular_dim(self.sh_degree), self.depth_offset_dim, self.uv_offset_dim],
            dim=-1,
        )
        gs_depth = pred_depth
        if self.config.depth_offset:
            gs_depth_offset = gs_depth_offset / scene_rescale
            if media_logger is not None:
                media_logger.log("gs_stats/depth_offset", gs_depth_offset.detach().mean().item())
            # Linear activation on depth offset
            gs_depth = gs_depth + gs_depth_offset
        gs_distance = torch.stack([gs_depth[bidx] / renderings[bidx].distance_to_depth_scale for bidx in range(B)])

        # If scales and the potential UV offsets are predicted in the pixel unit (so that they're resolution-agnostic).
        # We need to scale them by the pixel scale, defined by footprint(cone) * distance.
        # Note that if rendered via 2D evaluation, the most accurate definition should be footprint(plane) * depth.
        # Here we simply assume that 3DGUT is always used.
        pixel_scale = torch.stack([renderings[bidx].ray_footprints for bidx in range(B)]) * gs_distance
        gs_scale = self.gaussian_activations.scale(gs_scale, scene_rescale=scene_rescale, pixel_scale=pixel_scale)
        gs_valid_mask = KelvinSemanticClass.opacity_mask_from_semantic_probs(
            torch.softmax(context_semantic_logits, dim=-1)
        )  # (B, V, H, W, 1)
        gs_opacity = self.gaussian_activations.opacity(gs_opacity) * (gs_valid_mask > 0.5).float().detach()
        gs_world_quaternion = self.gaussian_activations.rotation(gs_world_quaternion)
        gs_xyz = torch.stack(
            [renderings[bidx].rays[..., :3] + renderings[bidx].rays[..., 3:] * gs_distance[bidx] for bidx in range(B)]
        )
        if self.config.uv_offset:
            if media_logger is not None:
                media_logger.log("gs_stats/uv_offset", gs_uv_offset.detach().mean().item())
            gs_uv_offset = einsum(
                torch.stack(
                    [renderings[bidx].uv_directions_frame_end for bidx in range(B)],
                ),
                gs_uv_offset * pixel_scale,
                "B V UV D, B V H W UV -> B V H W D",
            )
            gs_xyz = gs_xyz + gs_uv_offset

        gs_params = GaussianParams(
            rgb=context_rgb,
            scale=gs_scale,
            rotation=gs_world_quaternion,
            opacity=gs_opacity,
            xyz=gs_xyz,
            activated=True,
        )

        supervision_packs = [
            KelvinInstantNuRecSupervisionPack(
                context_rgb=context_rgb[bidx],
                context_depth=pred_depth[bidx],
                context_depth_conf=pred_depth_conf[bidx],
                context_semantic_logits=context_semantic_logits[bidx],
                context_world_normal=context_world_normal[bidx],
                motion_supervisions=motion_supervisions[bidx],
            )
            for bidx in range(B)
        ]

        # Log Gaussian statistics and supervision packs information
        if media_logger is not None and media_logger.should_log_media:
            num_views = len(set([meta.unique_sensor_idx for meta in data[0].meta]))
            self._log_gaussian_statistics(
                media_logger,
                gs_params[0],
                supervision_packs[0],
                batches[0],
                grid_width=V // num_views,
            )
            # Log unprojected DPT point cloud (validation only; PLY overwritten each time; disabled for now)
            # media_logger.log_ply_point_cloud(
            #     "gspcd",
            #     gs_xyz[0].detach().reshape(-1, 3).cpu().numpy(),
            #     color=context_rgb[0].detach().reshape(-1, 3).cpu().numpy(),
            #     other_attributes={
            #         "confidence": pred_depth_conf[0].detach().reshape(-1).cpu().numpy(),
            #         "view_index": np.repeat(np.arange(V, dtype=np.uint8), H * W),
            #     },
            # )

        # Optionally dump meshing data
        # self._dump_meshing_data(gs_params[0], supervision_packs[0], batches[0])

        # Build up the primitive
        return_values: list[KelvinDecoderReturn] = []
        for bidx in range(B):
            gs_bidx = gs_params[bidx].flatten()
            world_points = unpack_optional(gs_bidx.xyz)
            prev_world_points = world_points + context_prev_flow[bidx].reshape(-1, 3)
            next_world_points = world_points + context_next_flow[bidx].reshape(-1, 3)

            dynamic_mask = context_dynamic_mask[bidx].reshape(-1)
            static_mask = torch.where(~dynamic_mask)[0]
            dynamic_mask = torch.where(dynamic_mask)[0]

            # Derive per-gaussian semantic class from logits (argmax) for the static layer
            sem_class = torch.argmax(context_semantic_logits[bidx], dim=-1).reshape(-1)  # (V*H*W,)
            semantic_class_static = sem_class[static_mask].unsqueeze(-1).to(torch.uint8)  # (n_static, 1)
            normals_static = context_world_normal[bidx].reshape(-1, 3)[static_mask]  # (n_static, 3)

            static_layer = KelvinStaticLayer(
                positions=world_points[static_mask],
                rotations=gs_bidx.rotation[static_mask],
                scales=gs_bidx.scale[static_mask],
                densities=gs_bidx.opacity[static_mask],
                rgb=gs_bidx.rgb[static_mask],
                semantic_class=semantic_class_static,
                normals=normals_static,
            )

            dynamic_layer = KelvinDynamicLayer(
                keyframe_positions=torch.stack(
                    [
                        prev_world_points[dynamic_mask],
                        world_points[dynamic_mask],
                        next_world_points[dynamic_mask],
                    ],
                    dim=1,
                ),
                keyframe_timestamps_us=torch.stack(
                    [
                        prev_target_timestamps_us[bidx].reshape(-1)[dynamic_mask],
                        source_timestamps_us[bidx].reshape(-1)[dynamic_mask],
                        next_target_timestamps_us[bidx].reshape(-1)[dynamic_mask],
                    ],
                    dim=1,
                ),
                max_densities=gs_bidx.opacity[dynamic_mask],
                rotations=gs_bidx.rotation[dynamic_mask],
                scales=gs_bidx.scale[dynamic_mask],
                rgb=gs_bidx.rgb[dynamic_mask],
            )
            # Dynamic layers have a typical smaller timespan so their presence should be guaranteed.
            dynamic_layer = dynamic_layer.ensure_minimum_density(0.75)
            return_values.append(
                KelvinDecoderReturn(
                    static_layer=static_layer,
                    dynamic_layers=[dynamic_layer],
                    supervision_pack=supervision_packs[bidx],
                )
            )

        return return_values


class KelvinPointQueryCADecoder(KelvinDPTDecoder):
    """Replaces the DPT gaussian head with a cross-attention head that uses
    depth-derived xyz positions (2D grid tokenization) as queries."""

    def __init__(self, config: KelvinPointQueryCADecoderConfig, model_config: KelvinModelConfig):
        dpt_config = KelvinDPTDecoderConfig(
            name="dpt-decoder",
            dpt_dim=config.dpt_dim,
            dpt_reassemble_hidden_dims=config.dpt_reassemble_hidden_dims,
            checkpointing=config.checkpointing,
            dpt_chunk_size=config.dpt_chunk_size,
            time_encoding_dim=config.time_encoding_dim,
            motion_depth=config.motion_depth,
        )
        super().__init__(dpt_config, model_config)
        self.config = config  # type: ignore[assignment]

        del self.gaussians_head
        del self.gaussians_head_init_values
        del self.sh_degree
        del self.depth_offset_dim
        del self.uv_offset_dim

        embed_dim = model_config.encoder.embed_dim
        num_heads = model_config.encoder.n_heads

        gs_per_token = config.grid_center_stride**2
        n_gs_params = 3 + 4 + 1  # scale, rotation, opacity

        self.xyz_embed = nn.Sequential(
            nn.Linear(gs_per_token * 3, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.ca_feature_norm = nn.LayerNorm(embed_dim)
        self.ca_kv_projector = KVProjector(
            dim=embed_dim,
            n_heads=num_heads,
            kv_bias=True,
            k_norm=config.use_qk_norm,
        )
        self.ca_blocks = TypedModuleList(
            [
                CrossAttentionBlock(
                    embed_dim,
                    num_heads,
                    qkv_bias=True,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    qk_norm=config.use_qk_norm,
                    mlp_ratio=4.0,
                    dropout=0.0,
                    layer_scale_init_values=config.layer_scale_init_values,
                    kv_projector=None,
                )
                for _ in range(config.ca_depth)
            ]
        )

        # Number of encoder feature taps feeding the cross-attention KV projection.
        # Mirror KelvinDPTDecoder.__init__'s encoder handling so the point-query head
        # composes with the VGGT/VGGT encoder (whose taps come from
        # `return_attn_block_idx`) as well as DAv3 (`take_block_indices`). The parent
        # __init__ already set self.n_blocks to this exact tap count (and asserted it
        # equals len(dpt_reassemble_hidden_dims)), so reuse it to keep the two in lockstep.
        if not isinstance(model_config.encoder, (KelvinDAv3EncoderConfig, KelvinVGGTEncoderConfig)):
            raise ValueError(
                "KelvinPointQueryCADecoder requires a KelvinDAv3EncoderConfig or "
                f"KelvinVGGTEncoderConfig encoder, got {type(model_config.encoder).__name__}"
            )
        n_scales = self.n_blocks
        self.ca_kv_proj = nn.Linear(n_scales * embed_dim, embed_dim, bias=False)

        self.gs_linear = nn.Linear(embed_dim, gs_per_token * n_gs_params)
        self.last_voxel_dedup_diagnostics: list[VoxelDedupDiagnostics] | None = None

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        if "dav3" not in loaded_state_dicts:
            logger.warning("[KelvinPointQueryCADecoder] No dav3 state dict found, skipping weight initialization.")
            return

        state_dict = convert_dav3_state_dict_to_instant_nurec(
            loaded_state_dicts["dav3"],
            patch_embed_name=None,
            backbone_name=None,
            dpt_reassemble_name="depth_head.reassemble",
            dpt_depth_head_name="depth_head.fusion_head",
            dpt_rays_head_name=None,
            camera_encoder_name=None,
        )
        self.context_head.zero_init(init_values=self.context_head_init_values)

        for m in [self.xyz_embed, self.ca_feature_norm, self.ca_kv_projector, self.ca_blocks]:
            m.apply(_tokengs_init_weights)
        torch.nn.init.trunc_normal_(self.gs_linear.weight, std=0.002)
        if self.gs_linear.bias is not None:
            torch.nn.init.constant_(self.gs_linear.bias, 0)
        for prefix, mod in (
            ("rgb_fusion", self.rgb_fusion),
            ("context_head", self.context_head),
            ("context_motion_head", self.context_motion_head),
            ("xyz_embed", self.xyz_embed),
            ("ca_feature_norm", self.ca_feature_norm),
            ("ca_kv_projector", self.ca_kv_projector),
            ("ca_blocks", self.ca_blocks),
            ("ca_kv_proj", self.ca_kv_proj),
            ("gs_linear", self.gs_linear),
            ("gaussian_activations", self.gaussian_activations),
        ):
            state_dict |= {f"{prefix}.{k}": v for k, v in mod.state_dict().items()}

        state_dict["cuboids_dims_padding"] = self.cuboids_dims_padding.data
        self.load_state_dict(state_dict, strict=True)

    def _predict_point_query_gaussian_attributes(
        self,
        *,
        encoded_latent: KelvinMultiscaleFeaturesLatent,
        token_groups: list[GridTokens],
        all_tokens: GridTokens,
        renderings,
        context_metric_distance: torch.Tensor,
        scene_rescale: float,
        media_logger: BatchMediaLogger | None,
    ) -> torch.Tensor:
        """Return flattened ``[B,N,8]`` scale/rotation/opacity logits."""
        del all_tokens, renderings, context_metric_distance, scene_rescale, media_logger
        assert isinstance(self.config, KelvinPointQueryCADecoderConfig)
        query_xyz_all = torch.cat([tokens.query_xyz for tokens in token_groups], dim=1)
        queries = self.xyz_embed(query_xyz_all.flatten(-2, -1))
        kv_features = self.ca_kv_proj(
            torch.cat([rearrange(feature, "B V h w C -> B (V h w) C") for feature in encoded_latent.features], dim=-1)
        )
        kv_features = self.ca_feature_norm(kv_features)
        k_proj, v_proj = self.ca_kv_projector(k=kv_features, v=kv_features)
        for block in self.ca_blocks:
            queries = block(queries, k_proj, v_proj)

        points_per_token = self.config.grid_center_stride**2
        attributes_by_token = rearrange(
            self.gs_linear(queries),
            "B M (K P) -> B M K P",
            K=points_per_token,
        )
        offsets = [0]
        for tokens in token_groups:
            offsets.append(offsets[-1] + tokens.query_xyz.shape[1])
        return torch.cat(
            [
                rearrange(
                    attributes_by_token[:, offsets[index] : offsets[index + 1]],
                    "B M K P -> B (M K) P",
                )
                for index in range(len(token_groups))
            ],
            dim=1,
        )

    def get_potential_unused_parameters(self) -> Iterator[nn.Parameter]:
        return self.context_motion_head.parameters()

    @ScopedTimer("KelvinPointQueryCADecoder.decode")
    @torch.autocast("cuda", enabled=False)
    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> list[KelvinDecoderReturn]:
        assert isinstance(encoded_latent, KelvinMultiscaleFeaturesLatent), (
            "Encoded latent must be a KelvinMultiscaleFeaturesLatent"
        )
        assert isinstance(self.config, KelvinPointQueryCADecoderConfig)
        renderings = [unpack_optional(unpack_optional(batch.rendering).camera) for batch in batches]
        data = [unpack_optional(batch.data.camera) for batch in batches]

        img_rgb = torch.stack([unpack_optional(d.labels.rgb) for d in data], dim=0)
        B, V, H, W, _ = img_rgb.shape
        img_feats = [rearrange(feat, "B V h w C -> (B V) h w C") for feat in encoded_latent.features]

        # --- Timestamps for motion ---
        source_timestamps_us = torch.stack(
            [unpack_optional(renderings[bidx].rays_timestamps_us) for bidx in range(B)], dim=0
        )  # (B, V, H, W, 1)
        frame_gap_timestamps_us = torch.stack(
            [time_remappings[bidx].frame_gap_timestamps_us for bidx in range(B)], dim=0
        ).to(img_rgb.device)  # (B, 2)
        prev_target_timestamps_us = source_timestamps_us - frame_gap_timestamps_us[..., 0][..., None, None, None]
        next_target_timestamps_us = source_timestamps_us + frame_gap_timestamps_us[..., 1][..., None, None, None]

        # --- Depth (shared with DPT) ---
        depth_and_dconf = self.depth_head(img_feats, output_shape=(H, W), chunk_size=self.config.dpt_chunk_size)
        depth_and_dconf = rearrange(depth_and_dconf, "(B V) C H W -> B V C H W", B=B, V=V)
        pred_depth = torch.exp(depth_and_dconf[:, :, 0].unsqueeze(-1) - math.log(scene_rescale))
        pred_depth_conf = torch.exp(depth_and_dconf[:, :, 1].unsqueeze(-1)) + 1.0

        # --- Context (shared with DPT, used for supervision + gaussian RGB) ---
        img_rgb_nchw = rearrange(img_rgb, "B V H W C -> (B V) C H W")
        if self.config.checkpointing:
            rgb_fusion_features = torch.utils.checkpoint.checkpoint(self.rgb_fusion, img_rgb_nchw, use_reentrant=False)
        else:
            rgb_fusion_features = self.rgb_fusion(img_rgb_nchw)
        context_features_tensor = self.context_head(
            img_feats, output_shape=(H, W), fusion_features=rgb_fusion_features, chunk_size=self.config.dpt_chunk_size
        )
        context_features_tensor = rearrange(context_features_tensor, "(B V) C H W -> B V H W C", B=B, V=V)
        context_rgb, context_world_normal, context_semantic_logits = context_features_tensor.split(
            [3, 3, self.n_semantic_classes],
            dim=-1,
        )
        context_rgb = self.gaussian_activations.rgb(context_rgb)
        context_world_normal = torch.nn.functional.normalize(context_world_normal, dim=-1)

        # --- XYZ from depth ---
        gs_distance = torch.stack([pred_depth[bidx] / renderings[bidx].distance_to_depth_scale for bidx in range(B)])
        full_rays = torch.stack([renderings[bidx].rays for bidx in range(B)], dim=0)
        full_xyz = torch.stack(
            [renderings[bidx].rays[..., :3] + renderings[bidx].rays[..., 3:] * gs_distance[bidx] for bidx in range(B)]
        )

        # --- Full-resolution flow + dynamic mask (mirrors KelvinDPTDecoder) ---
        #
        # Keep the learned and teacher-forced paths separate.  The learned motion
        # head is supervised against cuboid-warp reference flow even when tracks
        # are available, while track flow remains the stable source used to build
        # dynamic Gaussians.  The previous point-query implementation made these
        # branches mutually exclusive, so every normal NCore batch (which carries
        # cuboid tracks) skipped the motion head and mislabeled teacher-forced flow
        # as a learned prediction in primitive_velocity/context-flow metrics.
        context_dynamic_mask = torch.argmax(context_semantic_logits, dim=-1) == KelvinSemanticClass.MOVABLE.value
        context_prev_flow: torch.Tensor | None = None
        context_next_flow: torch.Tensor | None = None
        motion_supervisions: list[list[KelvinMotionSupervision]] = [[] for _ in range(B)]
        if isinstance(self.context_motion_head, self.TimeModulatedMotionHead):
            context_prev_flow, context_next_flow = self.context_motion_head.forward(
                encoded_latent,
                output_shape=(H, W),
                fusion_features=rgb_fusion_features,
                chunk_size=self.config.dpt_chunk_size,
                time_remappings=time_remappings,
                source_timestamps_us=source_timestamps_us,
                prev_target_timestamps_us=prev_target_timestamps_us,
                next_target_timestamps_us=next_target_timestamps_us,
            )
            context_prev_flow = context_prev_flow / scene_rescale
            context_next_flow = context_next_flow / scene_rescale
            motion_supervisions = [
                [
                    KelvinMotionSupervision(
                        source_timestamps_us=source_timestamps_us[bidx],
                        target_timestamps_us=prev_target_timestamps_us[bidx],
                        context_flow=context_prev_flow[bidx],
                    ),
                    KelvinMotionSupervision(
                        source_timestamps_us=source_timestamps_us[bidx],
                        target_timestamps_us=next_target_timestamps_us[bidx],
                        context_flow=context_next_flow[bidx],
                    ),
                ]
                for bidx in range(B)
            ]

        # If cuboid tracks are provided, use their flow for dynamic Gaussian
        # assembly without replacing the learned-flow supervision above.
        if cuboid_tracks is not None:
            context_prev_flow_list: list[torch.Tensor] = []
            context_next_flow_list: list[torch.Tensor] = []
            context_dynamic_mask_list: list[torch.Tensor] = []
            for bidx in range(B):
                dynamic_track = CuboidTracks.Ops.subset_from_mask(
                    cuboid_tracks[bidx], cuboid_tracks[bidx].tracks_flags & TrackFlags.DYNAMIC != 0
                )
                context_xyz = full_xyz[bidx]
                context_prev_flow_list.append(
                    dynamic_track.warp_world_points_to_timestamps(
                        context_xyz,
                        source_timestamps_us[bidx],
                        prev_target_timestamps_us[bidx],
                        self.cuboids_dims_padding,
                    )
                    - context_xyz
                )
                context_next_flow_list.append(
                    dynamic_track.warp_world_points_to_timestamps(
                        context_xyz,
                        source_timestamps_us[bidx],
                        next_target_timestamps_us[bidx],
                        self.cuboids_dims_padding,
                    )
                    - context_xyz
                )
                # Use point_intersection_interpolate_pose to get both the dynamic mask and
                # the per-point track index in a single shot (the bool mask is just
                # `tracks_idx != -1`). `source_timestamps_us[bidx]` has shape (V, H, W, 1);
                # the function reshapes points/timestamps to flat tensors internally so the
                # trailing-one dim is fine.
                tracks_idx = dynamic_track.point_intersection_interpolate_pose(
                    context_xyz,
                    source_timestamps_us[bidx],
                    self.cuboids_dims_padding,
                ).intersections_tracks_idx[..., 0]
                context_dynamic_mask_list.append((tracks_idx != -1).unsqueeze(-1))
            context_prev_flow = torch.stack(context_prev_flow_list, dim=0)
            context_next_flow = torch.stack(context_next_flow_list, dim=0)
            context_dynamic_mask = torch.stack(context_dynamic_mask_list, dim=0).squeeze(-1)

        if context_prev_flow is None or context_next_flow is None:
            raise RuntimeError(
                "No motion head found in the model, and cuboid tracks are not provided. Dynamic actors cannot be inferred."
            )

        # --- Full-resolution tensors needed for tokenization ---
        full_footprints = torch.stack([renderings[bidx].ray_footprints for bidx in range(B)], dim=0)  # (B, V, H, W, 1)
        full_sem_class = torch.argmax(context_semantic_logits, dim=-1)  # (B, V, H, W)

        # Compute full-res GT semantic if any path needs it.
        need_gt_semantic = self.config.use_gt_semantic_mask or (
            self.config.road_xyz_downsample_stride is not None and self.config.road_use_gt_semantic_mask
        )
        if need_gt_semantic:
            gt_semantic = torch.stack(
                [KelvinSemanticClass.get_target_from_frame_labels(data[bidx].labels) for bidx in range(B)],
                dim=0,
            )  # (B, V, H, W, 1) uint8
        else:
            gt_semantic = None

        # Full-res sky/ego opacity mask (matches existing semantics; downsampling happens in helper).
        if self.config.use_gt_semantic_mask:
            assert gt_semantic is not None
            full_opacity_mask = (
                (gt_semantic != KelvinSemanticClass.SKY) & (gt_semantic != KelvinSemanticClass.EGO)
            ).float()
        else:
            full_semantic_probs = torch.softmax(context_semantic_logits, dim=-1)
            full_opacity_mask = KelvinSemanticClass.opacity_mask_from_semantic_probs(full_semantic_probs)

        # --- Regular point-query tokens (uniform 2D grid) ---
        cs = self.config.grid_center_stride
        full = FullResInputs(
            xyz=full_xyz,
            prev_flow=context_prev_flow,
            next_flow=context_next_flow,
            dyn_mask=context_dynamic_mask,
            ts_src=source_timestamps_us,
            ts_prev=prev_target_timestamps_us,
            ts_next=next_target_timestamps_us,
            footprints=full_footprints,
            distance=gs_distance,
            opacity_mask=full_opacity_mask,
            ctx_rgb=context_rgb,
            sem_class=full_sem_class,
            world_normal=context_world_normal,
            ray_origins=full_rays[..., :3],
            ray_directions=full_rays[..., 3:],
        )
        if self.config.voxel_size_m is None:
            reg_tokens = build_grid_tokens(
                stride=self.config.xyz_downsample_stride,
                cs=cs,
                full=full,
            )
            self.last_voxel_dedup_diagnostics = None
        else:
            assert self.config.max_total_tokens is not None
            reg_tokens, self.last_voxel_dedup_diagnostics = build_voxel_dedup_tokens(
                stride=self.config.xyz_downsample_stride,
                grid_center_stride=cs,
                voxel_size_m=self.config.voxel_size_m,
                max_total_tokens=self.config.max_total_tokens,
                full=full,
            )
            dynamic_overflows = [
                diagnostic.dynamic_overflow_tokens
                for diagnostic in self.last_voxel_dedup_diagnostics
                if diagnostic.dynamic_overflow_tokens > 0
            ]
            if dynamic_overflows:
                logger.warning(
                    "Point-query dynamic coverage exceeded the %d-token static-aware budget by up to %d tokens; "
                    "retaining every dynamic token and emitting no static tokens for the affected sample(s).",
                    self.config.max_total_tokens,
                    max(dynamic_overflows),
                )
            logger.debug("Point-query voxel dedup diagnostics: %s", self.last_voxel_dedup_diagnostics)
            if media_logger is not None:
                diagnostic_fields = [
                    "raw_candidates",
                    "valid_static_candidates",
                    "valid_dynamic_candidates",
                    "occupied_static_voxels",
                    "static_tokens_before_cap",
                    "dynamic_tokens",
                    "dynamic_overflow_tokens",
                    "kept_static_tokens",
                    "kept_static_points",
                    "padded_tokens",
                    "static_overflow_tokens",
                ]
                for field_name in diagnostic_fields:
                    mean_value = sum(
                        getattr(diagnostic, field_name) for diagnostic in self.last_voxel_dedup_diagnostics
                    ) / len(self.last_voxel_dedup_diagnostics)
                    media_logger.log(f"gs_stats/dedup_{field_name}", mean_value)
                mean_static_retention_fraction = sum(
                    diagnostic.kept_static_points / max(1, diagnostic.valid_static_candidates)
                    for diagnostic in self.last_voxel_dedup_diagnostics
                ) / len(self.last_voxel_dedup_diagnostics)
                mean_static_overflow_fraction = sum(
                    diagnostic.static_overflow_tokens / max(1, diagnostic.static_tokens_before_cap)
                    for diagnostic in self.last_voxel_dedup_diagnostics
                ) / len(self.last_voxel_dedup_diagnostics)
                media_logger.log(
                    "gs_stats/dedup_static_retention_fraction",
                    mean_static_retention_fraction,
                )
                media_logger.log(
                    "gs_stats/dedup_static_overflow_fraction",
                    mean_static_overflow_fraction,
                )

                source_counts_by_camera: dict[str, int] = {}
                source_counts_by_time: dict[int, int] = {}
                for batch_idx, diagnostic in enumerate(self.last_voxel_dedup_diagnostics):
                    next_time_idx_by_sensor: dict[int, int] = {}
                    for frame_meta, count in zip(data[batch_idx].meta, diagnostic.kept_static_source_counts):
                        camera_name = frame_meta.logical_sensor_id or f"sensor_{frame_meta.unique_sensor_idx}"
                        source_counts_by_camera[camera_name] = source_counts_by_camera.get(camera_name, 0) + count
                        time_idx = next_time_idx_by_sensor.get(frame_meta.unique_sensor_idx, 0)
                        source_counts_by_time[time_idx] = source_counts_by_time.get(time_idx, 0) + count
                        next_time_idx_by_sensor[frame_meta.unique_sensor_idx] = time_idx + 1
                for camera_name, count in source_counts_by_camera.items():
                    media_logger.log(
                        f"gs_stats/dedup_kept_static_{camera_name}",
                        count / len(self.last_voxel_dedup_diagnostics),
                    )
                for time_idx, count in source_counts_by_time.items():
                    media_logger.log(
                        f"gs_stats/dedup_kept_static_time_{time_idx:02d}",
                        count / len(self.last_voxel_dedup_diagnostics),
                    )

        # --- Optional road point-query tokens (oversampled on ROAD pixels) ---
        road_tokens: GridTokens | None = None
        if self.config.road_xyz_downsample_stride is not None:
            if self.config.road_use_gt_semantic_mask:
                assert gt_semantic is not None
                road_pixel_mask = gt_semantic.squeeze(-1) == int(KelvinSemanticClass.ROAD)
            else:
                road_pixel_mask = full_sem_class == int(KelvinSemanticClass.ROAD)

            road_tokens = build_grid_tokens(
                stride=self.config.road_xyz_downsample_stride,
                cs=cs,
                full=full,
                keep_pixel_mask=road_pixel_mask,
                keep_token_threshold=self.config.road_token_threshold,
                max_tokens_per_view=self.config.road_max_tokens_per_view,
            )

        token_groups = [reg_tokens] + ([road_tokens] if road_tokens is not None else [])
        all_tokens = concat_grid_tokens(token_groups)
        gs_attrs = self._predict_point_query_gaussian_attributes(
            encoded_latent=encoded_latent,
            token_groups=token_groups,
            all_tokens=all_tokens,
            renderings=renderings,
            context_metric_distance=gs_distance,
            scene_rescale=scene_rescale,
            media_logger=media_logger,
        )

        # --- Predict & activate gaussian attributes ---
        gs_rgb = all_tokens.gs_ctx_rgb
        if isinstance(self.config, KelvinPointQueryDeformAttnDecoderConfig) and self.config.deform_predict_rgb_residual:
            gs_rgb_residual, gs_scale, gs_rotation, gs_opacity = gs_attrs.split([3, 3, 4, 1], dim=-1)
            gs_rgb = apply_bounded_rgb_logit_residual(
                gs_rgb,
                gs_rgb_residual,
                max_logit_delta=self.config.deform_rgb_residual_max_logit_delta,
            )
        else:
            gs_scale, gs_rotation, gs_opacity = gs_attrs.split([3, 4, 1], dim=-1)
        gs_scale = self.gaussian_activations.scale(
            gs_scale, scene_rescale=scene_rescale, pixel_scale=all_tokens.pixel_scale
        )
        gs_rotation = self.gaussian_activations.rotation(gs_rotation)
        gs_opacity = self.gaussian_activations.opacity(gs_opacity) * all_tokens.point_valid_mask.detach()

        gs_params = GaussianParams(
            rgb=gs_rgb,
            scale=gs_scale,
            rotation=gs_rotation,
            opacity=gs_opacity,
            xyz=all_tokens.gs_xyz,
            activated=True,
        )

        # --- Supervision packs (full-resolution flow for primitive_velocity loss) ---
        supervision_packs = [
            KelvinInstantNuRecSupervisionPack(
                context_rgb=context_rgb[bidx],
                context_depth=pred_depth[bidx],
                context_depth_conf=pred_depth_conf[bidx],
                context_semantic_logits=context_semantic_logits[bidx],
                context_world_normal=context_world_normal[bidx],
                motion_supervisions=motion_supervisions[bidx],
            )
            for bidx in range(B)
        ]

        # --- Build return (static + dynamic layers) ---
        return_values: list[KelvinDecoderReturn] = []
        for bidx in range(B):
            gs_bidx = gs_params[bidx].flatten()
            world_points = unpack_optional(gs_bidx.xyz)
            prev_world_points = world_points + all_tokens.gs_prev_flow[bidx]
            next_world_points = world_points + all_tokens.gs_next_flow[bidx]

            dynamic_mask_bool = all_tokens.gs_dyn_mask[bidx]
            if self.config.voxel_size_m is None:
                static_indices = torch.where(~dynamic_mask_bool)[0]
            else:
                # Fixed query padding is a compute-shape detail, not a Gaussian. Exclude
                # zero-validity rows from the returned static layer and point-count metrics.
                valid_point_mask = all_tokens.point_valid_mask[bidx].squeeze(-1) > 0.0
                static_indices = torch.where(~dynamic_mask_bool & valid_point_mask)[0]
            dynamic_indices = torch.where(dynamic_mask_bool)[0]

            semantic_class_static = all_tokens.gs_sem_class[bidx][static_indices].unsqueeze(-1).to(torch.uint8)
            normals_static = all_tokens.gs_normals[bidx][static_indices]
            static_layer = KelvinStaticLayer(
                positions=world_points[static_indices],
                rotations=gs_bidx.rotation[static_indices],
                scales=gs_bidx.scale[static_indices],
                densities=gs_bidx.opacity[static_indices],
                rgb=gs_bidx.rgb[static_indices],
                semantic_class=semantic_class_static,
                normals=normals_static,
            )
            dynamic_layer = KelvinDynamicLayer(
                keyframe_positions=torch.stack(
                    [
                        prev_world_points[dynamic_indices],
                        world_points[dynamic_indices],
                        next_world_points[dynamic_indices],
                    ],
                    dim=1,
                ),
                keyframe_timestamps_us=torch.stack(
                    [
                        all_tokens.gs_ts_prev[bidx][dynamic_indices],
                        all_tokens.gs_ts_src[bidx][dynamic_indices],
                        all_tokens.gs_ts_next[bidx][dynamic_indices],
                    ],
                    dim=1,
                ),
                max_densities=gs_bidx.opacity[dynamic_indices],
                rotations=gs_bidx.rotation[dynamic_indices],
                scales=gs_bidx.scale[dynamic_indices],
                rgb=gs_bidx.rgb[dynamic_indices],
            )
            # Match KelvinDPTDecoder: dynamic layers span a short time range, so cap min density for visibility.
            dynamic_layer = dynamic_layer.ensure_minimum_density(0.75)
            return_values.append(
                KelvinDecoderReturn(
                    static_layer=static_layer,
                    dynamic_layers=[dynamic_layer],
                    supervision_pack=supervision_packs[bidx],
                )
            )
        return return_values


class KelvinPointQueryDeformAttnDecoder(KelvinPointQueryCADecoder):
    """PointQuery variant with explicit projection and local deformable attention.

    Tokenization and all one-stage Kelvin context heads are inherited from the
    established PQ decoder. Attention is per retained Gaussian point rather
    than per packed 64-point token; packing remains only a bounded dedup/storage
    implementation detail.
    """

    def __init__(self, config: KelvinPointQueryDeformAttnDecoderConfig, model_config: KelvinModelConfig):
        if not isinstance(model_config.encoder, KelvinVGGTEncoderConfig):
            raise ValueError("KelvinPointQueryDeformAttnDecoder requires the VGGT encoder")
        super().__init__(config, model_config)
        self.config = config  # type: ignore[assignment]

        # Remove the global all-image KV path. The replacement samples a small
        # projected neighborhood, so cost grows linearly with views and points.
        del self.xyz_embed
        del self.ca_feature_norm
        del self.ca_kv_projector
        del self.ca_blocks
        del self.ca_kv_proj
        del self.gs_linear

        self.context_camera_names = tuple(model_config.encoder.camera_name_order or ())
        self.deform_attention = PointQueryDeformAttn(
            encoder_dim=model_config.encoder.embed_dim,
            fusion_dim=config.deform_feature_dim,
            num_sampling_points=config.deform_num_sampling_points,
            offset_radius_feature_px=config.deform_offset_radius_feature_px,
            depth_residual_scale_m=config.deform_depth_residual_scale_m,
            occlusion_abs_tolerance_m=config.deform_occlusion_abs_tolerance_m,
            occlusion_rel_tolerance=config.deform_occlusion_rel_tolerance,
            same_time_tolerance_us=config.deform_same_time_tolerance_us,
            view_fusion_mode=config.deform_view_fusion_mode,
            view_interaction_dim=config.deform_view_interaction_dim,
        )
        self.gs_linear = nn.Linear(config.deform_feature_dim, 3 + 4 + 1)
        self.rgb_residual_head: nn.Linear | None = None
        if config.deform_predict_rgb_residual:
            self.rgb_residual_head = nn.Linear(config.deform_feature_dim, 3)

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        # The VGGT checkpoint initializes the encoder separately. The Kelvin
        # DPT depth/context/motion heads and this new attention head are trained
        # jointly from scratch, matching the requested one-stage experiment.
        self.context_head.zero_init(init_values=self.context_head_init_values)
        self.deform_attention.apply(_tokengs_init_weights)
        torch.nn.init.zeros_(self.deform_attention.offset_head.weight)
        torch.nn.init.zeros_(self.deform_attention.offset_head.bias)
        torch.nn.init.zeros_(self.deform_attention.offset_weight_head.weight)
        torch.nn.init.zeros_(self.deform_attention.offset_weight_head.bias)
        final_view_score = self.deform_attention.view_score[-1]
        assert isinstance(final_view_score, nn.Linear)
        torch.nn.init.zeros_(final_view_score.weight)
        torch.nn.init.zeros_(final_view_score.bias)
        torch.nn.init.trunc_normal_(self.gs_linear.weight, std=0.002)
        torch.nn.init.zeros_(self.gs_linear.bias)
        if self.rgb_residual_head is not None:
            torch.nn.init.zeros_(self.rgb_residual_head.weight)
            torch.nn.init.zeros_(self.rgb_residual_head.bias)
        logger.info(
            "[KelvinPointQueryDeformAttnDecoder] Initialized new Kelvin and geometric-attention heads; "
            "checkpoint components=%s",
            sorted(loaded_state_dicts),
        )

    def _predict_point_query_gaussian_attributes(
        self,
        *,
        encoded_latent: KelvinMultiscaleFeaturesLatent,
        token_groups: list[GridTokens],
        all_tokens: GridTokens,
        renderings,
        context_metric_distance: torch.Tensor,
        scene_rescale: float,
        media_logger: BatchMediaLogger | None,
    ) -> torch.Tensor:
        del token_groups
        assert isinstance(self.config, KelvinPointQueryDeformAttnDecoderConfig)
        batch_size, num_views = context_metric_distance.shape[:2]
        query_valid = (all_tokens.point_valid_mask[..., 0] > 0.0) & torch.isfinite(all_tokens.gs_xyz).all(dim=-1)

        plucker = torch.cat(
            [
                torch.cross(
                    all_tokens.gs_ray_origins * scene_rescale,
                    all_tokens.gs_ray_directions,
                    dim=-1,
                ),
                all_tokens.gs_ray_directions,
            ],
            dim=-1,
        )
        query_descriptor = torch.cat(
            [all_tokens.gs_xyz * scene_rescale, all_tokens.gs_ctx_rgb, plucker],
            dim=-1,
        )

        if self.config.deform_checkpointing and torch.is_grad_enabled():
            context_features = torch.utils.checkpoint.checkpoint(
                self.deform_attention.project_context_features,
                encoded_latent.features[-1],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            context_features = self.deform_attention.project_context_features(encoded_latent.features[-1])

        references = project_points_to_context_views(
            all_tokens.gs_xyz,
            query_valid,
            [rendering.sensor_model_parameters for rendering in renderings],
            torch.stack([rendering.poses_tquat_startend for rendering in renderings], dim=0),
            torch.stack([rendering.timestamps_startend_us for rendering in renderings], dim=0),
        )
        height, width = context_metric_distance.shape[2:4]
        full_resolution = (
            context_metric_distance.new_tensor([width, height]).reshape(1, 1, 2).expand(batch_size, num_views, 2)
        )
        context_feature_atlas = build_context_image_atlas(context_features)
        context_distance_atlas = build_context_image_atlas(context_metric_distance)

        query_attributes: list[torch.Tensor] = []
        observation_counts: list[torch.Tensor] = []
        view_valid_counts: list[torch.Tensor] = []
        view_attention_masses: list[torch.Tensor] = []
        in_frustum_observation_counts: list[torch.Tensor] = []
        depth_tested_observation_counts: list[torch.Tensor] = []
        occluded_observation_counts: list[torch.Tensor] = []
        deformed_attention_masses: list[torch.Tensor] = []
        learned_offset_magnitude_sums: list[torch.Tensor] = []
        attention_entropies: list[torch.Tensor] = []
        chunk_size = self.config.deform_query_chunk_size
        for start in range(0, query_descriptor.shape[1], chunk_size):
            end = min(start + chunk_size, query_descriptor.shape[1])

            def fuse_and_predict_chunk(
                chunk_descriptor: torch.Tensor,
                chunk_timestamps: torch.Tensor,
                chunk_dynamic: torch.Tensor,
                chunk_valid: torch.Tensor,
                projected_context_atlas: torch.Tensor,
                metric_distance_atlas: torch.Tensor,
                image_points: torch.Tensor,
                reference_valid: torch.Tensor,
                reference_timestamps: torch.Tensor,
                reference_distance: torch.Tensor,
                resolution: torch.Tensor,
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                chunk_queries = self.deform_attention.embed_point_queries(chunk_descriptor)
                output = self.deform_attention.fuse_chunk(
                    chunk_queries,
                    chunk_timestamps,
                    chunk_dynamic,
                    chunk_valid,
                    projected_context_atlas,
                    metric_distance_atlas,
                    image_points,
                    reference_valid,
                    reference_timestamps,
                    reference_distance,
                    resolution,
                )
                chunk_attributes = self.gs_linear(output.features)
                if self.rgb_residual_head is not None:
                    chunk_attributes = torch.cat(
                        [self.rgb_residual_head(output.features), chunk_attributes],
                        dim=-1,
                    )
                return (
                    chunk_attributes,
                    output.valid_observation_count,
                    output.view_valid_query_count,
                    output.view_attention_mass,
                    output.in_frustum_observation_count,
                    output.depth_tested_observation_count,
                    output.occluded_observation_count,
                    output.deformed_sample_attention_mass,
                    output.learned_offset_magnitude_sum,
                    output.attention_entropy,
                )

            arguments = (
                query_descriptor[:, start:end],
                all_tokens.gs_ts_src[:, start:end],
                all_tokens.gs_dyn_mask[:, start:end],
                query_valid[:, start:end],
                context_feature_atlas,
                context_distance_atlas,
                references.image_points[:, :, start:end],
                references.valid_mask[:, :, start:end],
                references.timestamps_us[:, :, start:end],
                references.query_metric_distance[:, :, start:end],
                full_resolution,
            )
            if self.config.deform_checkpointing and torch.is_grad_enabled():
                chunk_outputs = torch.utils.checkpoint.checkpoint(
                    fuse_and_predict_chunk,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                chunk_outputs = fuse_and_predict_chunk(*arguments)
            (
                chunk_attributes,
                chunk_counts,
                chunk_view_valid_counts,
                chunk_view_attention_mass,
                chunk_in_frustum_observation_count,
                chunk_depth_tested_observation_count,
                chunk_occluded_observation_count,
                chunk_deformed_attention_mass,
                chunk_learned_offset_magnitude_sum,
                chunk_attention_entropy,
            ) = chunk_outputs
            query_attributes.append(chunk_attributes)
            observation_counts.append(chunk_counts)
            view_valid_counts.append(chunk_view_valid_counts)
            view_attention_masses.append(chunk_view_attention_mass)
            in_frustum_observation_counts.append(chunk_in_frustum_observation_count)
            depth_tested_observation_counts.append(chunk_depth_tested_observation_count)
            occluded_observation_counts.append(chunk_occluded_observation_count)
            deformed_attention_masses.append(chunk_deformed_attention_mass)
            learned_offset_magnitude_sums.append(chunk_learned_offset_magnitude_sum)
            attention_entropies.append(chunk_attention_entropy)

        counts = torch.cat(observation_counts, dim=1)
        all_query_attributes = torch.cat(query_attributes, dim=1)
        if media_logger is not None:
            valid_query_count = query_valid.sum().float().clamp_min(1.0)
            media_logger.log("point_deform/query_count", query_valid.sum(dim=1).float().mean())
            media_logger.log(
                "point_deform/observations_per_query",
                counts[query_valid].float().mean() if bool(query_valid.any()) else 0.0,
            )
            media_logger.log(
                "point_deform/no_observation_fraction",
                ((counts == 0) & query_valid).sum().float() / query_valid.sum().clamp_min(1),
            )
            in_frustum_count = torch.stack(in_frustum_observation_counts).sum().float()
            depth_tested_count = torch.stack(depth_tested_observation_counts).sum().float()
            occluded_count = torch.stack(occluded_observation_counts).sum().float()
            media_logger.log(
                "point_deform/in_frustum_observations_per_query",
                in_frustum_count / valid_query_count,
            )
            media_logger.log(
                "point_deform/depth_tested_fraction",
                depth_tested_count / in_frustum_count.clamp_min(1.0),
            )
            media_logger.log(
                "point_deform/occlusion_rejection_fraction",
                occluded_count / depth_tested_count.clamp_min(1.0),
            )
            media_logger.log(
                "point_deform/deformed_sample_attention_fraction",
                torch.stack(deformed_attention_masses).sum() / valid_query_count,
            )
            media_logger.log(
                "point_deform/learned_offset_magnitude_feature_px",
                torch.stack(learned_offset_magnitude_sums).sum() / valid_query_count,
            )
            if self.config.deform_view_fusion_mode == "depth-aware-interaction":
                attention_entropy = torch.cat(attention_entropies, dim=1)
                entropy_valid = query_valid & (counts > 0)
                media_logger.log(
                    "point_deform/attention_entropy",
                    attention_entropy[entropy_valid].mean() if bool(entropy_valid.any()) else 0.0,
                )
                media_logger.log(
                    "point_deform/effective_view_count",
                    attention_entropy[entropy_valid].exp().mean() if bool(entropy_valid.any()) else 0.0,
                )
                static_entropy_valid = entropy_valid & (~all_tokens.gs_dyn_mask)
                dynamic_entropy_valid = entropy_valid & all_tokens.gs_dyn_mask
                media_logger.log(
                    "point_deform/static_attention_entropy",
                    attention_entropy[static_entropy_valid].mean() if bool(static_entropy_valid.any()) else 0.0,
                )
                media_logger.log(
                    "point_deform/dynamic_attention_entropy",
                    attention_entropy[dynamic_entropy_valid].mean() if bool(dynamic_entropy_valid.any()) else 0.0,
                )
            if self.config.deform_predict_rgb_residual:
                media_logger.log(
                    "point_deform/rgb_residual_logit_abs_mean",
                    (
                        torch.tanh(all_query_attributes[..., :3]).abs()[query_valid].mean()
                        * self.config.deform_rgb_residual_max_logit_delta
                        if bool(query_valid.any())
                        else 0.0
                    ),
                )

            valid_by_view = torch.stack(view_valid_counts).sum(dim=0)
            attention_by_view = torch.stack(view_attention_masses).sum(dim=0)
            for view_idx in range(num_views):
                media_logger.log(
                    f"point_deform/view_{view_idx:02d}_valid_observations_per_query",
                    valid_by_view[:, view_idx].sum().float() / valid_query_count,
                )
                media_logger.log(
                    f"point_deform/view_{view_idx:02d}_attention_fraction",
                    attention_by_view[:, view_idx].sum() / valid_query_count,
                )

            if self.context_camera_names:
                num_cameras = len(self.context_camera_names)
                if num_views % num_cameras != 0:
                    raise ValueError(f"{num_views} context views are not divisible by {num_cameras} configured cameras")
                num_frames = num_views // num_cameras
                for camera_idx, camera_name in enumerate(self.context_camera_names):
                    view_slice = slice(camera_idx * num_frames, (camera_idx + 1) * num_frames)
                    media_logger.log(
                        f"point_deform/camera_{camera_name}_attention_fraction",
                        attention_by_view[:, view_slice].sum() / valid_query_count,
                    )
                for time_idx in range(num_frames):
                    view_indices = torch.arange(
                        time_idx,
                        num_views,
                        num_frames,
                        device=attention_by_view.device,
                    )
                    media_logger.log(
                        f"point_deform/time_{time_idx:02d}_attention_fraction",
                        attention_by_view[:, view_indices].sum() / valid_query_count,
                    )
        return all_query_attributes


def make_decoder(config: KelvinModelConfig) -> KelvinDecoderBase:
    if isinstance(config.decoder, KelvinTokenGSDecoderConfig):
        return KelvinTokenGSDecoder(config.decoder, config)
    elif isinstance(config.decoder, KelvinPointQueryDeformAttnDecoderConfig):
        return KelvinPointQueryDeformAttnDecoder(config.decoder, config)
    elif isinstance(config.decoder, KelvinPointQueryCADecoderConfig):
        return KelvinPointQueryCADecoder(config.decoder, config)
    elif isinstance(config.decoder, KelvinDPTDecoderConfig):
        return KelvinDPTDecoder(config.decoder, config)
    else:
        raise ValueError(f"Unsupported decoder config: {config.decoder}")
