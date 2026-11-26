"""Satellite ray-guided cross-attention SEVA model used by this release."""

import functools
from typing import Any

import torch
import torch.nn as nn

from osm_diffusion.seva.model import Seva, SevaParams
from osm_diffusion.seva.modules.layers import timestep_embedding
from osm_diffusion.seva.modules.transformer import (
    MultiviewTransformer,
    SatelliteMultiviewTransformerWithBlocks,
)


class SatelliteRayGuidedCA(Seva):
    """UNet with per-resolution satellite ray-guided cross-attention blocks."""

    def __init__(
        self,
        params: SevaParams,
        skip_clip_attn: bool = False,
        sat_cond_dim: int = 384,
        embedding_dim: int = 320,
        use_mlp_for_sat_ca: bool = True,
        use_mlp_for_context: bool = False,
        use_context_norm: bool = False,
        sat_ca_before_sa: bool = False,
        use_mlp_at_output: bool = False,
        use_mlp_for_merging_query: bool = False,
        street_raymap_dim: int = 3,
        use_mlp_sat_pos: bool = False,
        use_height_cond: bool = False,
        num_height_cond_bins: int = 32,
        height_cond_dim: int = 384,
        use_input_mask: bool = False,
        use_cond_sample_mask: bool = False,
        use_register_sat_ray_embeddings: bool = False,
        use_layernorm_after_mlps: bool = False,
        sat_attn_init_to_zero: bool = True,
        residual_sat_block: bool = True,
        sat_block_use_self_attention: bool = False,
        use_rope_sat_attn: bool = True,
        use_view_embedding: bool = True,
        satellite_blocks_depth: int = 1,
        height_cond_before_sat_cond: bool = False,
        num_satellite_res: int = 1,
        use_sat_output_norm: bool = False,
        use_sat_output_gate: bool = False,
        use_mlp_output_gate: bool = False,
        scale_shift_predictor: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.skip_clip_attn = skip_clip_attn
        self.sat_cond_dim = sat_cond_dim
        self.embedding_dim = embedding_dim
        self.use_mlp_for_sat_ca = use_mlp_for_sat_ca
        self.use_mlp_for_context = use_mlp_for_context
        self.use_context_norm = use_context_norm
        self.sat_ca_before_sa = sat_ca_before_sa
        self.use_mlp_at_output = use_mlp_at_output
        self.use_mlp_for_merging_query = use_mlp_for_merging_query
        self.use_height_cond = use_height_cond
        self.height_cond_dim = height_cond_dim
        self.use_input_mask = use_input_mask
        self.use_cond_sample_mask = use_cond_sample_mask
        self.use_layernorm_after_mlps = use_layernorm_after_mlps
        self.sat_attn_init_to_zero = sat_attn_init_to_zero
        self.residual_sat_block = residual_sat_block
        self.sat_block_use_self_attention = sat_block_use_self_attention
        self.use_rope_sat_attn = use_rope_sat_attn
        self.use_view_embedding = use_view_embedding
        self.satellite_blocks_depth = satellite_blocks_depth
        self.height_cond_before_sat_cond = height_cond_before_sat_cond
        self.num_satellite_res = num_satellite_res
        self.use_sat_output_norm = use_sat_output_norm
        self.use_sat_output_gate = use_sat_output_gate
        self.use_mlp_output_gate = use_mlp_output_gate
        self.scale_shift_predictor = scale_shift_predictor

        super().__init__(params)

        self.block_id_to_res = {
            1: 1,
            2: 1,
            4: 2,
            5: 2,
            7: 4,
            8: 4,
        }

        time_embed_dim = self.model_channels * 4
        if self.use_view_embedding:
            self.t_embed_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, self.embedding_dim),
                nn.SiLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            self.ray_map_mlp = nn.Sequential(
                nn.Linear(street_raymap_dim, self.embedding_dim),
                nn.SiLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
        else:
            self.t_embed_mlp = None
            self.ray_map_mlp = None

        # Kept for checkpoint compatibility; unused by the released Metropolis config.
        self.use_mlp_sat_pos = use_mlp_sat_pos
        self.sat_pos_mlp = None
        self.sat_embed_mlp = None
        self.merged_sat_mlp = None
        self.satellite_pos_encoder = None
        self.use_register_sat_ray_embeddings = use_register_sat_ray_embeddings
        self.sat_ray_embeddings = None
        if self.use_layernorm_after_mlps and self.use_view_embedding:
            self.ray_embed_norm = nn.LayerNorm(self.embedding_dim)
            self.t_embed_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.ray_embed_norm = None
            self.t_embed_norm = None

        if use_height_cond:
            self.height_backbone = nn.Sequential(
                nn.Conv2d(num_height_cond_bins, height_cond_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(height_cond_dim, height_cond_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(height_cond_dim, height_cond_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(height_cond_dim, height_cond_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(height_cond_dim, height_cond_dim, kernel_size=1, padding=0),
            )
            self.height_encoder = nn.Sequential(
                nn.Linear(height_cond_dim, 2 * height_cond_dim),
                nn.GELU(),
                nn.Linear(2 * height_cond_dim, 2 * height_cond_dim),
                nn.GELU(),
                nn.Linear(2 * height_cond_dim, height_cond_dim),
            )

    def get_mv_transformer_cls(self):
        return functools.partial(
            SatelliteMultiviewTransformerWithBlocks,
            sat_query_dim=self.embedding_dim,
            use_mlp_for_sat_ca=self.use_mlp_for_sat_ca,
            use_mlp_for_context=self.use_mlp_for_context,
            use_context_norm=self.use_context_norm,
            height_cond_before_sat_cond=self.height_cond_before_sat_cond,
            sat_ca_before_sa=self.sat_ca_before_sa,
            use_mlp_at_output=self.use_mlp_at_output,
            use_mlp_for_merging_query=self.use_mlp_for_merging_query,
            residual_sat_block=self.residual_sat_block,
            use_self_attention=self.sat_block_use_self_attention,
            use_rope_sat_attn=self.use_rope_sat_attn,
            use_view_embedding=self.use_view_embedding,
            use_height_cond=self.use_height_cond,
            height_cond_dim=self.height_cond_dim,
            use_input_mask=self.use_input_mask,
            use_cond_sample_mask=self.use_cond_sample_mask,
            satellite_blocks_depth=self.satellite_blocks_depth,
            num_satellite_res=self.num_satellite_res,
            use_sat_output_norm=self.use_sat_output_norm,
            use_sat_output_gate=self.use_sat_output_gate,
            use_mlp_output_gate=self.use_mlp_output_gate,
            scale_shift_predictor=self.scale_shift_predictor,
            use_layernorm_after_mlps=self.use_layernorm_after_mlps,
            sat_attn_init_to_zero=self.sat_attn_init_to_zero,
        )

    def get_output_mv_transformer_cls(self):
        return MultiviewTransformer

    def additional_transformer_kwargs(self):
        return {"skip_attn2": self.skip_clip_attn, "sat_cond_dim": self.sat_cond_dim}

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        sat_cond: dict[int, torch.Tensor] | torch.Tensor,
        street_raymap_for_res: dict[int, torch.Tensor],
        street_raymap_raw_for_res: dict[int, torch.Tensor] | None = None,
        num_frames: int | None = None,
        debug_dict: dict[str, Any] | None = None,
        height_cond: dict[int, torch.Tensor] | None = None,
        disable_height_cond: bool = False,
        disable_sat_cond: bool = False,
        input_mask: torch.Tensor | None = None,
        cond_sample_mask: torch.Tensor | None = None,
        dump_step_idx: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        t_emb_ca = None
        if self.use_view_embedding:
            t_emb_ca = self.t_embed_mlp(t_emb)
            if self.t_embed_norm is not None:
                t_emb_ca = self.t_embed_norm(t_emb_ca)

        hs = []
        h = x
        intermeds = []
        height_cond_res = None
        for i, module in enumerate(self.input_blocks):
            if i in self.block_id_to_res and not disable_sat_cond:
                sat_cond_res = (
                    sat_cond[self.block_id_to_res[i]] if isinstance(sat_cond, dict) else sat_cond
                )
                if self.use_view_embedding:
                    street_raymap = street_raymap_for_res[self.block_id_to_res[i]]
                    street_raymap = self.ray_map_mlp(street_raymap)
                    if self.ray_embed_norm is not None:
                        street_raymap = self.ray_embed_norm(street_raymap)
                    x_embed = t_emb_ca[:, None, :] + street_raymap
                else:
                    x_embed = None
                if self.use_height_cond and not disable_height_cond:
                    height_cond_res = height_cond[self.block_id_to_res[i]]
            else:
                sat_cond_res = None
                x_embed = None
                height_cond_res = None
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                sat_cond=sat_cond_res,
                debug_dict=debug_dict,
                x_embed=x_embed,
                height_cond=height_cond_res if self.use_height_cond else None,
                input_mask=input_mask if self.use_input_mask else None,
                cond_sample_mask=cond_sample_mask,
            )
            if i in self.block_id_to_res:
                intermeds.append(h)
            hs.append(h)

        sat_cond_res = sat_cond[8] if isinstance(sat_cond, dict) else sat_cond
        if self.use_view_embedding:
            street_raymap = street_raymap_for_res[8]
            street_raymap = self.ray_map_mlp(street_raymap)
            if self.ray_embed_norm is not None:
                street_raymap = self.ray_embed_norm(street_raymap)
            x_embed = t_emb_ca[:, None, :] + street_raymap
        else:
            x_embed = None
        if self.use_height_cond and not disable_height_cond:
            height_cond_res = height_cond[8]

        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            sat_cond=sat_cond_res,
            debug_dict=debug_dict,
            x_embed=x_embed,
            height_cond=height_cond_res if self.use_height_cond else None,
            input_mask=input_mask if self.use_input_mask else None,
            cond_sample_mask=cond_sample_mask,
        )
        intermeds.append(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                debug_dict=debug_dict,
            )
        h = h.type(x.dtype)
        return self.out(h), intermeds
