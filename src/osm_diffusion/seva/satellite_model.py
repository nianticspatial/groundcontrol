import functools
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from osm_diffusion.seva.model import Seva, SevaParams, UNetInputBlock
from osm_diffusion.seva.modules.layers import (
    Downsample,
    DownsampleResidual,
    ResBlock,
    TimestepEmbedSequential,
    timestep_embedding,
)
from osm_diffusion.seva.modules.transformer import (
    MultiviewTransformer,
    PositionalEncoding,
    SatelliteMultiviewTransformer,
    SatelliteMultiviewTransformerFusedQueryRayAttn,
    SatelliteMultiviewTransformerRayAttn,
    SatelliteMultiviewTransformerRayAttnLearnedEmbed,
    SatelliteMultiviewTransformerTimeMix,
    SatelliteMultiviewTransformerWithBlocks,
    SatelliteMultiviewTransformerWithBlocksNoRay,
)


class SevaSat(Seva):
    def __init__(
        self, params: SevaParams, skip_clip_attn: bool = False, sat_cond_dim: int = 384
    ) -> None:
        # This needs to be set before calling super().__init__()
        self.skip_clip_attn = skip_clip_attn
        self.sat_cond_dim = sat_cond_dim
        super().__init__(params)

    def get_mv_transformer_cls(self):
        return SatelliteMultiviewTransformer

    def get_output_mv_transformer_cls(self):
        return SatelliteMultiviewTransformer

    def additional_transformer_kwargs(self):
        return {"skip_attn2": self.skip_clip_attn, "sat_cond_dim": self.sat_cond_dim}

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        sat_cond: torch.Tensor | None = None,
        num_frames: int | None = None,
        debug_dict: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                sat_cond=sat_cond,
                debug_dict=debug_dict,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            sat_cond=sat_cond,
            debug_dict=debug_dict,
        )
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                sat_cond=sat_cond,
                debug_dict=debug_dict,
            )
        h = h.type(x.dtype)
        return self.out(h)


class SevaSatTimeMix(SevaSat):
    def get_mv_transformer_cls(self):
        return SatelliteMultiviewTransformerTimeMix

    def get_output_mv_transformer_cls(self):
        return SatelliteMultiviewTransformerTimeMix


class SevaSatInputOnly(SevaSat):
    def get_output_mv_transformer_cls(self):
        assert not self.skip_clip_attn, "skip_clip_attn is not supported for input only"
        return MultiviewTransformer


class SevaSatInputOnlyRope(SevaSatInputOnly):
    def get_mv_transformer_cls(self):
        return functools.partial(SatelliteMultiviewTransformer, use_rope_attn3=True)


class SevaSatRope(SevaSat):
    def get_mv_transformer_cls(self):
        return functools.partial(SatelliteMultiviewTransformer, use_rope_attn3=True)

    def get_output_mv_transformer_cls(self):
        return functools.partial(SatelliteMultiviewTransformer, use_rope_attn3=True)


class SevaSatRayAttn(SevaSat):
    def __init__(
        self, params: SevaParams, skip_clip_attn: bool = False, sat_cond_dim: int = 384
    ) -> None:
        # This needs to be set before calling super().__init__()
        super().__init__(params, skip_clip_attn=skip_clip_attn, sat_cond_dim=sat_cond_dim)

        self.sat_downsample_layers = nn.ModuleDict(
            {
                "2": DownsampleResidual(sat_cond_dim),
                "4": DownsampleResidual(sat_cond_dim),
                "8": DownsampleResidual(sat_cond_dim),
            }
        )

    def get_mv_transformer_cls(self):
        return SatelliteMultiviewTransformerRayAttn

    def get_output_mv_transformer_cls(self):
        return SatelliteMultiviewTransformerRayAttn

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        sat_cond: torch.Tensor | None = None,
        num_frames: int | None = None,
        debug_dict: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # sat_cond shape is (b * n, z, c, h, w)
        sat_bn, sat_z, sat_c, sat_h, sat_w = sat_cond.shape
        sat_cond = sat_cond.reshape(-1, sat_c, sat_h, sat_w)

        sat_cond_for_res = {}
        sat_cond_for_res["1"] = rearrange(
            sat_cond, "(b n z) c h w -> (b n) (h w) z c", n=num_frames, z=sat_z
        )
        for k in ["2", "4", "8"]:
            sat_cond = self.sat_downsample_layers[k](sat_cond)
            sat_cond_for_res[k] = rearrange(
                sat_cond, "(b n z) c h w -> (b n) (h w) z c", n=num_frames, z=sat_z
            )

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                sat_cond=sat_cond_for_res,
                debug_dict=debug_dict,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            sat_cond=sat_cond_for_res,
            debug_dict=debug_dict,
        )
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                sat_cond=sat_cond_for_res,
                debug_dict=debug_dict,
            )
        h = h.type(x.dtype)
        return self.out(h)


class SevaSatRayAttnInputOnly(SevaSatRayAttn):
    def get_output_mv_transformer_cls(self):
        return MultiviewTransformer


class SevaSatRayAttnLearnedQueryInputOnly(SevaSatRayAttn):
    def __init__(
        self,
        params: SevaParams,
        skip_clip_attn: bool = False,
        sat_cond_dim: int = 384,
        latent_dim: int = 32,
        embedding_dim: int = 320,
    ) -> None:
        self.embedding_dim = embedding_dim
        super().__init__(params, skip_clip_attn, sat_cond_dim)

        self.latent_dim = latent_dim
        self.query_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(
                    (self.latent_dim * self.latent_dim) // (i * i), self.embedding_dim
                )
                for i in [1, 2, 4, 8]
            }
        )
        with torch.no_grad():
            for i in [1, 2, 4, 8]:
                self.query_embeds[str(i)].weight.data.uniform_(-1, 1)

        self.block_id_to_res = {
            1: 1,
            2: 1,
            4: 2,
            5: 2,
            7: 4,
            8: 4,
        }

    def get_mv_transformer_cls(self):
        return functools.partial(
            SatelliteMultiviewTransformerRayAttnLearnedEmbed, embed_dim=self.embedding_dim
        )

    def get_output_mv_transformer_cls(self):
        return MultiviewTransformer

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        sat_cond: torch.Tensor | None = None,
        num_frames: int | None = None,
        debug_dict: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # sat_cond shape is (b * n, z, c, h, w)
        sat_bn, sat_z, sat_c, sat_h, sat_w = sat_cond.shape
        sat_cond = sat_cond.reshape(-1, sat_c, sat_h, sat_w)

        sat_cond_for_res = {}
        sat_cond_for_res["1"] = rearrange(
            sat_cond, "(b n z) c h w -> (b n) (h w) z c", n=num_frames, z=sat_z
        )
        for k in ["2", "4", "8"]:
            sat_cond = self.sat_downsample_layers[k](sat_cond)
            sat_cond_for_res[k] = rearrange(
                sat_cond, "(b n z) c h w -> (b n) (h w) z c", n=num_frames, z=sat_z
            )

        hs = []
        h = x
        for i, module in enumerate(self.input_blocks):
            if i in self.block_id_to_res:
                query_embed_res = self.block_id_to_res[i]
                query_embed = self.query_embeds[str(query_embed_res)].weight
                query_embed = query_embed[None].repeat(h.shape[0], 1, 1)
                added_kwargs = {"x_embed": query_embed}
            else:
                added_kwargs = {}
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                sat_cond=sat_cond_for_res,
                debug_dict=debug_dict,
                **added_kwargs,
            )
            hs.append(h)
        query_embed = self.query_embeds["8"].weight
        query_embed = query_embed[None].repeat(h.shape[0], 1, 1)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            sat_cond=sat_cond_for_res,
            debug_dict=debug_dict,
            x_embed=query_embed,
        )
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
        return self.out(h)


class SevaSatRayAttnFusedQueryInputOnly(SevaSatRayAttn):
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
        *args,
        **kwargs,
    ) -> None:
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
        super().__init__(params, skip_clip_attn, sat_cond_dim)
        del self.sat_downsample_layers

        self.block_id_to_res = {
            1: 1,
            2: 1,
            4: 2,
            5: 2,
            7: 4,
            8: 4,
        }

        self.use_view_embedding = getattr(self, "use_view_embedding", True)
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

        self.use_mlp_sat_pos = use_mlp_sat_pos
        if self.use_mlp_sat_pos:
            self.sat_pos_mlp = nn.Sequential(
                nn.Linear(sat_cond_dim, 2 * sat_cond_dim),
                nn.GELU(),
                nn.Linear(2 * sat_cond_dim, sat_cond_dim),
            )
            self.sat_embed_mlp = nn.Sequential(
                nn.Linear(sat_cond_dim, 2 * sat_cond_dim),
                nn.GELU(),
                nn.Linear(2 * sat_cond_dim, sat_cond_dim),
            )
            self.merged_sat_mlp = nn.Sequential(
                nn.Linear(sat_cond_dim, 2 * sat_cond_dim),
                nn.GELU(),
                nn.Linear(2 * sat_cond_dim, sat_cond_dim),
            )
            self.satellite_pos_encoder = PositionalEncoding(d_model=sat_cond_dim, max_len=64)
        else:
            self.sat_pos_mlp = None
            self.sat_embed_mlp = None

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
        self.use_register_sat_ray_embeddings = use_register_sat_ray_embeddings
        if self.use_register_sat_ray_embeddings:
            self.sat_ray_embeddings = nn.Parameter(torch.randn(1, 1, 1, sat_cond_dim) * 0.1)
        if self.use_layernorm_after_mlps and self.use_view_embedding:
            self.ray_embed_norm = nn.LayerNorm(self.embedding_dim)
            self.t_embed_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.ray_embed_norm = None
            self.t_embed_norm = None

    def get_mv_transformer_cls(self):
        return functools.partial(
            SatelliteMultiviewTransformerFusedQueryRayAttn,
            sat_query_dim=self.embedding_dim,
            use_mlp_for_sat_ca=self.use_mlp_for_sat_ca,
            use_mlp_for_context=self.use_mlp_for_context,
            sat_ca_before_sa=self.sat_ca_before_sa,
            use_mlp_at_output=self.use_mlp_at_output,
            use_mlp_for_merging_query=self.use_mlp_for_merging_query,
            sat_attn_init_to_zero=self.sat_attn_init_to_zero,
        )

    def get_output_mv_transformer_cls(self):
        return MultiviewTransformer

    @staticmethod
    def _tensor_to_debug_numpy(tensor: torch.Tensor) -> np.ndarray:
        tensor_cpu = tensor.detach().cpu()
        if tensor_cpu.dtype == torch.bfloat16:
            tensor_cpu = tensor_cpu.float()
        return tensor_cpu.numpy()

    def _maybe_dump_pre_embedding_inputs(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        sat_cond: dict[int, torch.Tensor] | torch.Tensor,
        street_raymap_for_res: dict[int, torch.Tensor],
        street_raymap_raw_for_res: dict[int, torch.Tensor] | None,
        num_frames: int,
        height_cond: dict[int, torch.Tensor] | None,
        disable_height_cond: bool,
        disable_sat_cond: bool,
        input_mask: torch.Tensor | None,
        dump_step_idx: int | None,
    ) -> None:
        if not getattr(self, "debug_dump_model_inputs_pre_embed", False):
            return
        dump_dir = getattr(self, "debug_dump_model_inputs_dir", None)
        if not dump_dir:
            return
        target_step_indices = getattr(self, "debug_dump_model_inputs_step_indices", None)
        if target_step_indices is not None and dump_step_idx not in target_step_indices:
            return

        max_dumps = int(getattr(self, "debug_dump_model_inputs_max_dumps", 1))
        dump_count = int(getattr(self, "_debug_dump_model_inputs_count", 0))
        if dump_count >= max_dumps:
            return

        dump_root = Path(dump_dir)
        dump_root.mkdir(parents=True, exist_ok=True)
        setattr(self, "_debug_dump_model_inputs_count", dump_count + 1)

        arrays: dict[str, np.ndarray] = {
            "x": self._tensor_to_debug_numpy(x),
            "t": self._tensor_to_debug_numpy(t),
            "y": self._tensor_to_debug_numpy(y),
            "dense_y": self._tensor_to_debug_numpy(dense_y),
            "num_frames": np.array([num_frames], dtype=np.int32),
            "disable_height_cond": np.array([disable_height_cond], dtype=np.bool_),
            "disable_sat_cond": np.array([disable_sat_cond], dtype=np.bool_),
            "dump_step_idx": np.array([dump_step_idx if dump_step_idx is not None else -1], dtype=np.int32),
        }

        if input_mask is not None:
            arrays["input_mask"] = self._tensor_to_debug_numpy(input_mask)

        if isinstance(sat_cond, dict):
            for key, value in sorted(sat_cond.items(), key=lambda kv: str(kv[0])):
                arrays[f"sat_cond_res_{key}"] = self._tensor_to_debug_numpy(value)
        else:
            arrays["sat_cond"] = self._tensor_to_debug_numpy(sat_cond)

        for key, value in sorted(street_raymap_for_res.items(), key=lambda kv: str(kv[0])):
            arrays[f"street_raymap_res_{key}"] = self._tensor_to_debug_numpy(value)
        if street_raymap_raw_for_res is not None:
            for key, value in sorted(street_raymap_raw_for_res.items(), key=lambda kv: str(kv[0])):
                arrays[f"street_raymap_raw_res_{key}"] = self._tensor_to_debug_numpy(value)

        if height_cond is not None:
            for key, value in sorted(height_cond.items(), key=lambda kv: str(kv[0])):
                arrays[f"height_cond_res_{key}"] = self._tensor_to_debug_numpy(value)

        timestep = int(t.flatten()[0].detach().cpu().item()) if t.numel() > 0 else -1
        output_path = dump_root / (
            f"pre_embed_inputs_{self.__class__.__name__}_{dump_count:04d}_t{timestep:04d}.npz"
        )
        np.savez_compressed(output_path, **arrays)

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
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        self._maybe_dump_pre_embedding_inputs(
            x=x,
            t=t,
            y=y,
            dense_y=dense_y,
            sat_cond=sat_cond,
            street_raymap_for_res=street_raymap_for_res,
            street_raymap_raw_for_res=street_raymap_raw_for_res,
            num_frames=num_frames,
            height_cond=height_cond,
            disable_height_cond=disable_height_cond,
            disable_sat_cond=disable_sat_cond,
            input_mask=input_mask,
            dump_step_idx=dump_step_idx,
        )
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        t_emb_ca = None
        if self.use_view_embedding:
            t_emb_ca = self.t_embed_mlp(t_emb)  # bt, c
            if self.t_embed_norm is not None:
                t_emb_ca = self.t_embed_norm(t_emb_ca)

        if self.use_mlp_sat_pos and not disable_sat_cond:
            sat_seq_len = next(iter(sat_cond.values())).shape[-2]
            posenc = self.sat_pos_mlp(self.satellite_pos_encoder.pe[:, :sat_seq_len])
            for k, val in sat_cond.items():
                sat_cond[k] = self.merged_sat_mlp(self.sat_embed_mlp(val) + posenc)
            if self.use_register_sat_ray_embeddings:
                sat_cond = torch.cat(
                    [
                        sat_cond,
                        self.sat_ray_embeddings.repeat(sat_cond.shape[0], sat_cond.shape[1], 1, 1),
                    ],
                    dim=-1,
                )

        # sat_cond shape is (b * n, h, z, c)
        hs = []
        h = x
        intermeds = []
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
                    # height_cond_res = self.height_encoder(height_cond_res)
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


class SevaSatRayAttnFusedQuerySeparateBlock(SevaSatRayAttnFusedQueryInputOnly):
    def __init__(self, *args, **kwargs):
        self.residual_sat_block = kwargs.pop("residual_sat_block", True)
        self.sat_block_use_self_attention = kwargs.pop("sat_block_use_self_attention", False)
        self.use_rope_sat_attn = kwargs.pop("use_rope_sat_attn", True)
        self.use_view_embedding = kwargs.pop("use_view_embedding", True)
        self.satellite_blocks_depth = kwargs.pop("satellite_blocks_depth", 1)
        self.height_cond_before_sat_cond = kwargs.pop("height_cond_before_sat_cond", False)
        self.num_satellite_res = kwargs.pop("num_satellite_res", 1)
        self.use_sat_output_norm = kwargs.pop("use_sat_output_norm", False)
        self.use_sat_output_gate = kwargs.pop("use_sat_output_gate", False)
        self.use_mlp_output_gate = kwargs.pop("use_mlp_output_gate", False)
        self.scale_shift_predictor = kwargs.pop("scale_shift_predictor", False)
        super().__init__(*args, **kwargs)

    def get_mv_transformer_cls(self):
        # return MultiviewTransformer
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


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SevaSatRayAttnFusedQuerySeparateBlockControlNet(SevaSatRayAttnFusedQuerySeparateBlock):
    def __init__(self, params: SevaParams, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)

        self.input_blocks_frozen = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        time_embed_dim = params.model_channels * 4

        self.zero_convs = nn.ModuleList([self.make_zero_conv(params.model_channels)])

        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[UNetInputBlock] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            in_channels=ch,
                            n_heads=num_heads,
                            d_head=dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                self.input_blocks_frozen.append(TimestepEmbedSequential(*input_layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks_frozen.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch
                self.zero_convs.append(self.make_zero_conv(ch))

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block_frozen = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch
        self.middle_block_output = self.make_zero_conv(ch)

    def make_zero_conv(self, channels: int) -> TimestepEmbedSequential:
        return TimestepEmbedSequential(zero_module(conv_nd(2, channels, channels, 1, padding=0)))

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
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        self._maybe_dump_pre_embedding_inputs(
            x=x,
            t=t,
            y=y,
            dense_y=dense_y,
            sat_cond=sat_cond,
            street_raymap_for_res=street_raymap_for_res,
            street_raymap_raw_for_res=street_raymap_raw_for_res,
            num_frames=num_frames,
            height_cond=height_cond,
            disable_height_cond=disable_height_cond,
            disable_sat_cond=disable_sat_cond,
            input_mask=input_mask,
            dump_step_idx=dump_step_idx,
        )
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # RUN the control net
        t_emb_ca = self.t_embed_mlp(t_emb) if self.use_view_embedding else None  # bt, c

        if self.use_mlp_sat_pos and not disable_sat_cond:
            sat_seq_len = next(iter(sat_cond.values())).shape[-2]
            posenc = self.sat_pos_mlp(self.satellite_pos_encoder.pe[:, :sat_seq_len])
            for k, val in sat_cond.items():
                sat_cond[k] = self.merged_sat_mlp(self.sat_embed_mlp(val) + posenc)

        # sat_cond shape is (b * n, h, z, c)
        hs_control_net = []
        h = x
        for i, module in enumerate(self.input_blocks):
            if i in self.block_id_to_res and not disable_sat_cond:
                sat_cond_res = (
                    sat_cond[self.block_id_to_res[i]] if isinstance(sat_cond, dict) else sat_cond
                )
                if self.use_view_embedding:
                    street_raymap = street_raymap_for_res[self.block_id_to_res[i]]
                    street_raymap = self.ray_map_mlp(street_raymap)
                    x_embed = t_emb_ca[:, None, :] + street_raymap
                else:
                    x_embed = None
                if self.use_height_cond and not disable_height_cond:
                    height_cond_res = height_cond[self.block_id_to_res[i]]
                    # height_cond_res = self.height_encoder(height_cond_res)
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
            h = self.zero_convs[i](
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
            hs_control_net.append(h)

        sat_cond_res = sat_cond[8] if isinstance(sat_cond, dict) else sat_cond
        if self.use_view_embedding:
            street_raymap = street_raymap_for_res[8]
            street_raymap = self.ray_map_mlp(street_raymap)
            x_embed = t_emb_ca[:, None, :] + street_raymap
        else:
            x_embed = None
        if self.use_height_cond and not disable_height_cond:
            height_cond_res = height_cond[8]
            # height_cond_res = self.height_encoder(height_cond_res)

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
        h = self.middle_block_output(
            h, emb=t_emb, context=y, dense_emb=dense_y, num_frames=num_frames
        )
        hs_control_net.append(h)
        for i, h in enumerate(hs_control_net):
            print(
                f"{i} {t.cpu().numpy()[0]} hs_control_net shape: {h.shape}, norm: {h.norm()}, min: {h.min()}, max: {h.max()}"
            )
            print(
                f"{i} {t.cpu().numpy()[0]} hs_control_net shape: {h.shape}, norm: {h.norm()}, min: {h.min()}, max: {h.max()}"
            )
            h += torch.randn_like(h)

        # RUN the frozen encoder
        with torch.no_grad():
            hs_frozen = []
            h = x
            for module in self.input_blocks_frozen:
                h = module(
                    h,
                    emb=t_emb,
                    context=y,
                    dense_emb=dense_y,
                    num_frames=num_frames,
                )
                hs_frozen.append(h)
            h = self.middle_block_frozen(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
        h += hs_control_net.pop()

        intermeds = [h]
        for i, module in enumerate(self.output_blocks):
            control_inp = hs_control_net.pop() + hs_frozen.pop()
            h = torch.cat([h, control_inp], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                debug_dict=debug_dict,
            )
            intermeds.append(control_inp)
        h = h.type(x.dtype)
        return self.out(h), intermeds

    def base_encoder_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
        debug_dict: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks_frozen:
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
        )
        hs.append(h)
        hs.reverse()
        return hs


class SevaSatRayAttnFusedQuerySeparateBlockNoRay(SevaSatRayAttnFusedQueryInputOnly):
    def __init__(self, *args, **kwargs):
        self.residual_sat_block = kwargs.pop("residual_sat_block", True)
        self.sat_block_use_self_attention = kwargs.pop("sat_block_use_self_attention", False)
        self.use_rope_sat_attn = kwargs.pop("use_rope_sat_attn", True)
        self.use_view_embedding = kwargs.pop("use_view_embedding", True)
        self.satellite_blocks_depth = kwargs.pop("satellite_blocks_depth", 1)
        self.height_cond_before_sat_cond = kwargs.pop("height_cond_before_sat_cond", False)
        self.num_satellite_res = kwargs.pop("num_satellite_res", 1)
        print(
            "residual_sat_block:",
            self.residual_sat_block,
            "satellite_blocks_depth:",
            self.satellite_blocks_depth,
            "height_cond_before_sat_cond:",
            self.height_cond_before_sat_cond,
            "num_satellite_res:",
            self.num_satellite_res,
        )
        super().__init__(*args, **kwargs)

    def get_mv_transformer_cls(self):
        # return MultiviewTransformer
        return functools.partial(
            SatelliteMultiviewTransformerWithBlocksNoRay,
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
        )


class SevaSatRayAttnFusedQuerySeparateBlockInpOut(SevaSatRayAttnFusedQueryInputOnly):
    def __init__(self, *args, **kwargs):
        self.down_block_id_to_res = {
            3: 4,
            4: 4,
            5: 4,
            6: 2,
            7: 2,
            8: 2,
            9: 1,
            10: 1,
            11: 1,
        }
        self.residual_sat_block = kwargs.pop("residual_sat_block", True)
        self.sat_block_in_encoder = kwargs.pop("sat_block_in_encoder", True)
        self.sat_block_in_decoder = kwargs.pop("sat_block_in_decoder", True)
        self.sat_block_use_self_attention = kwargs.pop("sat_block_use_self_attention", False)
        self.use_rope_sat_attn = kwargs.pop("use_rope_sat_attn", True)
        self.use_view_embedding = kwargs.pop("use_view_embedding", True)
        self.satellite_blocks_depth = kwargs.pop("satellite_blocks_depth", 1)
        self.height_cond_before_sat_cond = kwargs.pop("height_cond_before_sat_cond", False)
        self.num_satellite_res = kwargs.pop("num_satellite_res", 1)
        self.use_sat_output_norm = kwargs.pop("use_sat_output_norm", False)
        self.use_sat_output_gate = kwargs.pop("use_sat_output_gate", False)
        self.use_mlp_output_gate = kwargs.pop("use_mlp_output_gate", False)
        self.scale_shift_predictor = kwargs.pop("scale_shift_predictor", False)
        self.sat_attn_num_heads_factor = kwargs.pop("sat_attn_num_heads_factor", 1)
        self.sat_attn_d_head_factor = kwargs.pop("sat_attn_d_head_factor", 1)
        self.sat_attn_num_heads = kwargs.pop("sat_attn_num_heads", None)
        self.sat_attn_d_head = kwargs.pop("sat_attn_d_head", None)
        self.sat_attn_dropout = kwargs.pop("sat_attn_dropout", 0.0)
        print(
            "residual_sat_block:",
            self.residual_sat_block,
            "satellite_blocks_depth:",
            self.satellite_blocks_depth,
        )
        super().__init__(*args, **kwargs)

    def get_sat_transformer(self):
        return functools.partial(
            SatelliteMultiviewTransformerWithBlocks,
            sat_query_dim=self.embedding_dim,
            use_mlp_for_sat_ca=self.use_mlp_for_sat_ca,
            use_mlp_for_context=self.use_mlp_for_context,
            sat_ca_before_sa=self.sat_ca_before_sa,
            use_context_norm=self.use_context_norm,
            use_mlp_at_output=self.use_mlp_at_output,
            use_mlp_for_merging_query=self.use_mlp_for_merging_query,
            height_cond_before_sat_cond=self.height_cond_before_sat_cond,
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
            sat_attn_num_heads_factor=self.sat_attn_num_heads_factor,
            sat_attn_d_head_factor=self.sat_attn_d_head_factor,
            sat_attn_num_heads=self.sat_attn_num_heads,
            sat_attn_d_head=self.sat_attn_d_head,
            sat_attn_dropout=self.sat_attn_dropout,
        )

    def get_mv_transformer_cls(self):
        if self.sat_block_in_encoder:
            return self.get_sat_transformer()
        else:
            return MultiviewTransformer

    def get_output_mv_transformer_cls(self):
        if self.sat_block_in_decoder:
            return self.get_sat_transformer()
        else:
            return MultiviewTransformer

    def get_bottleneck_transformer_cls(self):
        if self.sat_block_in_encoder or self.sat_block_in_decoder:
            return self.get_sat_transformer()
        else:
            return MultiviewTransformer

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        sat_cond: dict[int, torch.Tensor],
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
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        num_frames = num_frames or self.params.num_frames
        self._maybe_dump_pre_embedding_inputs(
            x=x,
            t=t,
            y=y,
            dense_y=dense_y,
            sat_cond=sat_cond,
            street_raymap_for_res=street_raymap_for_res,
            street_raymap_raw_for_res=street_raymap_raw_for_res,
            num_frames=num_frames,
            height_cond=height_cond,
            disable_height_cond=disable_height_cond,
            disable_sat_cond=disable_sat_cond,
            input_mask=input_mask,
            dump_step_idx=dump_step_idx,
        )
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        t_emb_ca = self.t_embed_mlp(t_emb) if self.use_view_embedding else None  # bt, c

        sat_seq_len = next(iter(sat_cond.values())).shape[-2]
        if self.use_mlp_sat_pos and not disable_sat_cond:
            posenc = self.sat_pos_mlp(self.satellite_pos_encoder.pe[:, :sat_seq_len])
            for k, val in sat_cond.items():
                sat_cond[k] = self.merged_sat_mlp(self.sat_embed_mlp(val) + posenc)

        # sat_cond shape is (b * n, z, c, h, w)
        hs = []
        h = x
        intermeds = []
        for i, module in enumerate(self.input_blocks):
            if i in self.block_id_to_res and not disable_sat_cond:
                sat_cond_res = sat_cond[self.block_id_to_res[i]]
                if self.use_view_embedding:
                    street_raymap = street_raymap_for_res[self.block_id_to_res[i]]
                    street_raymap = self.ray_map_mlp(street_raymap)
                    x_embed = t_emb_ca[:, None, :] + street_raymap
                else:
                    x_embed = None
            else:
                sat_cond_res = None
                x_embed = None
            if height_cond is not None and not disable_height_cond:
                height_cond_res = height_cond[self.block_id_to_res[i]]
                # height_cond_res = self.height_encoder(height_cond_res)
            else:
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
                input_mask=input_mask if self.use_input_mask else None,
                cond_sample_mask=cond_sample_mask,
            )
            hs.append(h)
            if self.sat_block_in_encoder and i in self.block_id_to_res:
                intermeds.append(h)
            # module_intermeds_dict[f"input_block_{i}"] = embeddings

        if not disable_sat_cond:
            sat_cond_res = sat_cond[8]
            if self.use_view_embedding:
                street_raymap = street_raymap_for_res[8]
                street_raymap = self.ray_map_mlp(street_raymap)
                x_embed = t_emb_ca[:, None, :] + street_raymap
            else:
                x_embed = None
        else:
            sat_cond_res = None
            x_embed = None
        if height_cond is not None and not disable_height_cond:
            height_cond_res = height_cond[8]
            # height_cond_res = self.height_encoder(height_cond_res)
        else:
            height_cond_res = None

        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            sat_cond=sat_cond_res,
            debug_dict=debug_dict,
            x_embed=x_embed,
            input_mask=input_mask if self.use_input_mask else None,
            height_cond=height_cond_res if self.use_height_cond else None,
            cond_sample_mask=cond_sample_mask,
        )
        intermeds.append(h)
        # module_intermeds_dict["middle_block"] = embeddings
        for i, module in enumerate(self.output_blocks):
            if i in self.down_block_id_to_res and not disable_sat_cond:
                sat_cond_res = sat_cond[self.down_block_id_to_res[i]]
                if self.use_view_embedding:
                    street_raymap = street_raymap_for_res[self.down_block_id_to_res[i]]
                    street_raymap = self.ray_map_mlp(street_raymap)
                    x_embed = t_emb_ca[:, None, :] + street_raymap
                else:
                    x_embed = None
            else:
                sat_cond_res = None
                x_embed = None
            if height_cond is not None and not disable_height_cond:
                height_cond_res = height_cond[self.down_block_id_to_res[i]]
                # height_cond_res = self.height_encoder(height_cond_res)
            else:
                height_cond_res = None
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                debug_dict=debug_dict,
                x_embed=x_embed,
                sat_cond=sat_cond_res,
                height_cond=height_cond_res if self.use_height_cond else None,
                input_mask=input_mask if self.use_input_mask else None,
                cond_sample_mask=cond_sample_mask,
            )
            if self.sat_block_in_decoder and i in self.down_block_id_to_res:
                intermeds.append(h)
        h = h.type(x.dtype)
        return self.out(h), intermeds


class SevaSatT2IAdapter(Seva):
    def __init__(
        self, params: SevaParams, skip_clip_attn: bool = False, sat_cond_dim: int = 384
    ) -> None:
        # This needs to be set before calling super().__init__()
        self.skip_clip_attn = skip_clip_attn
        self.sat_cond_dim = sat_cond_dim
        super().__init__(params)

        # First blocks is a pure resblock
        is_input_block_adapted = [False]

        for level, mult in enumerate(params.channel_mult):
            for i in range(params.num_res_blocks):
                if i == params.num_res_blocks - 1:
                    is_input_block_adapted.append(True)
                else:
                    is_input_block_adapted.append(False)
            if level != len(params.channel_mult) - 1:
                is_input_block_adapted.append(False)

        self.is_input_block_adapted = is_input_block_adapted

    def additional_transformer_kwargs(self):
        return {"skip_attn2": self.skip_clip_attn, "sat_cond_dim": self.sat_cond_dim}

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        sat_cond: list[torch.Tensor],
        num_frames: int | None = None,
        debug_dict: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        sat_cond_iter = iter(sat_cond)
        for i, module in enumerate(self.input_blocks):
            additional_residual = next(sat_cond_iter) if self.is_input_block_adapted[i] else None
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                debug_dict=debug_dict,
                additional_residual=additional_residual,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            debug_dict=debug_dict,
        )
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
        return self.out(h)


class SevaSatControlNext(SevaSatT2IAdapter):
    def __init__(
        self, params: SevaParams, skip_clip_attn: bool = False, sat_cond_dim: int = 384
    ) -> None:
        # This needs to be set before calling super().__init__()
        self.skip_clip_attn = skip_clip_attn
        self.sat_cond_dim = sat_cond_dim
        super().__init__(params)

        # First blocks is a pure resblock
        is_input_block_adapted = [True]

        for level, mult in enumerate(params.channel_mult):
            for i in range(params.num_res_blocks):
                is_input_block_adapted.append(False)
            if level != len(params.channel_mult) - 1:
                is_input_block_adapted.append(False)

        self.is_input_block_adapted = is_input_block_adapted

    def additional_transformer_kwargs(self):
        return {"skip_attn2": self.skip_clip_attn, "sat_cond_dim": self.sat_cond_dim}
        return {"skip_attn2": self.skip_clip_attn, "sat_cond_dim": self.sat_cond_dim}
