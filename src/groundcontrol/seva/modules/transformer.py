import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from groundcontrol.seva.modules.rope_attn import CrossAttentionRope
from groundcontrol.training.rope import PositionGetter, RotaryPositionEmbedding2D


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input tensor.

    This implementation is based on the original Transformer paper "Attention is All You Need".

    Args:
        d_model (int): The dimensionality of the embedding vector.
        dropout (float): The dropout probability.
        max_len (int): The maximum possible length of the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000, scale: float = 1.0):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2)

        pe = torch.zeros(max_len, d_model)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension so it can be easily added to the input tensor
        # Shape becomes (1, max_len, d_model)
        pe = scale * pe.unsqueeze(0)

        # Register 'pe' as a buffer. Buffers are part of the model's state,
        # but they are not considered parameters to be trained.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: The output tensor with positional information added.
        """
        # Add the positional encoding to the input tensor.
        # self.pe is sliced to match the input sequence length.
        # x.shape[1] is the sequence length.
        return x + self.pe[:, : x.size(1)]


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForwardZeroInit(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the module.
        The weights of the final output linear layer are initialized to zero.
        This ensures that the output of the entire Attention block is zero at the
        beginning of training.
        """
        # The default initialization for to_q, to_k, to_v (Kaiming uniform) is good.
        # We only need to change the final output layer.
        nn.init.zeros_(self.net[-1].weight)
        # If the linear layer had a bias, we would also initialize it to zero.
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        output_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_mlp_at_output: bool = False,
        name: str | None = None,
        rope: RotaryPositionEmbedding2D | None = None,
        init_to_zero: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        if use_mlp_at_output:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, 2 * inner_dim),
                nn.GELU(),
                nn.Linear(2 * inner_dim, 2 * inner_dim),
                nn.GELU(),
                nn.Linear(2 * inner_dim, output_dim or query_dim),
                nn.Dropout(dropout),
            )
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, output_dim or query_dim), nn.Dropout(dropout)
            )
        if init_to_zero:
            self._initialize_weights()
        self.name = name

        self.rope = rope

    def _initialize_weights(self):
        """
        Initializes the weights of the module.
        The weights of the final output linear layer are initialized to zero.
        This ensures that the output of the entire Attention block is zero at the
        beginning of training.
        """
        # The default initialization for to_q, to_k, to_v (Kaiming uniform) is good.
        # We only need to change the final output layer.
        nn.init.zeros_(self.to_out[-2].weight)
        # If the linear layer had a bias, we would also initialize it to zero.
        if self.to_out[-2].bias is not None:
            nn.init.zeros_(self.to_out[-2].bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        debug_dict: dict[str, Any] | None = None,
        pos: torch.Tensor | None = None,
        context_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads),
            (q, k, v),
        )

        if self.rope is not None and pos is not None and context_pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, context_pos)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.to_out(out)
        if debug_dict is not None:
            # This should not run during training, its expensive!
            val_shape = v.shape
            eye = (
                torch.eye(val_shape[-2], device=v.device, dtype=v.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(val_shape[0], val_shape[1], -1, -1)
            )
            attn = F.scaled_dot_product_attention(q, k, eye)
            debug_dict[f"{self.name}/attn"] = attn.detach().cpu().float()
            # Detach and move to CPU to avoid keeping GPU memory references
            debug_dict[f"{self.name}/out"] = out.detach().cpu().float()
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x



class TransformerBlockSatRayAttnOnly(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        sat_cond_dim: int,
        sat_ca_dim: int,
        dropout: float = 0.0,
        name: str | None = None,
        use_mlp_for_sat_ca: bool = True,
        use_mlp_for_context: bool = False,
        use_context_norm: bool = False,
        use_mlp_at_output: bool = False,
        use_mlp_for_merging_query: bool = False,
        use_self_attention: bool = False,
        use_rope_sat_attn: bool = False,
        rope_freq: float = 100.0,
        use_view_embedding: bool = False,
        use_sat_output_norm: bool = False,
        use_sat_output_gate: bool = False,
        use_mlp_output_gate: bool = False,
        sat_attn_scale: float = 1.0,
        sat_mlp_scale: float = 1.0,
        scale_shift_predictor: bool = False,
        use_layernorm_after_mlps: bool = False,
        sat_attn_init_to_zero: bool = True,
        use_cond_sample_mask: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        if use_rope_sat_attn:
            sat_attn_class = CrossAttentionRope
        else:
            sat_attn_class = Attention
        self.use_view_embedding = use_view_embedding
        self.sat_attn = sat_attn_class(
            query_dim=sat_ca_dim,
            context_dim=sat_cond_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            name=name + "/sat_attn" if name else None,
            output_dim=dim,
            use_mlp_at_output=use_mlp_at_output,
            init_to_zero=sat_attn_init_to_zero,
        )
        if use_self_attention:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
            self.self_attn = Attention(
                query_dim=dim,
                context_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                name=name + "/self_attn" if name else None,
                output_dim=dim,
                init_to_zero=True,
                rope=self.rope,
            )
        else:
            self.self_attn = None
        if use_mlp_for_sat_ca:
            self.sat_ca_adapter_mlp = nn.Sequential(
                nn.Linear(dim, sat_ca_dim),
                nn.GELU(),
                nn.Linear(sat_ca_dim, sat_ca_dim),
            )
        else:
            self.sat_ca_adapter_mlp = None
        if use_context_norm:
            self.context_norm = nn.LayerNorm(sat_cond_dim)
        else:
            self.context_norm = None
        if use_mlp_for_context:
            self.context_adapter_mlp = nn.Sequential(
                nn.Linear(sat_cond_dim, 2 * sat_cond_dim),
                nn.GELU(),
                nn.Linear(2 * sat_cond_dim, sat_cond_dim),
            )
        else:
            self.context_adapter_mlp = None
        self.query_norm = nn.LayerNorm(sat_ca_dim)
        if use_mlp_for_merging_query:
            self.merged_query_mlp = nn.Sequential(
                nn.Linear(sat_ca_dim, sat_ca_dim),
                nn.GELU(),
                nn.Linear(sat_ca_dim, sat_ca_dim),
            )
        else:
            self.merged_query_mlp = None
        if use_sat_output_norm:
            self.sat_output_norm = nn.LayerNorm(dim)
        else:
            self.sat_output_norm = None

        if use_sat_output_gate:
            self.sat_alpha = nn.Parameter(torch.tensor(0.1))
        else:
            self.sat_alpha = 1.0
        if use_mlp_output_gate:
            self.mlp_alpha = nn.Parameter(torch.tensor(0.1))
        else:
            self.mlp_alpha = 1.0

        self.ff = FeedForwardZeroInit(dim, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.use_input_mask = False
        self.use_cond_sample_mask = use_cond_sample_mask
        self.sat_attn_scale = sat_attn_scale
        if scale_shift_predictor:
            self.scale_shift_predictor = nn.Sequential(
                nn.Linear(2 * dim, 3 * dim),
                nn.GELU(),
                nn.Linear(3 * dim, 3 * dim),
                nn.GELU(),
                nn.Linear(3 * dim, 2 * dim),
            )
        else:
            self.scale_shift_predictor = None
        if use_layernorm_after_mlps:
            self.sat_ca_adapter_norm = nn.LayerNorm(sat_ca_dim)
        else:
            self.sat_ca_adapter_norm = None

    def _initialize_weights(self):
        """
        Initializes the weights of the module.
        The weights of the final output linear layer are initialized to zero.
        This ensures that the output of the entire Attention block is zero at the
        beginning of training.
        """
        if self.scale_shift_predictor is not None:
            nn.init.zeros_(self.scale_shift_predictor[-1].weight)

    def preprocess_sat_cond(self, sat_cond, **condn_kwargs):
        sat_cond = rearrange(sat_cond, "b w z c -> (b w) z c")
        if self.context_adapter_mlp is not None:
            sat_cond = self.context_adapter_mlp(sat_cond)
        if self.context_norm is not None:
            sat_cond = self.context_norm(sat_cond)
        # sat_cond = self.pe(sat_cond)
        return sat_cond, None

    def preprocess_input(self, x_in, x_embed, **condn_kwargs):
        if self.use_view_embedding and x_embed is not None:
            if self.sat_ca_adapter_mlp is not None:
                x_in = self.sat_ca_adapter_mlp(x_in)
                if self.sat_ca_adapter_norm is not None:
                    x_in = self.sat_ca_adapter_norm(x_in)
                x_in = x_in + x_embed
                if self.merged_query_mlp is not None:
                    x_in = self.merged_query_mlp(x_in)
            else:
                x_in = x_in + x_embed
        else:
            if self.sat_ca_adapter_mlp is not None:
                x_in = self.sat_ca_adapter_mlp(x_in)
            else:
                x_in = x_in

        x_in = self.query_norm(x_in)
        return x_in

    def postprocess_attn3_output(self, x, x_out, w, **condn_kwargs):
        x_out = rearrange(x_out, "(b w) h c -> b (h w) c", w=w)
        return x_out

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        sat_cond: torch.Tensor,
        debug_dict: dict[str, Any] | None = None,
        pos: torch.Tensor | None = None,
        x_embed: torch.Tensor | None = None,
        input_mask: torch.Tensor | None = None,
        # sat_pos: torch.Tensor | None = None,
        **condn_kwargs,
    ) -> torch.Tensor:
        x_in = self.preprocess_input(x, x_embed=x_embed, **condn_kwargs)
        b, w, z, c = sat_cond.shape
        x_in = rearrange(x_in, "b (h w) c -> (b w) h c", w=w)
        sat_cond, _ = self.preprocess_sat_cond(sat_cond, **condn_kwargs)

        x_out = self.sat_attn(
            x_in,
            context=sat_cond,
            debug_dict=debug_dict,
        )
        # mask = sat_cond.abs().sum(dim=(-1, -2)) > 1e-3
        # x_out = torch.where(mask[..., None, None], x_out, torch.zeros_like(x_out))

        attn3_out = self.postprocess_attn3_output(x, x_out, w=w, **condn_kwargs)
        if self.sat_output_norm is not None:
            attn3_out = self.sat_output_norm(attn3_out)
        attn3_out = self.sat_alpha * attn3_out

        if self.self_attn is not None:
            attn3_out = self.self_attn(
                attn3_out, context=None, debug_dict=debug_dict, pos=pos, context_pos=pos
            )

        cond_sample_mask = condn_kwargs.get("cond_sample_mask", None)
        if cond_sample_mask is not None and self.use_cond_sample_mask:
            cond_sample_mask = cond_sample_mask.to(attn3_out)
            attn3_out = attn3_out * cond_sample_mask[:, None, None]

        if input_mask is not None and self.use_input_mask:
            attn3_out[input_mask] = 0.0

        attn3_out = self.sat_attn_scale * attn3_out

        if self.scale_shift_predictor is not None:
            scale_shift = self.scale_shift_predictor(torch.cat([x, attn3_out], dim=-1))
            scale, shift = scale_shift.split(x.shape[-1], dim=-1)
            # scale and shift are initialized to zero, so that change is zero at the beginning
            x = x * (1 + scale) + shift
        else:
            x = attn3_out + x

        ff_out = self.mlp_alpha * self.ff(self.ff_norm(x))
        if cond_sample_mask is not None and self.use_cond_sample_mask:
            ff_out = ff_out * cond_sample_mask[:, None, None]
        if input_mask is not None and self.use_input_mask:
            ff_out[input_mask] = 0.0

        x = ff_out + x
        return x



class SkipConnect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        return x_spatial + x_temporal


class MultiviewTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        name: str,
        unflatten_names: list[str] = [],
        depth: int = 1,
        context_dim: int = 1024,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.name = name
        self.unflatten_names = unflatten_names

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.time_mixer = SkipConnect()
        self.time_mix_blocks = nn.ModuleList(
            [
                TransformerBlockTimeMix(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, num_frames: int, **kwargs: Any
    ) -> torch.Tensor:
        assert context.ndim == 3
        _, _, h, w = x.shape
        x_in = x

        time_context = context
        time_context_first_timestep = time_context[::num_frames]
        time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)

        if self.name in self.unflatten_names:
            context = context[::num_frames]

        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj_in(x)

        for block, mix_block in zip(self.transformer_blocks, self.time_mix_blocks):
            if self.name in self.unflatten_names:
                x = rearrange(x, "(b t) (h w) c -> b (t h w) c", t=num_frames, h=h, w=w)
            x = block(x, context=context)
            if self.name in self.unflatten_names:
                x = rearrange(x, "b (t h w) c -> (b t) (h w) c", t=num_frames, h=h, w=w)
            x_mix = mix_block(x, context=time_context, num_frames=num_frames)
            x = self.time_mixer(x_spatial=x, x_temporal=x_mix)

        x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        out = x + x_in
        return out



class SatelliteMultiviewTransformerWithBlocks(MultiviewTransformer):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        name: str,
        unflatten_names: list[str] = [],
        depth: int = 1,
        context_dim: int = 1024,
        sat_cond_dim: int = 384,
        sat_query_dim: int = 1024,
        use_mlp_for_sat_ca: bool = True,
        use_mlp_for_context: bool = False,
        use_context_norm: bool = False,
        sat_ca_before_sa: bool = False,
        use_mlp_at_output: bool = False,
        use_mlp_for_merging_query: bool = False,
        dropout: float = 0.0,
        use_rope_attn3: bool = False,
        rope_freq: float = 100.0,
        skip_attn2: bool = False,
        residual_sat_block: bool = True,
        use_self_attention: bool = False,
        use_rope_sat_attn: bool = True,
        use_view_embedding: bool = True,
        use_height_cond: bool = False,
        height_cond_dim: int = 128,
        height_cond_before_sat_cond: bool = False,
        use_input_mask: bool = False,
        satellite_blocks_depth: int = 1,
        num_satellite_res: int = 1,
        use_sat_output_norm: bool = False,
        use_sat_output_gate: bool = False,
        use_mlp_output_gate: bool = False,
        scale_shift_predictor: bool = False,
        use_layernorm_after_mlps: bool = False,
        sat_attn_init_to_zero: bool = True,
        sat_attn_num_heads_factor: int = 1,
        sat_attn_d_head_factor: int = 1,
        sat_attn_num_heads: int | None = None,
        sat_attn_d_head: int | None = None,
        sat_attn_dropout: float = 0.0,
        use_cond_sample_mask: bool = False,
        **transformer_kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            n_heads=n_heads,
            d_head=d_head,
            name=name,
            unflatten_names=unflatten_names,
            depth=depth,
            context_dim=context_dim,
            dropout=dropout,
        )
        inner_dim = n_heads * d_head
        sat_transformer_block_cls = self.get_satellite_transformer_block()
        self.satellite_blocks = nn.ModuleList(
            [
                sat_transformer_block_cls(
                    dim=inner_dim,
                    n_heads= sat_attn_num_heads or n_heads // sat_attn_num_heads_factor,
                    d_head=sat_attn_d_head or d_head // sat_attn_d_head_factor,
                    context_dim=context_dim,
                    sat_cond_dim=sat_cond_dim,
                    sat_ca_dim=sat_query_dim,
                    use_mlp_for_sat_ca=use_mlp_for_sat_ca,
                    use_mlp_for_context=use_mlp_for_context,
                    use_context_norm=use_context_norm,
                    use_mlp_at_output=use_mlp_at_output,
                    use_mlp_for_merging_query=use_mlp_for_merging_query,
                    dropout=sat_attn_dropout,
                    name=name + f"/sat_block_{d}",
                    use_rope=use_rope_attn3,
                    rope_freq=rope_freq,
                    skip_attn2=skip_attn2,
                    use_self_attention=use_self_attention,
                    use_rope_sat_attn=use_rope_sat_attn,
                    use_view_embedding=use_view_embedding,
                    use_input_mask=use_input_mask,
                    use_sat_output_norm=use_sat_output_norm,
                    use_sat_output_gate=use_sat_output_gate,
                    use_mlp_output_gate=use_mlp_output_gate,
                    scale_shift_predictor=scale_shift_predictor,
                    use_layernorm_after_mlps=use_layernorm_after_mlps,
                    sat_attn_init_to_zero=sat_attn_init_to_zero,
                    use_cond_sample_mask=use_cond_sample_mask,
                    **transformer_kwargs,
                )
                for d in range(satellite_blocks_depth)
            ]
        )
        self.use_height_cond = use_height_cond
        if self.use_height_cond:
            self.height_cond_blocks = nn.ModuleList(
                [
                    sat_transformer_block_cls(
                        dim=inner_dim,
                        n_heads=n_heads,
                        d_head=d_head,
                        context_dim=context_dim,
                        sat_cond_dim=height_cond_dim,
                        sat_ca_dim=sat_query_dim,
                        use_mlp_for_sat_ca=use_mlp_for_sat_ca,
                        use_mlp_for_context=use_mlp_for_context,
                        use_context_norm=use_context_norm,
                        use_mlp_at_output=use_mlp_at_output,
                        use_mlp_for_merging_query=use_mlp_for_merging_query,
                        dropout=dropout,
                        name=name + f"/height_cond_block_{d}",
                        use_rope=use_rope_attn3,
                        rope_freq=rope_freq,
                        skip_attn2=skip_attn2,
                        use_self_attention=use_self_attention,
                        use_rope_sat_attn=use_rope_sat_attn,
                        use_view_embedding=use_view_embedding,
                        use_input_mask=use_input_mask,
                        use_cond_sample_mask=use_cond_sample_mask,
                        use_layernorm_after_mlps=use_layernorm_after_mlps,
                        sat_attn_init_to_zero=sat_attn_init_to_zero,
                        **transformer_kwargs,
                    )
                    for d in range(satellite_blocks_depth)
                ]
            )
        self.height_cond_before_sat_cond = height_cond_before_sat_cond

        self.residual_sat_block = residual_sat_block
        self.num_satellite_res = num_satellite_res
        if self.num_satellite_res > 1:
            assert (
                self.num_satellite_res == satellite_blocks_depth
            ), "num_satellite_res must be equal to satellite_blocks_depth"

    def get_satellite_transformer_block(self):
        return TransformerBlockSatRayAttnOnly

    def run_time_mix_block(self, block, x, context, sat_cond, num_frames, **condn_kwargs):
        return block(x, context=context, num_frames=num_frames, **condn_kwargs)

    def _run_satellite_block(
        self,
        sat_block_idx,
        x,
        context,
        sat_cond,
        height_cond,
        x_embed,
        debug_dict,
        input_mask,
        **condn_kwargs,
    ):
        if not self.height_cond_before_sat_cond:
            if sat_cond is not None:
                sat_cond_res = (
                    sat_cond if self.num_satellite_res == 1 else sat_cond[:, sat_block_idx]
                )
                # Everything inside sat_block is residual.
                x_sat = self.satellite_blocks[sat_block_idx](
                    x,
                    context=context,
                    sat_cond=sat_cond_res,
                    x_embed=x_embed,
                    debug_dict=debug_dict,
                    input_mask=input_mask,
                    **condn_kwargs,
                )
                if self.residual_sat_block:
                    x = x + x_sat
                else:
                    x = x_sat
            if self.use_height_cond:
                height_cond_res = (
                    height_cond if self.num_satellite_res == 1 else height_cond[:, sat_block_idx]
                )
                x = self.height_cond_blocks[sat_block_idx](
                    x,
                    context=context,
                    sat_cond=height_cond_res,
                    x_embed=x_embed,
                    debug_dict=debug_dict,
                    input_mask=input_mask,
                    **condn_kwargs,
                )
        else:
            if self.use_height_cond:
                height_cond_res = (
                    height_cond if self.num_satellite_res == 1 else height_cond[:, sat_block_idx]
                )
                x = self.height_cond_blocks[sat_block_idx](
                    x,
                    context=context,
                    sat_cond=height_cond_res,
                    x_embed=x_embed,
                    debug_dict=debug_dict,
                    input_mask=input_mask,
                    **condn_kwargs,
                )
            if sat_cond is not None:
                sat_cond_res = (
                    sat_cond if self.num_satellite_res == 1 else sat_cond[:, sat_block_idx]
                )
                x = self.satellite_blocks[sat_block_idx](
                    x,
                    context=context,
                    sat_cond=sat_cond_res,
                    x_embed=x_embed,
                    debug_dict=debug_dict,
                    input_mask=input_mask,
                    **condn_kwargs,
                )
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        num_frames: int,
        sat_cond: torch.Tensor | None,
        x_embed: torch.Tensor,
        height_cond: torch.Tensor | None = None,
        debug_dict: dict[str, Any] | None = None,
        input_mask: torch.Tensor | None = None,
        **condn_kwargs,
    ) -> torch.Tensor:
        assert context.ndim == 3
        _, _, h, w = x.shape
        x_in = x

        time_context = context
        time_context_first_timestep = time_context[::num_frames]
        time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)

        if self.name in self.unflatten_names:
            context = context[::num_frames]

        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj_in(x)

        for i, (block, mix_block) in enumerate(zip(self.transformer_blocks, self.time_mix_blocks)):
            if self.name in self.unflatten_names:
                x = rearrange(x, "(b t) (h w) c -> b (t h w) c", t=num_frames, h=h, w=w)
            x = block(x, context=context)
            if self.name in self.unflatten_names:
                x = rearrange(x, "b (t h w) c -> (b t) (h w) c", t=num_frames, h=h, w=w)
            x_mix = mix_block(x, context=time_context, num_frames=num_frames)
            x = self.time_mixer(x_spatial=x, x_temporal=x_mix)
            x = self._run_satellite_block(
                i,
                x,
                context,
                sat_cond,
                height_cond,
                x_embed,
                debug_dict,
                input_mask,
                **condn_kwargs,
            )

        for j in range(len(self.transformer_blocks), len(self.satellite_blocks)):
            x = self._run_satellite_block(
                j,
                x,
                context,
                sat_cond,
                height_cond,
                x_embed,
                debug_dict,
                input_mask,
                **condn_kwargs,
            )

        x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        out = x + x_in
        return out



