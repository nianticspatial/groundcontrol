from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from groundcontrol.seva.modules.layers import (
    Downsample,
    GroupNorm32,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
    timestep_embedding,
)
from groundcontrol.seva.modules.transformer import (
    MultiviewTransformer,
    SatelliteMultiviewTransformerWithBlocks,
)

UNetInputBlock = (
    ResBlock | MultiviewTransformer | SatelliteMultiviewTransformerWithBlocks | Downsample
)
UNetOutputBlock = ResBlock | MultiviewTransformer | Upsample


@dataclass
class SevaParams(object):
    in_channels: int = 11
    model_channels: int = 320
    out_channels: int = 4
    num_frames: int = 21
    num_res_blocks: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 2, 1])
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_head_channels: int = 64
    transformer_depth: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    context_dim: int = 1024
    dense_in_channels: int = 6
    dropout: float = 0.0
    unflatten_names: list[str] = field(
        default_factory=lambda: ["middle_ds8", "output_ds4", "output_ds2"]
    )

    def __post_init__(self):
        assert len(self.channel_mult) == len(self.transformer_depth)


class Seva(nn.Module):
    def __init__(self, params: SevaParams, **kwargs: Any) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
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

        mv_transformer_cls = self.get_mv_transformer_cls()
        output_mv_transformer_cls = self.get_output_mv_transformer_cls()
        bottleneck_transformer_cls = self.get_bottleneck_transformer_cls()
        additional_transformer_kwargs = self.additional_transformer_kwargs()
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
                        mv_transformer_cls(
                            in_channels=ch,
                            n_heads=num_heads,
                            d_head=dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                            **additional_transformer_kwargs,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            bottleneck_transformer_cls(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
                **additional_transformer_kwargs,
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

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[UNetOutputBlock] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels

                    output_layers.append(
                        output_mv_transformer_cls(
                            in_channels=ch,
                            n_heads=num_heads,
                            d_head=dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                            **additional_transformer_kwargs,
                        )
                    )
                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )

    def get_mv_transformer_cls(self):
        return MultiviewTransformer

    def get_output_mv_transformer_cls(self):
        return MultiviewTransformer

    def get_bottleneck_transformer_cls(self):
        return self.get_mv_transformer_cls()

    def additional_transformer_kwargs(self):
        return {}

    def forward(
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
        for module in self.input_blocks:
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
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
        h = h.type(x.dtype)
        return self.out(h)


class SGMWrapper(nn.Module):
    def __init__(self, module: Seva):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.module(
            x,
            t=t,
            y=c["crossattn"],
            dense_y=c["dense_vector"],
            **kwargs,
        )


class SevaEncoder(Seva):
    def __init__(self, params: SevaParams, run_decoder: bool = False, **kwargs: Any) -> None:
        self.decoder_blocks_with_txr = kwargs.pop(
            "decoder_blocks_with_txr", [3, 4, 5, 6, 7, 8, 9, 10, 11]
        )
        super().__init__(params, **kwargs)
        del self.out
        if not run_decoder:
            del self.output_blocks
        self.blocks_with_txr = [1, 2, 4, 5, 7, 8]
        self.run_decoder = run_decoder

    def forward(
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
        intermeds = []
        h = x
        for i, module in enumerate(self.input_blocks):
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
            if self.run_decoder:
                hs.append(h)
            if i in self.blocks_with_txr and not self.run_decoder:
                intermeds.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
        )
        if self.run_decoder:
            intermeds = [h]
            for i, module in enumerate(self.output_blocks):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(
                    h,
                    emb=t_emb,
                    context=y,
                    dense_emb=dense_y,
                    num_frames=num_frames,
                )
                if i in self.decoder_blocks_with_txr:
                    intermeds.append(h)
            return h, intermeds
        else:
            intermeds.append(h)
            return h, intermeds
