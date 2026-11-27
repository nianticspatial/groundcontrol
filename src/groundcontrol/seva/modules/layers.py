import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from groundcontrol.seva.modules.transformer import (
    MultiviewTransformer,
    SatelliteMultiviewTransformerWithBlocks,
)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> torch.Tensor:
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class Upsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = nn.Conv2d(self.channels, self.out_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class DownsampleResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.op = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.meanpool = nn.AvgPool2d(2, stride=2)
        with torch.no_grad():
            nn.init.normal_(self.op.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.op.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        res = self.meanpool(x)
        res = self.op(x) + res
        return res


class GroupNorm32(nn.GroupNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.float()).type(input.dtype)


class TimestepEmbedSequential(nn.Sequential):
    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor,
        dense_emb: torch.Tensor,
        num_frames: int,
        sat_cond: torch.Tensor | dict[str, torch.Tensor] | None = None,
        debug_dict: dict[str, Any] | None = None,
        x_embed: torch.Tensor | None = None,
        additional_residual: torch.Tensor | None = None,
        height_cond: torch.Tensor | None = None,
        input_mask: torch.Tensor | None = None,
        cond_sample_mask: torch.Tensor | None = None,
        use_cross_normalization: bool = False,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, SatelliteMultiviewTransformerWithBlocks):
                x = layer(
                    x,
                    context,
                    num_frames,
                    sat_cond,
                    debug_dict=debug_dict,
                    x_embed=x_embed,
                    height_cond=height_cond,
                    input_mask=input_mask,
                    cond_sample_mask=cond_sample_mask,
                )
            elif isinstance(layer, MultiviewTransformer):
                assert num_frames is not None
                x = layer(x, context, num_frames)
            elif isinstance(layer, ResBlock):
                x = layer(x, emb, dense_emb)
            else:
                x = layer(x)
        if additional_residual is not None:
            if use_cross_normalization:
                mean_latents, std_latents = torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.std(
                    x, dim=(1, 2, 3), keepdim=True
                )
                mean_control, std_control = torch.mean(
                    additional_residual, dim=(1, 2, 3), keepdim=True
                ), torch.std(additional_residual, dim=(1, 2, 3), keepdim=True)
                additional_residual = (additional_residual - mean_control) * (
                    std_latents / (std_control + 1e-12)
                ) + mean_latents

            x = x + additional_residual
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int | None,
        dense_in_channels: int,
        dropout: float,
    ):
        super().__init__()
        out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, 1, 1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels))
        self.dense_emb_layers = nn.Sequential(nn.Conv2d(dense_in_channels, 2 * channels, 1, 1, 0))
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, dense_emb: torch.Tensor) -> torch.Tensor:
        in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
        h = in_rest(x)
        dense = self.dense_emb_layers(
            F.interpolate(dense_emb, size=h.shape[2:], mode="bilinear", align_corners=True)
        ).type(h.dtype)
        dense_scale, dense_shift = torch.chunk(dense, 2, dim=1)
        h = h * (1 + dense_scale) + dense_shift
        h = in_conv(h)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h


# def FusionBlock(nn.Module):
#     def __init__(
#         self,
#         channels: int,
#         emb_channels: int,
#         out_channels: int | None,
#         dense_in_channels: int,
#         dropout: float,
#     ):
#         super().__init__()
#         out_channels = out_channels or channels

#         self.in_layers = nn.Sequential(
#             GroupNorm32(32, channels),
#             nn.SiLU(),
#             nn.Conv2d(channels, out_channels, 3, 1, 1),
#         )
#         self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels))
#         self.dense_emb_layers = nn.Sequential(nn.Conv2d(dense_in_channels, 2 * channels, 1, 1, 0))
#         self.out_layers = nn.Sequential(
#             GroupNorm32(32, out_channels),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#         )
#         if out_channels == channels:
#             self.skip_connection = nn.Identity()
#         else:
#             self.skip_connection = nn.Conv2d(channels, out_channels, 1, 1, 0)

#     def forward(self, x: torch.Tensor, cond: torch.Tensor, emb: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
#         in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
#         h = in_rest(x)
#         dense = self.dense_emb_layers(x).type(h.dtype)
#         dense_scale, dense_shift = torch.chunk(dense, 2, dim=1)
#         h = h * (1 + dense_scale) + dense_shift
#         h = in_conv(h)
#         emb_out = self.emb_layers(emb).type(h.dtype)
#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]
#         h = h + emb_out
#         h = self.out_layers(h)
#         h = self.skip_connection(x) + h
#         return h
