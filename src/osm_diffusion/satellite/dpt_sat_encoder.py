import math
from copy import deepcopy
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from osm_diffusion.satellite.dpt_head import DPTHead


class SatelliteDPTEncoder(nn.Module):
    """
    Wraps a ViT-based satellite encoder with a DPT head to produce dense
    multi-scale feature maps that can be consumed by the downstream pipeline.
    """

    def __init__(self, backbone: nn.Module, dpt_head_config: dict | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        head_cfg = deepcopy(dpt_head_config) if dpt_head_config is not None else {}

        in_channels: Sequence[int] | None = head_cfg.get("in_channels")
        if in_channels is None:
            embed_dim = getattr(backbone, "embed_dim", None)
            if embed_dim is None:
                raise ValueError(
                    "Backbone must expose `embed_dim` when DPT `in_channels` are not provided."
                )
            in_channels = [embed_dim] * head_cfg.get("num_dpt_blocks", 4)
            head_cfg["in_channels"] = in_channels

        self.num_dpt_blocks = len(in_channels)

        self.dpt_head = DPTHead(**head_cfg)
        self._dpt_output_dim = self.dpt_head.n_output_channels

    @property
    def output_dim(self) -> int:
        return self._dpt_output_dim

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self.forward_features(x)

    def forward_features(self, x: Tensor) -> dict[str, Tensor]:
        dpt_inputs = self._collect_intermediate_inputs(x)
        feature_map = self.dpt_head(dpt_inputs)
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        return {
            "x_norm_patchtokens": tokens,
            "dpt_feature_map": feature_map,
        }

    def get_dpt_head_parameters(self) -> Iterable[nn.Parameter]:
        return self.dpt_head.parameters()

    def set_backbone_trainable(self, state: bool) -> None:
        self.backbone.requires_grad_(state)
        self.backbone.train(state)

    def set_dpt_head_trainable(self, state: bool) -> None:
        self.dpt_head.requires_grad_(state)
        self.dpt_head.train(state)

    def _collect_intermediate_inputs(self, x: Tensor) -> List[Tuple[Tensor, Tensor]]:
        get_layers = getattr(self.backbone, "get_intermediate_layers", None)
        if get_layers is None:
            raise AttributeError("Backbone must implement `get_intermediate_layers` for DPT.")

        try:
            outputs = get_layers(
                x,
                n=self.num_dpt_blocks,
                return_class_token=True,
                reshape=True,
            )
        except TypeError:
            outputs = get_layers(x, n=self.num_dpt_blocks, return_class_token=True)

        processed: List[Tuple[Tensor, Tensor]] = []
        for out in outputs:
            if isinstance(out, tuple):
                # (patch_tokens, cls_token) or variants
                patch_tokens, cls_token = out[0], out[-1]
            else:
                patch_tokens = out
                cls_token = out[:, :1]

            patch_tokens = self._ensure_spatial(patch_tokens)
            cls_token = cls_token.squeeze(1) if cls_token.ndim == 3 else cls_token
            processed.append((patch_tokens, cls_token))
        return processed

    def _ensure_spatial(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 4:
            return tokens
        if tokens.ndim != 3:
            raise ValueError(f"Unexpected tensor rank {tokens.ndim} for ViT tokens.")

        bsz, seq_len, dim = tokens.shape
        spatial_tokens = tokens
        h = int(math.sqrt(seq_len))
        if h * h != seq_len:
            if (seq_len - 1) <= 0:
                raise ValueError("Cannot infer spatial resolution from tokens.")
            seq_wo_cls = seq_len - 1
            h = int(math.sqrt(seq_wo_cls))
            if h * h != seq_wo_cls:
                raise ValueError("Token count does not form a square grid.")
            spatial_tokens = tokens[:, 1:, :]
        spatial_tokens = spatial_tokens.transpose(1, 2).reshape(bsz, dim, h, h)
        return spatial_tokens


