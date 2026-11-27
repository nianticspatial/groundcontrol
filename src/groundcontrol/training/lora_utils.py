import math
from collections.abc import Callable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper for nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA branch: x @ A @ B, where A: in_features->rank and B: rank->out_features.
        self.lora_A = nn.Parameter(
            torch.empty(
                base.in_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                rank,
                base.out_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze base weights; only LoRA params are trainable in this wrapper.
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return base_out + self.scaling * lora_out


def inject_lora_into_linears(
    module: nn.Module,
    *,
    should_wrap: Callable[[str], bool],
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
) -> int:
    """Recursively replace selected nn.Linear layers with LoRALinear wrappers."""
    replaced = 0
    for child_name, child in module.named_children():
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and should_wrap(full_name):
            setattr(
                module,
                child_name,
                LoRALinear(base=child, rank=rank, alpha=alpha, dropout=dropout),
            )
            replaced += 1
        else:
            replaced += inject_lora_into_linears(
                child,
                should_wrap=should_wrap,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                prefix=full_name,
            )
    return replaced


def is_lora_state_key(key: str) -> bool:
    return key.endswith(".lora_A") or key.endswith(".lora_B")


def extract_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if is_lora_state_key(k)}
