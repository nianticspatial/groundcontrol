from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

# --- Helper Functions (Adapted from Llama) ---


def precompute_freqs_cis_1d(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the (cos + i*sin) frequency values for 1D RoPE.

    Args:
        dim (int): The dimension of the head (must be even).
        end (int): The maximum sequence length.
        theta (float): The base for frequency calculation.

    Returns:
        torch.Tensor: A complex tensor of shape (end, dim / 2)
    """
    # Calculate the frequencies (theta_i)
    # 1.0 / (theta^(2i / dim))
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Calculate the positions (m)
    t = torch.arange(end, dtype=torch.float32)

    # Calculate the angles (m * theta_i)
    # shape: (end, dim / 2)
    angles = torch.outer(t, inv_freq)

    # Convert to complex numbers (cos(angle) + i*sin(angle))
    # This is the (cos + i*sin) tensor
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the complex frequency tensor to be broadcastable with the input tensor.

    Args:
        freqs_cis (torch.Tensor): Shape (seq_len, dim / 2)
        x (torch.Tensor): Shape (batch, num_heads, seq_len, dim / 2)

    Returns:
        torch.Tensor: Broadcastable shape (1, 1, seq_len, dim / 2)
    """
    ndim = x.ndim
    assert 0 <= 2 < ndim

    # Get the shape of x (e.g., [B, H, N, D/2])
    # We want freqs_cis to be [1, 1, N, D/2]
    # We take the last 2 dims from freqs_cis and add 1s to the front
    shape = [1] * (ndim - 2) + list(freqs_cis.shape)
    return freqs_cis.view(*shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies the rotary position embedding to a single tensor (q, k, or v).

    Args:
        x (torch.Tensor): Input tensor (e.g., B, H, N, D)
        freqs_cis (torch.Tensor): Precomputed complex frequencies (N, D/2)

    Returns:
        torch.Tensor: The rotated tensor.
    """
    # 1. Reshape x to view the last dim as pairs (..., dim/2, 2)
    #    and cast to complex numbers (..., dim/2)
    #    (x.float() is for stability with float16)
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # 2. Reshape freqs_cis to be broadcastable
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)

    # 3. Apply the rotation by complex multiplication
    # (x + iy) * (cos + i*sin) = (x*cos - y*sin) + i*(x*sin + y*cos)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)

    # 4. Cast back to the original datatype
    return x_out.type_as(x)


# --- Base Attention Module (from your code) ---


class Attention(nn.Module):
    """A standard Multi-Head Attention module."""

    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# --- RoPE Attention Class You Can Use ---


class CrossAttentionRope(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        output_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rope_theta: float = 100.0,
        use_mlp_at_output: bool = False,
        name: str | None = None,
        init_to_zero: bool = True,
        *args,
        **kwargs,
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

        self.rope_theta = rope_theta

        # --- Caching Logic ---
        # We store the frequencies, but not as a persistent buffer.
        # This is the "dynamic caching" part.
        self.register_buffer("freqs_cis_cache", None, persistent=False)
        self._cached_seq_len = 0
        # --- End Caching Logic ---

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

    def _get_freqs_cis(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Retrieve or compute and cache the RoPE frequencies.
        """
        # Check if we need to re-compute the cache
        if self.freqs_cis_cache is None or seq_len > self._cached_seq_len:
            print(f"RoPE: Re-computing cache for seq_len={seq_len}")
            self.freqs_cis_cache = precompute_freqs_cis_1d(
                self.dim_head, seq_len, self.rope_theta
            ).to(device)
            self._cached_seq_len = seq_len

        # Return the cached frequencies, sliced to the current seq_len
        return self.freqs_cis_cache[:seq_len]

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        debug_dict: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads),
            (q, k, v),
        )

        freqs_cis = self._get_freqs_cis(k.shape[2], x.device)
        k = apply_rotary_emb(k, freqs_cis=freqs_cis)

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
            attn = attn.cpu()
            debug_dict[f"{self.name}/attn"] = attn.detach().cpu().float()
            debug_dict[f"{self.name}/out"] = out.detach().cpu().float()
        return out


# class SelfAttentionRope(CrossAttentionRope):
#     def forward(
#         self,
#         x: torch.Tensor,
#         context: torch.Tensor | None = None,
#         debug_dict: dict[str, Any] | None = None,
#     ) -> torch.Tensor:
#         assert context is None, "SelfAttentionRope does not take context"
#         q = self.to_q(x)
#         k = self.to_k(x)
#         v = self.to_v(x)
#         q, k, v = map(
#             lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads),
#             (q, k, v),
#         )

#         freqs_cis = self._get_freqs_cis(k.shape[2], x.device)
#         k = apply_rotary_emb(k, freqs_cis=freqs_cis)

#         with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
#             out = F.scaled_dot_product_attention(q, k, v)

#         out = rearrange(out, "b h l d -> b l (h d)")
#         out = self.to_out(out)
#         if debug_dict is not None:
#             # This should not run during training, its expensive!
#             val_shape = v.shape
#             eye = (
#                 torch.eye(val_shape[-2], device=v.device, dtype=v.dtype)
#                 .unsqueeze(0)
#                 .unsqueeze(0)
#                 .expand(val_shape[0], val_shape[1], -1, -1)
#             )
#             attn = F.scaled_dot_product_attention(q, k, eye)
#             attn = attn.cpu()
#             debug_dict[f"{self.name}/attn"] = attn
#             debug_dict[f"{self.name}/out"] = out
#         return out
