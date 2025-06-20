from util import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from memory import memory
import math
import os, random
from math import ceil, log2

_phi_cache = {}
@torch.no_grad()
def get_phi(L, K, device, dtype):
    key = (L, K, device, dtype)
    if key not in _phi_cache:
        _phi_cache[key] = get_spectral_filters(L, K, device=device, dtype=dtype)
    return _phi_cache[key]


class SpectralMoEHeads(nn.Module):
    r"""
    Multi-head variant of SpectralMoE.
    For H heads and K experts per head:
        x: (B, T, H * d_h)  →  y: (B, T, H * d_o)
    """
    def __init__(self, in_features: int, out_features: int,
                 num_experts: int, num_heads: int,
                 max_seq_len: int,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        assert in_features % num_heads == 0, "in_features must divide num_heads"
        assert out_features % num_heads == 0, "out_features must divide num_heads"

        d_h = in_features  // num_heads  # per-head embed dim
        d_o = out_features // num_heads  # per-head output dim
        self.H, self.K = num_heads, num_experts

        # (H, K, d_h, d_o)
        self.weight = nn.Parameter(
            torch.empty(num_heads, num_experts, d_h, d_o, device=device)
        )
        self.register_buffer("phi", get_phi(max_seq_len, num_experts, device, dtype), persistent=False)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # shared routing table ϕ  (T, K) – same for all heads
        # phi = get_spectral_filters(max_seq_len, num_experts,
        #                            device=device, dtype=dtype)

        # stash dims so callers can inspect
        self.in_features  = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, H*d_h)  –> y: (B, T, H*d_o)
        """
        B, T, _ = x.shape
        H, K, d_h, d_o = self.weight.shape
        assert T <= self.phi.size(0), "Sequence length exceeds max_seq_len"

        # (B, T, H, d_h)
        x = x.view(B, T, H, d_h).transpose(1, 2)         # (B, H, T, d_h)

        # expert outputs: (B, H, K, T, d_o)
        x = torch.einsum('b h t i , h k i o -> b h k t o', x, self.weight)

        # mix experts with fixed gates: (B, T, H, d_o)
        x = torch.einsum('b h k t o , t k -> b t h o', x, self.phi[:T])

        # concat heads back: (B, T, H*d_o)
        return x.reshape(B, T, H * d_o)

def get_hankel(seq_len: int) -> torch.Tensor:
    # Create a tensor with values from 1 to seq_len (inclusive)
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64).to('cuda')
    # Compute the outer sum
    i_plus_j = entries[:, None] + entries[None, :]
    # Calculate Z using element-wise operations
    Z = 2.0 / ((i_plus_j**3 - i_plus_j) + 1e-7)
    return Z


def get_spectral_filters(
    seq_len: int, 
    K: int, 
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # Compute the Hankel matrix using PyTorch
    Z = get_hankel(seq_len)
    # Compute eigenvalues and eigenvectors for symmetric matrices
    sigma, phi = torch.linalg.eigh(Z)
    # Select the largest K eigenvalues and corresponding eigenvectors
    sigma, phi = sigma[-K:], phi[:, -K:]
    # Scale the eigenvectors with the eigenvalues raised to 0.25 (broadcasting applies)
    phi = phi * (sigma ** 0.25)
    return phi.to(device=device, dtype=dtype)



if __name__ == '__main__':
    seq_len = 256
    n_embd = 64
    device = 'cuda'
    torch_dtype = torch.float32
    n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)
    phi = get_spectral_filters(seq_len, K, device, torch_dtype)

    layer = STU(
        n_embd=n_embd,
        idx=0,
        torch_dtype=torch_dtype,
        phi=phi,
        n=n,
        gating=True,
    ).to(device)

    x = torch.randn(2, seq_len, n_embd).to(device)

    out, mem = layer(x, None)
    print(out.shape)
    print(mem.shape if mem is not None else 0)