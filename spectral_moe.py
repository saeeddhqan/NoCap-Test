
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor


_phi_cache = {}
@torch.no_grad()
def get_phi(L, K, device, dtype):
    key = (L, K, device, dtype)
    if key not in _phi_cache:
        _phi_cache[key] = get_spectral_filters(L, K, device=device, dtype=dtype)
    return _phi_cache[key]


_decay_cache = {}
@torch.no_grad()
def get_decay(L, K, device, dtype):
    key = (L, K, device, dtype)
    if key not in _decay_cache:

        _decay_cache[key] = torch.tensor([[1 - (x / (L - 1)), x / (L - 1)] for x in range(L)]).to(device=device, dtype=dtype)
    return _decay_cache[key]

class SpectralMoEHeads(nn.Module):
    """
        Multi-head spectral moe.
    """
    def __init__(self, in_features: int, out_features: int,
                 num_experts: int, num_heads: int,
                 max_seq_len: int,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        assert in_features % num_heads == 0, "in_features must divide num_heads"
        assert out_features % num_heads == 0, "out_features must divide num_heads"

        d_h = in_features  // num_heads
        d_o = out_features // num_heads
        self.H, self.K = num_heads, num_experts

        self.weight = nn.Parameter(
            torch.empty(num_heads, num_experts, d_h, d_o, device=device)
        )
        # shared routing table - same for all heads, and instances
        self.register_buffer("phi", get_phi(max_seq_len, num_experts, device, dtype), persistent=False)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            # self.weight /= math.sqrt(self.K)
        self.diag = nn.Parameter(torch.zeros(num_experts, in_features, device=device, dtype=dtype))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, H*d_h) => y: (B, T, H*d_o)
        """
        B, T, _ = x.shape
        H, K, d_h, d_o = self.weight.shape
        assert T <= self.phi.size(0), "Sequence length exceeds max_seq_len"
        scores = self.phi[:T].view(1, T, K) + F.linear(x, self.diag) # (B, H, T, K)
        x = x.view(B, T, H, d_h).transpose(1, 2) # (B, H, T, d_h)
        # expert outputs
        x = torch.einsum('b h t i , h k i o -> b h k t o', x, self.weight) # (B, H, K, T, d_o)
        # mix experts with fixed gates
        x = torch.einsum('b h k t o , b t k -> b t h o', x, scores) # (B, T, H, d_o)
        return x.reshape(B, T, H * d_o)

def get_hankel(seq_len: int) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64).to('cuda')
    i_plus_j = entries[:, None] + entries[None, :]
    # position-wise decay
    Z = 2.0 / ((i_plus_j**3 - i_plus_j) + 1e-7)
    return Z


def get_spectral_filters(
    seq_len: int, 
    K: int, 
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    Z = get_hankel(seq_len)
    sigma, phi = torch.linalg.eigh(Z)
    sigma, phi = sigma[-K:], phi[:, -K:]
    phi = phi * (sigma ** 0.25)
    return phi.to(device=device, dtype=dtype)



if __name__ == '__main__':
    pass
    # this too shall pass