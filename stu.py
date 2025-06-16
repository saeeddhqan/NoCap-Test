from util import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from memory import memory
import math
import os, random
try:
    from flashfftconv import FlashFFTConv
    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False
from math import ceil, log2


@torch.no_grad()
def precompute_v_fft(phi, n_fft):
    """
    phi:  (L, K)   real filters
    n_fft: power-of-two ≥ L
    returns v_fft: (K, F) complex64   with F = n_fft//2 + 1
    """
    L, K = phi.shape
    v_pad = F.pad(phi.T, (0, n_fft - L))
    return torch.fft.rfft(v_pad.to(torch.float32), n=n_fft, dim=1)


class STU(nn.Module):
    def __init__(
        self,
        n_embd,
        torch_dtype,
        is_causal: bool,
        phi, n, idx,
        K,
        use_gating: bool = False,
        num_slots: int = 16,
    ) -> None:
        super(STU, self).__init__()
        n_fft = 1 << math.ceil(math.log2(n + phi.size(0) - 1))
        self.register_buffer("phi", precompute_v_fft(phi, n_fft), persistent=False)

        self.n = n
        self.K = K
        self.dim = n_embd
        self.use_gating = use_gating
        self.dtype = torch_dtype
        self.flash_fft = (
            FlashFFTConv(self.n * 2, dtype=torch.bfloat16) if
            flash_fft_available
            else None
        ) # x2 for causality
        self.M_plus = nn.Parameter(
            torch.randn(self.K, self.dim, self.dim, dtype=torch_dtype) * 1e-5
        )
        self.M_minus = nn.Parameter(
            torch.randn(self.K, self.dim, self.dim, dtype=torch_dtype) * 1e-5
        )
        # self.gate  = nn.Linear(self.dim, self.dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate, x = self.gate(x)chunk(2, dim=-1)
        U_plus, U_minus = causal_fft_convolve_fast(x, self.phi, block=4)
        y = (
            torch.tensordot(U_plus.to(self.dtype),  self.M_plus,  dims=([2,3],[0,1])) +
            torch.tensordot(U_minus.to(self.dtype), self.M_minus, dims=([2,3],[0,1]))
        )
        return y



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
    # Return the tensor on the desired device and with the desired data type
    # L, K = phi.shape
    # n_fft = 1 << math.ceil(math.log2(L + L - 1))
    # v_pad = F.pad(phi.T, (0, n_fft - L))
    # v_fft = torch.fft.rfft(v_pad.to(torch.float32), n=n_fft, dim=1).to(device=device)
    # print(v_fft.shape)
    # return v_fft
    return phi.to(device=device, dtype=dtype)

def convolve(u: torch.Tensor, v: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1

    _, K = v.shape
    sgn = sgn.unsqueeze(-1)
    u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)
    U_plus, U_minus = torch.unbind(torch.fft.irfft(
        torch.fft.rfft(v.view(1, -1, K, 1, 1).to(torch.float32), n=n, dim=1) * #ufake
        torch.fft.rfft(
            torch.stack([u, u * sgn], dim=-1).to(torch.float32),
            n=n,
            dim=1,
        )
        , n=n, dim=1)[:, :seq_len]
    , dim=-1)
    print(U_plus.shape, U_minus.shape, sgn.shape)

    U_minus = U_minus * sgn
    return U_plus.to(u.dtype), U_minus.to(u.dtype)


def causal_fft_convolve_mem_efficient(
    u: torch.Tensor,          # (B, T, C)
    v: torch.Tensor,          # (L, K)
    n: int | None = None,     # optional block size along K (for huge K)
    *, block: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-friendly causal FFT convolution.
    Returns (Y_plus, Y_minus) with shape (B, T, K, C).
    """
    B, T, C = u.shape
    L, K    = v.shape
    n_fft   = 1 << math.ceil(math.log2(T + L - 1))
    device, dtype = u.device, u.dtype


    sgn_full = torch.ones(n_fft, device=device)      # length n_fft
    sgn_full[1::2] = -1                              # (-1)^n
    sgn_T = sgn_full[:T]                             # slice for the output later

    u_pad  = F.pad(u, (0, 0, 0, n_fft - T))                          # (B, n_fft, C)
    u_fft  = torch.fft.rfft(u_pad.to(torch.float32), n=n_fft, dim=1) # (B, F, C)

    u_minus_pad = u_pad * sgn_full.view(1, -1, 1)                    # broadcast
    us_fft = torch.fft.rfft(u_minus_pad.to(torch.float32), n=n_fft, dim=1)

    v_pad = F.pad(v.T, (0, n_fft - L))                               # (K, n_fft)
    v_fft = torch.fft.rfft(v_pad.to(torch.float32), n=n_fft, dim=1)  # (K, F)

    # reshape for broadcasting
    Freq = v_fft.shape[1]
    u_fft  = u_fft.view(B, 1, Freq, C)        # (B, 1, F, C)
    us_fft = us_fft.view(B, 1, Freq, C)       # (B, 1, F, C)
    v_fft  = v_fft.view(1, K, Freq, 1)        # (1, K, F, 1)

    Yp, Ym = [], []
    step   = block or K
    for k0 in range(0, K, step):
        k1 = min(k0 + step, K)
        vv = v_fft[:, k0:k1]                                 # (1, b, F, 1)

        y_p = torch.fft.irfft(vv * u_fft,  n=n_fft, dim=2)[:, :, :T]  # (B, b, T, C)
        y_m = torch.fft.irfft(vv * us_fft, n=n_fft, dim=2)[:, :, :T]
        y_m = y_m * sgn_T.view(1, 1, T, 1)                              # remove (-1)^n

        Yp.append(y_p)
        Ym.append(y_m)

    Y_plus  = torch.cat(Yp, dim=1).permute(0, 2, 1, 3).to(dtype)   # (B, T, K, C)
    Y_minus = torch.cat(Ym, dim=1).permute(0, 2, 1, 3).to(dtype)

    return Y_plus, Y_minus



def causal_fft_convolve_fast(
    u: torch.Tensor,          # (B, T, C), dtype bf16/fp16/fp32
    v_fft: torch.Tensor,      # (K, F)  — pre-computed once!
    block: int|None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C = u.shape
    K, Freq = v_fft.shape
    n_fft   = (Freq - 1) * 2
    device, dtype = u.device, u.dtype

    # ---------- forward FFTs ----------
    u_pad  = F.pad(u, (0, 0, 0, n_fft - T))                    # (B, n_fft, C)
    u_fft  = torch.fft.rfft(u_pad.float(), n=n_fft, dim=1)     # (B, F, C)

    sgn_full = torch.ones(n_fft, device=device)
    sgn_full[1::2] = -1                                        # (-1)^n
    u_minus_pad = u_pad * sgn_full.view(1, -1, 1)
    us_fft = torch.fft.rfft(u_minus_pad.float(), n=n_fft, dim=1)

    # ---------- point-wise multiply (no loop over K) ----------
    v_fft_bc = v_fft.view(1, K, Freq, 1)                       # (1, K, F, 1)
    spec_p   = v_fft_bc * u_fft .unsqueeze(1)                  # (B, K, F, C)
    spec_m   = v_fft_bc * us_fft.unsqueeze(1)

    # ---------- inverse FFTs ----------
    y_p = torch.fft.irfft(spec_p, n=n_fft, dim=2)[..., :T, :]  # (B, K, T, C)
    y_m = torch.fft.irfft(spec_m, n=n_fft, dim=2)[..., :T, :]

    # remove the (-1)^n modulation to get the strictly causal Y_minus
    sgn_T = sgn_full[:T].view(1, 1, T, 1)
    y_m   = y_m * sgn_T

    # ---------- final layout ----------
    Y_plus  = y_p.permute(0, 2, 1, 3).to(dtype)                # (B, T, K, C)
    Y_minus = y_m.permute(0, 2, 1, 3).to(dtype)
    return Y_plus, Y_minus


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