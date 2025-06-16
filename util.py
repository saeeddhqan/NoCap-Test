import math
import torch
from torch import Tensor
nn = torch.nn
F = nn.functional


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class GatedMLP(nn.Module):
    def __init__(self, dim: int = 1024, expansion_factor: int = 2):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.grow = Linear(dim, 2 * hidden, bias=False)
        self.shrink = Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.grow.weight.normal_(std=dim ** -0.5)
            self.shrink.weight.normal_(std=hidden ** -0.5)

    def forward(self, x: Tensor) -> Tensor:
        gate, x = self.grow(x).chunk(2, dim=-1)
        x = F.gelu(gate) * x
        return self.shrink(x)

class SquishSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        silu = F.gelu(x)
        return silu
        # return torch.where(x < 0, silu, silu ** 2)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = Linear(dim, 4 * dim)
        self.c_proj = Linear(4 * dim, dim)
        self.act = SquishSiLU()
        # with torch.no_grad():
            # nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
            # nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.01)
            # self.c_proj.weight.detach().zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        # x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def norm1(x):
    return F.rms_norm(x, (x.size(-1),))

def norm2(x):
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
    return x / rms

if hasattr(F, 'rms_norm'):
    norm = norm1
else:
    norm = norm2

def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return (
        1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x)))

def generate_synthetic_data(seqlen, dim, nsamples: int = 15):
    x_data = torch.randn(nsamples, 1, seqlen, dim)
    y_data = torch.randint(128, (nsamples, 1, seqlen))  # dummy targets
    return list(zip(x_data, y_data))